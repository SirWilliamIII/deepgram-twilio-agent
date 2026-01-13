"""Call handler - orchestrates the full audio pipeline for a single call."""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from src.brain import Conversation, get_brain
from src.config import settings
from src.stt import DeepgramSTT, TranscriptEvent
from src.tts import get_tts

logger = logging.getLogger(__name__)


class CallState(Enum):
    """State machine for a phone call."""

    CONNECTING = "connecting"
    GREETING = "greeting"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ENDED = "ended"


@dataclass
class CallMetadata:
    """Metadata about the current call."""

    call_sid: str = ""
    stream_sid: str = ""
    caller: str = ""
    called: str = ""
    start_time: datetime = field(default_factory=datetime.now)


class CallHandler:
    """
    Handles the full lifecycle of a single phone call.

    Orchestrates:
    - Twilio WebSocket for audio I/O
    - Deepgram STT for speech recognition
    - Claude for response generation
    - Deepgram TTS for speech synthesis
    """

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.state = CallState.CONNECTING
        self.metadata = CallMetadata()
        self.conversation = Conversation()

        # Components
        self.stt = DeepgramSTT()
        self.brain = get_brain()

        # State management
        self._current_utterance = ""
        self._is_speaking = False
        self._speech_queue: asyncio.Queue[str] = asyncio.Queue()
        self._stop_speaking = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def handle(self) -> None:
        """Main entry point - handle the complete call lifecycle."""
        try:
            logger.info("Starting call handler")

            # Connect to STT
            await self.stt.connect(on_transcript=self._on_transcript)

            # Start background tasks
            self._tasks.append(asyncio.create_task(self._speech_sender()))

            # Process incoming messages from Twilio
            await self._process_twilio_messages()

        except Exception as e:
            logger.error(f"Call handler error: {e}", exc_info=True)
        finally:
            await self._cleanup()

    async def _process_twilio_messages(self) -> None:
        """Process incoming WebSocket messages from Twilio."""
        async for message in self.ws.iter_text():
            try:
                data = json.loads(message)
                await self._handle_twilio_message(data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from Twilio: {message[:100]}")
            except Exception as e:
                logger.error(f"Error handling Twilio message: {e}")

    async def _handle_twilio_message(self, data: dict) -> None:
        """Handle a single message from Twilio."""
        event_type = data.get("event")

        if event_type == "connected":
            logger.info("Twilio connected")

        elif event_type == "start":
            # Call is starting - extract metadata
            start_data = data.get("start", {})
            self.metadata.call_sid = start_data.get("callSid", "")
            self.metadata.stream_sid = start_data.get("streamSid", "")

            custom_params = start_data.get("customParameters", {})
            self.metadata.caller = custom_params.get("caller", "Unknown")
            self.metadata.called = custom_params.get("called", "")

            logger.info(
                f"Call started: {self.metadata.call_sid} "
                f"from {self.metadata.caller}"
            )

            self.state = CallState.GREETING
            # Send initial greeting
            await self._speak_greeting()

        elif event_type == "media":
            # Audio data from caller
            media = data.get("media", {})
            payload = media.get("payload", "")

            if payload:
                # Forward to STT
                await self.stt.send_audio_base64(payload)

        elif event_type == "stop":
            logger.info("Call ended by Twilio")
            self.state = CallState.ENDED

        elif event_type == "mark":
            # A mark we sent was reached (audio playback milestone)
            mark_name = data.get("mark", {}).get("name", "")
            logger.debug(f"Mark reached: {mark_name}")

            if mark_name == "greeting_end":
                self.state = CallState.LISTENING

    def _on_transcript(self, event: TranscriptEvent) -> None:
        """Callback when STT produces a transcript."""
        if self.state == CallState.ENDED:
            return

        logger.debug(
            f"Transcript: '{event.text}' "
            f"(final={event.is_final}, speech_final={event.speech_final})"
        )

        if event.is_final:
            self._current_utterance = event.text

            if event.speech_final:
                # End of caller's turn - process the complete utterance
                utterance = self._current_utterance.strip()
                if utterance:
                    asyncio.create_task(self._process_utterance(utterance))
                self._current_utterance = ""
        else:
            # Interim result - if we're speaking and caller interrupts
            if self._is_speaking and event.text.strip():
                # Barge-in detected
                logger.info(f"Barge-in detected: '{event.text}'")
                self._stop_speaking.set()

    async def _process_utterance(self, utterance: str) -> None:
        """Process a complete utterance from the caller."""
        if not utterance:
            return

        logger.info(f"Processing: '{utterance}'")
        self.state = CallState.PROCESSING

        # Add to conversation
        self.conversation.add_user_message(utterance)

        # Generate response
        try:
            full_response = ""

            async for sentence in self.brain.respond_by_sentence(self.conversation):
                full_response += sentence + " "
                # Queue sentence for TTS
                await self._speech_queue.put(sentence)

            # Record assistant response in conversation
            self.conversation.add_assistant_message(full_response.strip())

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            await self._speech_queue.put(
                "I'm sorry, I'm having trouble understanding. "
                "Could you please repeat that?"
            )

    async def _speak_greeting(self) -> None:
        """Speak the initial greeting."""
        greeting = f"Hello, this is {settings.agent_name}. How can I help you?"
        await self._speech_queue.put(greeting)
        self.conversation.add_assistant_message(greeting)

    async def _speech_sender(self) -> None:
        """Background task that sends TTS audio to Twilio."""
        tts = await get_tts()

        while self.state != CallState.ENDED:
            try:
                # Wait for text to speak
                text = await asyncio.wait_for(
                    self._speech_queue.get(),
                    timeout=1.0,
                )

                self._is_speaking = True
                self._stop_speaking.clear()
                self.state = CallState.SPEAKING

                logger.info(f"Speaking: '{text}'")

                # Generate and send TTS audio
                audio_data = await tts.synthesize(text)
                await self._send_audio_to_twilio(audio_data)

                self._is_speaking = False

                # If queue is empty, go back to listening
                if self._speech_queue.empty():
                    self.state = CallState.LISTENING

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in speech sender: {e}")
                self._is_speaking = False

    async def _send_audio_to_twilio(self, audio_data: bytes) -> None:
        """Send audio data to Twilio Media Streams."""
        # Split into chunks and send
        chunk_size = 640  # ~40ms of audio at 8kHz mulaw
        offset = 0

        while offset < len(audio_data):
            if self._stop_speaking.is_set():
                logger.info("Speech interrupted, stopping audio")
                # Clear the queue
                while not self._speech_queue.empty():
                    try:
                        self._speech_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                # Send clear message to Twilio
                await self._send_twilio_clear()
                return

            chunk = audio_data[offset : offset + chunk_size]
            offset += chunk_size

            # Encode and send
            payload = base64.b64encode(chunk).decode("utf-8")
            message = {
                "event": "media",
                "streamSid": self.metadata.stream_sid,
                "media": {"payload": payload},
            }

            try:
                await self.ws.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send audio: {e}")
                return

            # Small delay to pace audio
            await asyncio.sleep(0.02)

    async def _send_twilio_clear(self) -> None:
        """Send clear message to stop Twilio audio playback."""
        message = {"event": "clear", "streamSid": self.metadata.stream_sid}
        try:
            await self.ws.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send clear: {e}")

    async def _send_twilio_mark(self, name: str) -> None:
        """Send a mark to Twilio to track audio playback position."""
        message = {
            "event": "mark",
            "streamSid": self.metadata.stream_sid,
            "mark": {"name": name},
        }
        try:
            await self.ws.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send mark: {e}")

    async def _cleanup(self) -> None:
        """Clean up resources when call ends."""
        logger.info("Cleaning up call handler")
        self.state = CallState.ENDED

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close STT
        await self.stt.close()

        # Save transcript
        await self._save_transcript()

    async def _save_transcript(self) -> None:
        """Save the call transcript to a file."""
        if not self.conversation.messages:
            return

        transcript_dir = Path(settings.transcripts_dir)
        transcript_dir.mkdir(exist_ok=True)

        timestamp = self.metadata.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"call_{timestamp}_{self.metadata.call_sid[:8]}.txt"
        filepath = transcript_dir / filename

        content = (
            f"Call Transcript\n"
            f"===============\n"
            f"Time: {self.metadata.start_time.isoformat()}\n"
            f"Caller: {self.metadata.caller}\n"
            f"Call SID: {self.metadata.call_sid}\n"
            f"\n"
            f"Conversation:\n"
            f"-------------\n"
            f"{self.conversation.get_transcript()}\n"
        )

        filepath.write_text(content)
        logger.info(f"Transcript saved to {filepath}")
