"""Deepgram Speech-to-Text integration via WebSocket."""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import websockets
from websockets.asyncio.client import ClientConnection

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptEvent:
    """A transcription event from Deepgram."""

    text: str
    is_final: bool
    speech_final: bool  # True when utterance is complete (end of turn)
    confidence: float


class DeepgramSTT:
    """Real-time speech-to-text using Deepgram's WebSocket API."""

    def __init__(self):
        self._ws: ClientConnection | None = None
        self._receive_task: asyncio.Task | None = None
        self._transcript_callback: Callable[[TranscriptEvent], None] | None = None
        self._connected = asyncio.Event()
        self._closed = False

    async def connect(self, on_transcript: Callable[[TranscriptEvent], None]) -> None:
        """
        Connect to Deepgram STT WebSocket.

        Args:
            on_transcript: Callback function called for each transcript event.
        """
        self._transcript_callback = on_transcript
        self._closed = False

        # Validate API key is present
        if not settings.deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY is not set! Check your .env file.")

        # Log key prefix for debugging (without exposing full key)
        key_preview = settings.deepgram_api_key[:8] + "..." if len(settings.deepgram_api_key) > 8 else "???"
        logger.info(f"Using Deepgram API key: {key_preview}")

        headers = {"Authorization": f"Token {settings.deepgram_api_key}"}

        logger.info("Connecting to Deepgram STT...")
        logger.debug(f"STT URL: {settings.deepgram_stt_url}")

        try:
            self._ws = await websockets.connect(
                settings.deepgram_stt_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
        except websockets.exceptions.InvalidStatus as e:
            if e.response.status_code == 403:
                logger.error(
                    "Deepgram returned 403 Forbidden. "
                    "This usually means the API key is invalid or not authorized. "
                    f"Key used: {key_preview}"
                )
            raise

        self._connected.set()
        logger.info("Connected to Deepgram STT")

        # Start receiving transcripts in background
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        """Background task to receive and process transcripts."""
        if not self._ws:
            return

        try:
            async for message in self._ws:
                if self._closed:
                    break

                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    logger.debug(f"Deepgram message: {msg_type}")

                    event = self._parse_transcript(data)
                    if event and self._transcript_callback:
                        logger.info(f"Transcript: '{event.text}' (final={event.is_final}, speech_final={event.speech_final})")
                        self._transcript_callback(event)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse Deepgram message: {message[:100]}")
                except Exception as e:
                    logger.error(f"Error processing transcript: {e}")

        except websockets.ConnectionClosed:
            logger.info("Deepgram STT connection closed")
        except Exception as e:
            logger.error(f"Error in STT receive loop: {e}")

    def _parse_transcript(self, data: dict) -> TranscriptEvent | None:
        """Parse a Deepgram response into a TranscriptEvent."""
        msg_type = data.get("type")

        if msg_type == "Results":
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])

            if not alternatives:
                return None

            transcript = alternatives[0].get("transcript", "").strip()
            if not transcript:
                return None

            return TranscriptEvent(
                text=transcript,
                is_final=data.get("is_final", False),
                speech_final=data.get("speech_final", False),
                confidence=alternatives[0].get("confidence", 0.0),
            )

        elif msg_type == "UtteranceEnd":
            # This signals the end of a speech turn
            logger.debug("Utterance end detected")
            return None

        return None

    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send audio data to Deepgram for transcription.

        Args:
            audio_data: Raw audio bytes (mulaw encoded, 8kHz).
        """
        await self._connected.wait()

        if self._ws and not self._closed:
            try:
                await self._ws.send(audio_data)
            except websockets.ConnectionClosed:
                logger.warning("Cannot send audio: connection closed")

    _audio_chunks_sent: int = 0

    async def send_audio_base64(self, audio_base64: str) -> None:
        """
        Send base64-encoded audio data to Deepgram.

        Args:
            audio_base64: Base64-encoded audio (from Twilio).
        """
        audio_data = base64.b64decode(audio_base64)
        self._audio_chunks_sent += 1
        if self._audio_chunks_sent % 100 == 0:
            logger.info(f"Audio chunks sent to Deepgram: {self._audio_chunks_sent}")
        await self.send_audio(audio_data)

    async def close(self) -> None:
        """Close the Deepgram connection."""
        self._closed = True

        if self._ws:
            try:
                # Send close message to Deepgram
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await self._ws.close()
            except Exception as e:
                logger.warning(f"Error closing Deepgram connection: {e}")
            finally:
                self._ws = None

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        self._connected.clear()
        logger.info("Deepgram STT connection closed")
