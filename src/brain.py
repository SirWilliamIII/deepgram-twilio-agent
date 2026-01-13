"""OpenAI-powered conversation brain."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from openai import AsyncOpenAI

from src.config import settings

logger = logging.getLogger(__name__)

# Default system prompt if file doesn't exist
DEFAULT_SYSTEM_PROMPT = """You are a friendly and helpful phone assistant. You answer calls on behalf of the person whose phone this is.

Keep your responses conversational and concise - this is a phone call, not a text chat.
- Use short sentences
- Be warm but professional
- Don't use bullet points or formatting
- Respond naturally as you would in a real phone conversation
- If you don't know something, offer to take a message

When ending a call, say goodbye naturally."""


def load_system_prompt() -> str:
    """Load the system prompt from file or use default."""
    prompt_path = Path(settings.system_prompt_path)

    if prompt_path.exists():
        logger.info(f"Loading system prompt from {prompt_path}")
        return prompt_path.read_text().strip()
    else:
        logger.info("Using default system prompt")
        return DEFAULT_SYSTEM_PROMPT


@dataclass
class Message:
    """A conversation message."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    """Manages conversation history for a call."""

    messages: list[Message] = field(default_factory=list)
    system_prompt: str = field(default_factory=load_system_prompt)

    def add_user_message(self, text: str) -> None:
        """Add a user (caller) message to history."""
        # Merge consecutive user messages (API requires alternating roles)
        if self.messages and self.messages[-1].role == "user":
            self.messages[-1].content += " " + text
        else:
            self.messages.append(Message(role="user", content=text))

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant response to history."""
        self.messages.append(Message(role="assistant", content=text))

    def to_api_format(self) -> list[dict]:
        """Convert messages to OpenAI API format (includes system message)."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend({"role": m.role, "content": m.content} for m in self.messages)
        return messages

    def get_transcript(self) -> str:
        """Get a formatted transcript of the conversation."""
        lines = []
        for msg in self.messages:
            speaker = "Caller" if msg.role == "user" else "Assistant"
            lines.append(f"{speaker}: {msg.content}")
        return "\n".join(lines)


class OpenAIBrain:
    """OpenAI-powered conversation handler with streaming."""

    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._sentence_pattern = re.compile(r"([.!?]+)\s*")

    async def respond(self, conversation: Conversation) -> str:
        """
        Generate a complete response (non-streaming).

        Args:
            conversation: The conversation history.

        Returns:
            The assistant's response text.
        """
        response = await self._client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=settings.max_tokens,
            messages=conversation.to_api_format(),
        )

        return response.choices[0].message.content

    async def respond_streaming(
        self, conversation: Conversation
    ) -> AsyncIterator[str]:
        """
        Generate a response with streaming.

        Args:
            conversation: The conversation history.

        Yields:
            Text chunks as they arrive.
        """
        stream = await self._client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=settings.max_tokens,
            messages=conversation.to_api_format(),
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def respond_by_sentence(
        self, conversation: Conversation
    ) -> AsyncIterator[str]:
        """
        Generate a response, yielding complete sentences.

        This is optimized for TTS - sending complete sentences
        produces better prosody than arbitrary chunks.

        Args:
            conversation: The conversation history.

        Yields:
            Complete sentences as they become available.
        """
        buffer = ""

        stream = await self._client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=settings.max_tokens,
            messages=conversation.to_api_format(),
            stream=True,
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                buffer += content

                # Check for complete sentences
                while True:
                    match = self._sentence_pattern.search(buffer)
                    if match:
                        # Found end of sentence
                        end_pos = match.end()
                        sentence = buffer[:end_pos].strip()
                        buffer = buffer[end_pos:]

                        if sentence:
                            yield sentence
                    else:
                        break

        # Yield any remaining text
        if buffer.strip():
            yield buffer.strip()


# Singleton instance
_brain_instance: OpenAIBrain | None = None


def get_brain() -> OpenAIBrain:
    """Get or create the brain singleton."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = OpenAIBrain()
    return _brain_instance
