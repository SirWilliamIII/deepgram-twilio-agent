"""Deepgram Text-to-Speech integration."""

import logging
from typing import AsyncIterator

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class DeepgramTTS:
    """Text-to-speech using Deepgram's Aura API."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "DeepgramTTS":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech audio.

        Args:
            text: The text to speak.

        Returns:
            Audio data as bytes (mulaw encoded, 8kHz).
        """
        if not self._client:
            self._client = httpx.AsyncClient(timeout=30.0)

        headers = {
            "Authorization": f"Token {settings.deepgram_api_key}",
            "Content-Type": "application/json",
        }

        payload = {"text": text}

        logger.debug(f"TTS request: {text[:50]}...")

        response = await self._client.post(
            settings.deepgram_tts_url,
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            logger.error(f"TTS error: {response.status_code} - {response.text}")
            raise Exception(f"TTS failed: {response.status_code}")

        logger.debug(f"TTS response: {len(response.content)} bytes")
        return response.content

    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """
        Convert text to speech with streaming response.

        This allows sending audio chunks as they become available,
        reducing time-to-first-audio.

        Args:
            text: The text to speak.

        Yields:
            Audio data chunks (mulaw encoded, 8kHz).
        """
        if not self._client:
            self._client = httpx.AsyncClient(timeout=30.0)

        headers = {
            "Authorization": f"Token {settings.deepgram_api_key}",
            "Content-Type": "application/json",
        }

        payload = {"text": text}

        logger.debug(f"TTS streaming request: {text[:50]}...")

        async with self._client.stream(
            "POST",
            settings.deepgram_tts_url,
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"TTS error: {response.status_code} - {error_text}")
                raise Exception(f"TTS failed: {response.status_code}")

            async for chunk in response.aiter_bytes(chunk_size=1024):
                yield chunk

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance for reuse
_tts_instance: DeepgramTTS | None = None


async def get_tts() -> DeepgramTTS:
    """Get or create the TTS singleton."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = DeepgramTTS()
    return _tts_instance
