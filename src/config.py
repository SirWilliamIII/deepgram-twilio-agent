"""Configuration management for the phone agent."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    deepgram_api_key: str
    openai_api_key: str

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Agent
    agent_name: str = "AI Assistant"

    # Deepgram STT settings
    stt_model: str = "nova-2"
    stt_language: str = "en-US"

    # Deepgram TTS settings
    tts_model: str = "aura-asteria-en"  # Female voice, natural sounding
    tts_sample_rate: int = 8000  # Match Twilio's mulaw format

    # OpenAI settings
    openai_model: str = "gpt-4o-mini"
    max_tokens: int = 300  # Keep responses concise for phone

    # Paths
    transcripts_dir: Path = Path("transcripts")
    system_prompt_path: Path = Path("system_prompt.md")

    @property
    def deepgram_stt_url(self) -> str:
        """WebSocket URL for Deepgram STT."""
        return (
            f"wss://api.deepgram.com/v1/listen"
            f"?model={self.stt_model}"
            f"&language={self.stt_language}"
            f"&encoding=mulaw"
            f"&sample_rate=8000"
            f"&channels=1"
            f"&punctuate=true"
            f"&interim_results=true"
            f"&utterance_end_ms=1000"
            f"&vad_events=true"
            f"&endpointing=300"
        )

    @property
    def deepgram_tts_url(self) -> str:
        """HTTP URL for Deepgram TTS."""
        return f"https://api.deepgram.com/v1/speak?model={self.tts_model}&encoding=mulaw&sample_rate={self.tts_sample_rate}"


settings = Settings()
