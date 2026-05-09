from __future__ import annotations

import logging
from typing import Any

from src.config import Settings


logger = logging.getLogger(__name__)


class SpeechServiceError(RuntimeError):
    def __init__(self, provider: str, status_code: int, message: str) -> None:
        super().__init__(f"{provider} API error {status_code}: {message}")
        self.provider = provider
        self.status_code = status_code
        self.message = message


class SpeechToTextClient:
    api_url = "https://api.deepgram.com/v1/listen"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.deepgram_api_key)

    def transcribe(self, audio: bytes, mimetype: str | None = None) -> str:
        if not self.settings.deepgram_api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not configured.")
        if not audio:
            raise RuntimeError("Audio payload is empty.")

        import requests

        response = requests.post(
            self.api_url,
            params={
                "model": self.settings.deepgram_model,
                "language": self.settings.deepgram_language,
                "smart_format": "true",
            },
            headers={
                "Authorization": f"Token {self.settings.deepgram_api_key}",
                "Content-Type": mimetype or "application/octet-stream",
            },
            data=audio,
            timeout=self.settings.request_timeout_seconds,
        )
        _raise_for_status(response, "Deepgram")
        return _extract_transcript(response.json())


class TextToSpeechClient:
    api_base = "https://api.elevenlabs.io/v1"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.elevenlabs_api_key and self.settings.elevenlabs_voice_id)

    def synthesize(self, text: str) -> bytes:
        if not self.settings.elevenlabs_api_key or not self.settings.elevenlabs_voice_id:
            raise RuntimeError("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be configured.")
        clean_text = text.strip()
        if not clean_text:
            raise RuntimeError("TTS text is empty.")

        import requests

        url = f"{self.api_base}/text-to-speech/{self.settings.elevenlabs_voice_id}"
        response = requests.post(
            url,
            params={"output_format": self.settings.elevenlabs_output_format},
            headers={
                "xi-api-key": self.settings.elevenlabs_api_key,
                "Content-Type": "application/json",
                "Accept": _audio_accept_header(self.settings.elevenlabs_output_format),
            },
            json={
                "text": clean_text,
                "model_id": self.settings.elevenlabs_model_id,
                "language_code": self.settings.deepgram_language,
            },
            timeout=self.settings.request_timeout_seconds,
        )
        _raise_for_status(response, "ElevenLabs")
        return response.content


def _raise_for_status(response: Any, provider: str) -> None:
    if 200 <= response.status_code < 300:
        return
    raise SpeechServiceError(provider, response.status_code, _response_error_message(response))


def _response_error_message(response: Any) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (getattr(response, "text", "") or "No response body.").strip()[:700]

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict):
            message = detail.get("message") or detail.get("status")
            if message:
                return str(message)
        message = payload.get("message") or payload.get("error")
        if message:
            return str(message)
    return str(payload)[:700]


def _audio_accept_header(output_format: str | None) -> str:
    if (output_format or "").lower().startswith("opus"):
        return "audio/ogg"
    return "audio/mpeg"


def _extract_transcript(payload: dict[str, Any]) -> str:
    channels = ((payload.get("results") or {}).get("channels")) or []
    if not channels:
        return ""
    alternatives = channels[0].get("alternatives") or []
    if not alternatives:
        return ""
    transcript = alternatives[0].get("transcript") or ""
    return transcript.strip()
