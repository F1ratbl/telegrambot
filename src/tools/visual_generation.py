from __future__ import annotations

import logging
from typing import Any

from src.config import Settings


logger = logging.getLogger(__name__)


class EconomyVisualGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    def parse_request(self, text: str) -> str | None:
        lowered = text.lower()
        markers = [
            "infografik",
            "görsel",
            "gorsel",
            "şema",
            "sema",
            "görselle anlat",
            "gorselle anlat",
            "resim oluştur",
            "resim olustur",
        ]
        if not any(marker in lowered for marker in markers):
            return None
        return text.strip()

    def generate(self, request_text: str) -> tuple[bytes, str]:
        if not self.settings.image_enabled:
            raise RuntimeError("Gorsel uretimi icin GOOGLE_API_KEY ve IMAGE_GENERATION_ENABLED=true gerekli.")

        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.settings.google_api_key)

        prompt = self._build_prompt(request_text)
        response = self._client.models.generate_content(
            model=self.settings.gemini_image_model,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        image = self._extract_image_bytes(response)
        if not image:
            raise RuntimeError("Gemini gorsel uretmedi.")
        return image, "Ekonomi gorseli"

    def _build_prompt(self, request_text: str) -> str:
        return (
            "Turkce ekonomi ve finans egitimi icin sade, profesyonel bir infografik uret. "
            "Gorselde yaniltici fiyat, sahte grafik veya kesin yatirim tavsiyesi olmasin. "
            "Kisa basliklar, net oklar ve finansal analiz mantigina uygun akış kullan. "
            "Telegram'da okunacak sekilde temiz, modern ve yuksek kontrastli tasarla. "
            f"Konu: {request_text}"
        )

    def _extract_image_bytes(self, response: Any) -> bytes | None:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline_data = getattr(part, "inline_data", None)
                data = getattr(inline_data, "data", None)
                if data:
                    return data
        return None
