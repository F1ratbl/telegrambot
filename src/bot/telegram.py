from __future__ import annotations

import logging
from typing import Any

from src.config import Settings


logger = logging.getLogger(__name__)


class TelegramClient:
    api_base = "https://api.telegram.org"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def send_message(
        self,
        chat_id: int | str,
        text: str,
        reply_to_message_id: int | None = None,
    ) -> None:
        if not self.settings.telegram_bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN is not configured; skipping Telegram send.")
            return

        chunks = list(_split_telegram_text(text))
        for index, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
            }
            if index == 0 and reply_to_message_id is not None:
                payload["reply_to_message_id"] = reply_to_message_id
                payload["allow_sending_without_reply"] = True
            self._post("sendMessage", payload)

    def _post(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        import requests

        url = f"{self.api_base}/bot{self.settings.telegram_bot_token}/{method}"
        response = requests.post(
            url,
            json=payload,
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API returned an error: {data}")
        return data


def _split_telegram_text(text: str, limit: int = 3900) -> list[str]:
    clean = text.strip() or "Cevap olusturamadim."
    if len(clean) <= limit:
        return [clean]

    chunks: list[str] = []
    remaining = clean
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks
