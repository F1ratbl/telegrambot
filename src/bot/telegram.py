from __future__ import annotations

import html
from io import BytesIO
import logging
import re
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

        chunks = list(_split_telegram_text(_sanitize_telegram_text(text)))
        for index, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
                "parse_mode": "HTML",
            }
            if index == 0 and reply_to_message_id is not None:
                payload["reply_to_message_id"] = reply_to_message_id
                payload["allow_sending_without_reply"] = True
            self._post("sendMessage", payload)

    def send_voice(
        self,
        chat_id: int | str,
        audio: bytes,
        reply_to_message_id: int | None = None,
        caption: str | None = None,
    ) -> None:
        if not self.settings.telegram_bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN is not configured; skipping Telegram voice send.")
            return
        if not audio:
            raise RuntimeError("Voice audio is empty.")

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "parse_mode": "HTML",
        }
        if caption:
            payload["caption"] = _sanitize_telegram_text(caption)[:1024]
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
            payload["allow_sending_without_reply"] = True

        voice_file = BytesIO(audio)
        voice_file.name = "reply.mp3"
        self._post_file("sendVoice", payload, {"voice": ("reply.mp3", voice_file, "audio/mpeg")})

    def download_file(self, file_id: str) -> bytes:
        if not self.settings.telegram_bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")

        import requests

        file_data = self._post("getFile", {"file_id": file_id})
        file_path = ((file_data.get("result") or {}).get("file_path")) or ""
        if not file_path:
            raise RuntimeError("Telegram did not return a file_path.")
        url = f"{self.api_base}/file/bot{self.settings.telegram_bot_token}/{file_path}"
        response = requests.get(url, timeout=self.settings.request_timeout_seconds)
        response.raise_for_status()
        return response.content

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

    def _post_file(
        self,
        method: str,
        payload: dict[str, Any],
        files: dict[str, tuple[str, BytesIO, str]],
    ) -> dict[str, Any]:
        import requests

        url = f"{self.api_base}/bot{self.settings.telegram_bot_token}/{method}"
        response = requests.post(
            url,
            data=payload,
            files=files,
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


def _sanitize_telegram_text(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return "Cevap olusturamadim."

    clean = clean.replace("**", "")
    clean = re.sub(r"(?m)^\s*[*-]\s+", "", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    anchors: list[str] = []

    def keep_anchor(match: re.Match[str]) -> str:
        url = match.group(1).strip()
        label = match.group(2).strip()
        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            return html.escape(label or "Haberi oku")
        anchor = f'<a href="{html.escape(url, quote=True)}">{html.escape(label or "Haberi oku")}</a>'
        anchors.append(anchor)
        return f"@@ANCHOR_{len(anchors) - 1}@@"

    clean = re.sub(
        r'<a\s+href="(https?://[^"]+)">([^<>]+)</a>',
        keep_anchor,
        clean,
        flags=re.IGNORECASE,
    )
    clean = html.escape(clean)
    for index, anchor in enumerate(anchors):
        clean = clean.replace(f"@@ANCHOR_{index}@@", anchor)
    return clean.strip()
