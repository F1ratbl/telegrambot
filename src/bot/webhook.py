from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from src.ai.agent import EconomyAgent
from src.bot.telegram import TelegramClient
from src.config import Settings


logger = logging.getLogger(__name__)


def create_telegram_blueprint(
    settings: Settings,
    agent: EconomyAgent,
    telegram: TelegramClient,
) -> Blueprint:
    blueprint = Blueprint("telegram", __name__)

    @blueprint.get("/telegram/webhook")
    @blueprint.get("/webhook")
    def webhook_info() -> tuple[dict[str, object], int]:
        return (
            {
                "ok": True,
                "route": "/telegram/webhook",
                "method": "POST",
                "secret_required": bool(settings.telegram_webhook_secret),
            },
            200,
        )

    @blueprint.post("/telegram/webhook")
    @blueprint.post("/webhook")
    def telegram_webhook() -> tuple[dict[str, object], int]:
        if settings.telegram_webhook_secret:
            secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
            if secret != settings.telegram_webhook_secret:
                return jsonify({"ok": False, "error": "invalid webhook secret"}), 403

        update = request.get_json(silent=True) or {}
        try:
            handled = _handle_update(update, agent, telegram)
        except Exception:
            logger.exception("Failed to process Telegram update.")
            return jsonify({"ok": True, "handled": False}), 200
        return jsonify({"ok": True, "handled": handled}), 200

    return blueprint


def _handle_update(
    update: dict[str, Any],
    agent: EconomyAgent,
    telegram: TelegramClient,
) -> bool:
    message = update.get("message") or update.get("edited_message")
    if not isinstance(message, dict):
        return False

    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return False

    text = message.get("text") or message.get("caption")
    if not isinstance(text, str) or not text.strip():
        return False

    reply = agent.reply(
        user_message=text,
        chat_id=str(chat_id),
    )
    telegram.send_message(
        chat_id=chat_id,
        text=reply,
        reply_to_message_id=message.get("message_id"),
    )
    return True
