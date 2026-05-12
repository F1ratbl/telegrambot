from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from src.ai.agent import EconomyAgent
from src.audio.speech import SpeechServiceError, SpeechToTextClient, TextToSpeechClient
from src.bot.telegram import TelegramClient
from src.config import Settings


logger = logging.getLogger(__name__)


def create_telegram_blueprint(
    settings: Settings,
    agent: EconomyAgent,
    telegram: TelegramClient,
    speech_to_text: SpeechToTextClient | None = None,
    text_to_speech: TextToSpeechClient | None = None,
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
            handled = _handle_update(update, agent, telegram, speech_to_text, text_to_speech)
        except Exception:
            logger.exception("Failed to process Telegram update.")
            return jsonify({"ok": True, "handled": False}), 200
        return jsonify({"ok": True, "handled": handled}), 200

    return blueprint


def _handle_update(
    update: dict[str, Any],
    agent: EconomyAgent,
    telegram: TelegramClient,
    speech_to_text: SpeechToTextClient | None = None,
    text_to_speech: TextToSpeechClient | None = None,
) -> bool:
    message = update.get("message") or update.get("edited_message")
    if not isinstance(message, dict):
        return False

    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return False

    voice = message.get("voice") or message.get("audio")
    if isinstance(voice, dict):
        return _handle_voice_message(
            message=message,
            chat_id=chat_id,
            agent=agent,
            telegram=telegram,
            speech_to_text=speech_to_text,
            text_to_speech=text_to_speech,
        )

    text = message.get("text") or message.get("caption")
    if not isinstance(text, str) or not text.strip():
        return False

    reply = agent.reply(
        user_message=text,
        chat_id=_memory_chat_id(message, chat_id),
    )
    telegram.send_message(
        chat_id=chat_id,
        text=reply,
        reply_to_message_id=message.get("message_id"),
    )
    return True


def _handle_voice_message(
    message: dict[str, Any],
    chat_id: int | str,
    agent: EconomyAgent,
    telegram: TelegramClient,
    speech_to_text: SpeechToTextClient | None,
    text_to_speech: TextToSpeechClient | None,
) -> bool:
    if speech_to_text is None or text_to_speech is None or not speech_to_text.enabled or not text_to_speech.enabled:
        telegram.send_message(
            chat_id=chat_id,
            text=(
                "Sesli cevap icin Deepgram ve ElevenLabs ayarlari eksik. "
                "DEEPGRAM_API_KEY, ELEVENLABS_API_KEY ve ELEVENLABS_VOICE_ID eklenmeli."
            ),
            reply_to_message_id=message.get("message_id"),
        )
        return True

    voice = message.get("voice") or message.get("audio") or {}
    file_id = voice.get("file_id")
    if not isinstance(file_id, str) or not file_id:
        return False

    try:
        audio = telegram.download_file(file_id)
    except Exception:
        logger.exception("Failed to download Telegram voice file.")
        telegram.send_message(
            chat_id=chat_id,
            text="Ses dosyasini Telegram'dan indiremedim. Biraz sonra tekrar deneyebilir misiniz?",
            reply_to_message_id=message.get("message_id"),
        )
        return True

    try:
        transcript = speech_to_text.transcribe(audio, mimetype=voice.get("mime_type") or "audio/ogg")
    except Exception:
        logger.exception("Deepgram transcription failed.")
        telegram.send_message(
            chat_id=chat_id,
            text="Deepgram sesi yaziya ceviremedi. API key, kota veya ses formatini kontrol etmek gerekiyor.",
            reply_to_message_id=message.get("message_id"),
        )
        return True

    if not transcript:
        telegram.send_message(
            chat_id=chat_id,
            text="Sesi anlayamadim. Biraz daha net tekrar gonderebilir misiniz?",
            reply_to_message_id=message.get("message_id"),
        )
        return True

    try:
        reply = agent.reply(user_message=transcript, chat_id=_memory_chat_id(message, chat_id))
    except Exception:
        logger.exception("Agent failed after voice transcription.")
        telegram.send_message(
            chat_id=chat_id,
            text="Sesi yazıya cevirdim ama cevabi olustururken sorun yasadim. Biraz sonra tekrar deneyebilir misiniz?",
            reply_to_message_id=message.get("message_id"),
        )
        return True

    try:
        spoken_reply = text_to_speech.synthesize(reply)
    except SpeechServiceError as exc:
        logger.exception("ElevenLabs synthesis failed: %s", exc)
        error_message = exc.message.lower()
        if exc.provider == "ElevenLabs" and exc.status_code == 402 and "library voices" in error_message:
            error_text = (
                "Sesli yanit su an secili ElevenLabs sesi bu planda API'den kullanilamadigi icin uretilemedi. "
                "Cevabi yazili birakiyorum:\n\n"
            )
        elif exc.provider == "ElevenLabs" and exc.status_code == 402:
            error_text = (
                "Sesli yanit su an ElevenLabs plan/kota kisiti nedeniyle uretilemedi. "
                "Cevabi yazili birakiyorum:\n\n"
            )
        else:
            error_text = "Sesli yanit su an uretilemedi. Cevabi yazili birakiyorum:\n\n"
        telegram.send_message(
            chat_id=chat_id,
            text=f"{error_text}{reply}",
            reply_to_message_id=message.get("message_id"),
        )
        return True
    except Exception:
        logger.exception("ElevenLabs synthesis failed.")
        telegram.send_message(
            chat_id=chat_id,
            text=f"Sesli cevap uretemedim. Cevabi yazili birakiyorum:\n\n{reply}",
            reply_to_message_id=message.get("message_id"),
        )
        return True

    try:
        telegram.send_voice(
            chat_id=chat_id,
            audio=spoken_reply,
            reply_to_message_id=message.get("message_id"),
        )
    except Exception:
        logger.exception("Failed to send Telegram voice reply.")
        telegram.send_message(
            chat_id=chat_id,
            text="Cevabi olusturdum ama Telegram'a ses olarak gonderemedim. Ses formatini kontrol etmek gerekiyor.",
            reply_to_message_id=message.get("message_id"),
        )
        return True

    return True


def _memory_chat_id(message: dict[str, Any], chat_id: int | str) -> str:
    sender = message.get("from") or {}
    user_id = sender.get("id") if isinstance(sender, dict) else None
    if user_id is None:
        return str(chat_id)
    return f"{chat_id}:{user_id}"
