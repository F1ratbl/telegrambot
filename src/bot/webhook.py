from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from src.ai.agent import EconomyAgent
from src.audio.speech import SpeechServiceError, SpeechToTextClient, TextToSpeechClient
from src.bot.telegram import TelegramClient
from src.config import Settings
from src.tools.charting import ChartRequest, PriceChartTool
from src.tools.visual_generation import EconomyVisualGenerator


logger = logging.getLogger(__name__)
_MEDIA_INTERPRETATION_UNAVAILABLE = object()


def create_telegram_blueprint(
    settings: Settings,
    agent: EconomyAgent,
    telegram: TelegramClient,
    speech_to_text: SpeechToTextClient | None = None,
    text_to_speech: TextToSpeechClient | None = None,
    price_chart: PriceChartTool | None = None,
    visual_generator: EconomyVisualGenerator | None = None,
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
            handled = _handle_update(
                update,
                agent,
                telegram,
                speech_to_text,
                text_to_speech,
                price_chart or PriceChartTool(settings),
                visual_generator or EconomyVisualGenerator(settings),
            )
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
    price_chart: PriceChartTool | None = None,
    visual_generator: EconomyVisualGenerator | None = None,
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

    media_payload = (
        _interpret_media_request(agent, text, message, chat_id, visual_generator)
        if price_chart or visual_generator
        else _MEDIA_INTERPRETATION_UNAVAILABLE
    )
    if media_payload is not _MEDIA_INTERPRETATION_UNAVAILABLE:
        if isinstance(media_payload, dict) and _handle_interpreted_media_request(
            media_payload,
            text,
            message,
            chat_id,
            telegram,
            price_chart,
            visual_generator,
        ):
            return True
    else:
        if price_chart and _handle_price_chart_request(text, message, chat_id, telegram, price_chart, agent):
            return True

        if visual_generator and _handle_visual_request(text, message, chat_id, telegram, visual_generator):
            return True

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


def _handle_price_chart_request(
    text: str,
    message: dict[str, Any],
    chat_id: int | str,
    telegram: TelegramClient,
    price_chart: PriceChartTool,
    agent: EconomyAgent,
) -> bool:
    chart_request = _interpret_chart_request(agent, price_chart, text, message, chat_id)
    if chart_request is None:
        chart_request = price_chart.parse_request(text)
    if chart_request is None:
        return False
    return _send_price_chart(chart_request, message, chat_id, telegram, price_chart)


def _send_price_chart(
    chart_request: ChartRequest,
    message: dict[str, Any],
    chat_id: int | str,
    telegram: TelegramClient,
    price_chart: PriceChartTool,
) -> bool:
    try:
        image, caption = price_chart.create_price_chart(chart_request)
        telegram.send_photo(
            chat_id=chat_id,
            image=image,
            caption=caption,
            reply_to_message_id=message.get("message_id"),
        )
    except Exception:
        logger.exception("Failed to create or send price chart.")
        telegram.send_message(
            chat_id=chat_id,
            text="Grafiği şu an oluşturamadım. Veri sağlayıcıya erişim veya grafik üretimi tarafında sorun olabilir.",
            reply_to_message_id=message.get("message_id"),
        )
    return True


def _interpret_media_request(
    agent: EconomyAgent,
    text: str,
    message: dict[str, Any],
    chat_id: int | str,
    visual_generator: EconomyVisualGenerator | None,
) -> dict[str, Any] | object | None:
    interpreter = getattr(agent, "interpret_media_request", None)
    if not callable(interpreter):
        return _MEDIA_INTERPRETATION_UNAVAILABLE

    settings = getattr(agent, "settings", None)
    if settings is not None and not getattr(settings, "google_api_key", None):
        return _MEDIA_INTERPRETATION_UNAVAILABLE

    visual_chat_id = _memory_chat_id(message, chat_id)
    try:
        return interpreter(
            user_message=text,
            chat_id=visual_chat_id,
            has_reference_image=bool(_largest_photo_file_id(message)),
            has_visual_context=_has_visual_context(visual_generator, visual_chat_id) if visual_generator else False,
        )
    except TypeError:
        try:
            return interpreter(user_message=text, chat_id=visual_chat_id)
        except Exception:
            logger.exception("Failed to interpret media request.")
            return None
    except Exception:
        logger.exception("Failed to interpret media request.")
        return None


def _handle_interpreted_media_request(
    payload: dict[str, Any],
    text: str,
    message: dict[str, Any],
    chat_id: int | str,
    telegram: TelegramClient,
    price_chart: PriceChartTool | None,
    visual_generator: EconomyVisualGenerator | None,
) -> bool:
    intent = _media_intent(payload)
    if intent == "price_chart":
        if price_chart is None:
            return False
        chart_payload = dict(payload)
        chart_payload["is_chart_request"] = True
        chart_request = price_chart.request_from_interpretation(chart_payload)
        if chart_request is None:
            return False
        return _send_price_chart(chart_request, message, chat_id, telegram, price_chart)

    if intent in {"visual", "visual_edit"}:
        if visual_generator is None:
            return False
        visual_request = _visual_request_from_interpretation(payload, text)
        return _handle_visual_request(
            text,
            message,
            chat_id,
            telegram,
            visual_generator,
            visual_request=visual_request,
            force_reference_image=intent == "visual_edit" or bool(payload.get("use_reference_image")),
        )

    return False


def _media_intent(payload: dict[str, Any]) -> str:
    intent = payload.get("intent") or payload.get("type") or payload.get("action")
    if not isinstance(intent, str):
        return "none"
    normalized = intent.strip().lower().replace("-", "_")
    aliases = {
        "chart": "price_chart",
        "graph": "price_chart",
        "image": "visual",
        "image_generation": "visual",
        "generate_image": "visual",
        "create_image": "visual",
        "image_edit": "visual_edit",
        "edit_image": "visual_edit",
    }
    return aliases.get(normalized, normalized)


def _visual_request_from_interpretation(payload: dict[str, Any], fallback_text: str) -> str:
    request_text = payload.get("request_text") or payload.get("prompt") or payload.get("instruction")
    if isinstance(request_text, str) and request_text.strip():
        return request_text.strip()
    return fallback_text.strip()


def _interpret_chart_request(
    agent: EconomyAgent,
    price_chart: PriceChartTool,
    text: str,
    message: dict[str, Any],
    chat_id: int | str,
) -> ChartRequest | None:
    interpreter = getattr(agent, "interpret_chart_request", None)
    if not callable(interpreter):
        return None
    try:
        payload = interpreter(
            user_message=text,
            chat_id=_memory_chat_id(message, chat_id),
        )
        return price_chart.request_from_interpretation(payload)
    except Exception:
        logger.exception("Failed to interpret price chart request.")
        return None


def _handle_visual_request(
    text: str,
    message: dict[str, Any],
    chat_id: int | str,
    telegram: TelegramClient,
    visual_generator: EconomyVisualGenerator,
    visual_request: str | None = None,
    force_reference_image: bool = False,
) -> bool:
    photo_file_id = _largest_photo_file_id(message)
    visual_chat_id = _memory_chat_id(message, chat_id)
    has_visual_context = _has_visual_context(visual_generator, visual_chat_id)
    if visual_request is None:
        if photo_file_id:
            visual_request = _parse_visual_request(visual_generator, text, has_reference_image=True)
        else:
            visual_request = _parse_visual_request(visual_generator, text, has_visual_context=has_visual_context)
    if visual_request is None:
        return False
    try:
        reference_image = telegram.download_file(photo_file_id) if photo_file_id else None
        if reference_image is None and (force_reference_image or _is_visual_followup(visual_generator, text)):
            context_image = _context_reference_image(visual_generator, visual_chat_id)
            if context_image is not None:
                reference_image = context_image
                visual_request = _contextual_visual_request(visual_generator, visual_chat_id, visual_request)
            elif force_reference_image:
                telegram.send_message(
                    chat_id=chat_id,
                    text="Düzenlenecek görseli bulamadım. Görseli yanıtlayarak ya da yeniden yükleyerek gönderebilir misin?",
                    reply_to_message_id=message.get("message_id"),
                )
                return True
        if reference_image is None:
            image, caption = visual_generator.generate(visual_request)
        else:
            image, caption = visual_generator.generate(visual_request, reference_image=reference_image)
        _remember_visual_context(visual_generator, visual_chat_id, visual_request, image)
        telegram.send_photo(
            chat_id=chat_id,
            image=image,
            caption=caption,
            reply_to_message_id=message.get("message_id"),
        )
    except Exception:
        logger.exception("Failed to generate or send economy visual.")
        telegram.send_message(
            chat_id=chat_id,
            text="Görseli şu an oluşturamadım. Gemini görsel modeli, API anahtarı veya kota tarafını kontrol etmek gerekiyor.",
            reply_to_message_id=message.get("message_id"),
        )
    return True


def _parse_visual_request(
    visual_generator: EconomyVisualGenerator,
    text: str,
    has_reference_image: bool = False,
    has_visual_context: bool = False,
) -> str | None:
    try:
        return visual_generator.parse_request(
            text,
            has_reference_image=has_reference_image,
            has_visual_context=has_visual_context,
        )
    except TypeError:
        if has_reference_image:
            try:
                return visual_generator.parse_request(text, has_reference_image=True)
            except TypeError:
                return visual_generator.parse_request(text)
        return visual_generator.parse_request(text)


def _has_visual_context(visual_generator: EconomyVisualGenerator, chat_id: str | None) -> bool:
    checker = getattr(visual_generator, "has_visual_context", None)
    return bool(checker(chat_id)) if callable(checker) else False


def _is_visual_followup(visual_generator: EconomyVisualGenerator, text: str) -> bool:
    checker = getattr(visual_generator, "is_visual_followup", None)
    return bool(checker(text)) if callable(checker) else False


def _context_reference_image(visual_generator: EconomyVisualGenerator, chat_id: str | None) -> bytes | None:
    getter = getattr(visual_generator, "context_reference_image", None)
    return getter(chat_id) if callable(getter) else None


def _contextual_visual_request(
    visual_generator: EconomyVisualGenerator,
    chat_id: str | None,
    request_text: str,
) -> str:
    builder = getattr(visual_generator, "contextual_request", None)
    return builder(chat_id, request_text) if callable(builder) else request_text


def _remember_visual_context(
    visual_generator: EconomyVisualGenerator,
    chat_id: str | None,
    request_text: str,
    image: bytes,
) -> None:
    remember = getattr(visual_generator, "remember_visual_context", None)
    if callable(remember):
        remember(chat_id, request_text, image)


def _largest_photo_file_id(message: dict[str, Any]) -> str | None:
    direct_photo = _photo_file_id_from_message(message)
    if direct_photo:
        return direct_photo
    reply_to_message = message.get("reply_to_message")
    if isinstance(reply_to_message, dict):
        return _photo_file_id_from_message(reply_to_message)
    return None


def _photo_file_id_from_message(message: dict[str, Any]) -> str | None:
    photos = message.get("photo") or []
    if not isinstance(photos, list) or not photos:
        return None
    candidates = [photo for photo in photos if isinstance(photo, dict) and isinstance(photo.get("file_id"), str)]
    if not candidates:
        return None
    best = max(candidates, key=lambda photo: int(photo.get("file_size") or 0))
    return best["file_id"]


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
