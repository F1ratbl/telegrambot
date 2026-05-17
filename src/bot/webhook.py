from __future__ import annotations

import logging
import re
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
_MEDIA_INTERPRETATION_SKIPPED = object()


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

    if _handle_newsletter_request(text, message, chat_id, telegram, agent):
        return True

    media_payload = _media_interpretation_payload(agent, text, message, chat_id, price_chart, visual_generator)
    if media_payload is _MEDIA_INTERPRETATION_UNAVAILABLE:
        if _handle_legacy_media_request(text, message, chat_id, telegram, price_chart, visual_generator, agent):
            return True
    elif media_payload is not _MEDIA_INTERPRETATION_SKIPPED:
        if isinstance(media_payload, dict) and _media_intent(media_payload) == "unavailable":
            if _handle_legacy_media_request(text, message, chat_id, telegram, price_chart, visual_generator, agent):
                return True
        elif isinstance(media_payload, dict) and _handle_interpreted_media_request(
            media_payload,
            text,
            message,
            chat_id,
            telegram,
            price_chart,
            visual_generator,
        ):
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


def _handle_newsletter_request(
    text: str,
    message: dict[str, Any],
    chat_id: int | str,
    telegram: TelegramClient,
    agent: EconomyAgent,
) -> bool:
    memory_chat_id = _memory_chat_id(message, chat_id)
    is_signup_start = _is_newsletter_signup_message(text)
    is_signup_followup = _is_newsletter_followup(agent, memory_chat_id)
    if not is_signup_start and not is_signup_followup and _is_newsletter_info_message(text):
        reply = _newsletter_info_response(text)
        telegram.send_message(
            chat_id=chat_id,
            text=reply,
            reply_to_message_id=message.get("message_id"),
        )
        _remember_agent_exchange(agent, memory_chat_id, text, reply)
        return True

    if not is_signup_start and not is_signup_followup:
        return False

    email = _extract_email(text)
    full_name = _extract_newsletter_name(text, email)
    if not email or not full_name:
        reply = "Harika! Bültenimize kaydolmak için tam adınızı ve e-posta adresinizi alabilir miyim?"
        telegram.send_message(
            chat_id=chat_id,
            text=reply,
            reply_to_message_id=message.get("message_id"),
        )
        _remember_agent_exchange(agent, memory_chat_id, text, reply)
        return True

    subscriber = getattr(agent, "subscribe_newsletter", None)
    if not callable(subscriber):
        return False
    result = subscriber(
        full_name=full_name,
        email=email,
        consent_text=text,
        chat_id=memory_chat_id,
    )
    status = result.get("status") if isinstance(result, dict) else None
    if status == "ok":
        first_name = full_name.split()[0]
        reply = f"Harika {first_name}, bültenimize hoş geldin! Kaydın tamamlandı."
    elif status == "not_configured":
        reply = "Bülten kaydını şu an tamamlayamıyorum; Zapier webhook ayarı production ortamında eksik görünüyor."
    elif status == "missing_fields":
        reply = "Kaydı tamamlamak için tam adınızı ve geçerli e-posta adresinizi birlikte paylaşır mısınız?"
    else:
        reply = "Bülten kaydını Zapier'e gönderirken bir sorun oldu. Biraz sonra tekrar deneyebilir misiniz?"
    telegram.send_message(
        chat_id=chat_id,
        text=reply,
        reply_to_message_id=message.get("message_id"),
    )
    _remember_agent_exchange(agent, memory_chat_id, text, reply)
    return True


def _is_newsletter_signup_message(text: str) -> bool:
    lowered = text.lower()
    if not _mentions_newsletter(lowered):
        return False
    signup_markers = [
        "kayıt olmak istiyorum",
        "kayit olmak istiyorum",
        "kaydolmak istiyorum",
        "kayıt olayım",
        "kayit olayim",
        "abone olmak istiyorum",
        "abone olayım",
        "abone olayim",
        "abone olacağım",
        "abone olacagim",
        "üye olmak istiyorum",
        "uye olmak istiyorum",
        "beni kaydet",
        "beni ekle",
        "ekler misin",
        "listeye ekle",
        "subscribe",
        "sign me up",
    ]
    return any(marker in lowered for marker in signup_markers)


def _is_newsletter_info_message(text: str) -> bool:
    lowered = text.lower()
    if not _mentions_newsletter(lowered):
        return False
    info_markers = [
        "bilgi",
        "detay",
        "anlat",
        "bahset",
        "hakkında",
        "hakkinda",
        "nedir",
        "ne işe yarar",
        "ne ise yarar",
        "ne var",
        "neler var",
        "içerik",
        "icerik",
        "konu",
        "sıklık",
        "siklik",
        "ne zaman",
        "hangi gün",
        "hangi gun",
        "ücret",
        "ucret",
        "fiyat",
        "ücretsiz",
        "ucretsiz",
        "iptal",
        "çık",
        "cik",
        "gizlilik",
        "veri",
        "email",
        "e-posta",
        "eposta",
        "nasıl",
        "nasil",
    ]
    return any(marker in lowered for marker in info_markers)


def _mentions_newsletter(lowered: str) -> bool:
    newsletter_markers = ["bülten", "bulten", "newsletter", "mail listesi", "eposta listesi", "e-posta listesi"]
    return any(marker in lowered for marker in newsletter_markers)


def _newsletter_info_response(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in ["ücret", "ucret", "fiyat", "ücretsiz", "ucretsiz"]):
        return "Bülten ücretsizdir. Ekonomi gündemi, piyasa özeti ve önemli veri takvimi gibi başlıkları sade bir dille paylaşmak için hazırlanır."
    if any(marker in lowered for marker in ["iptal", "çık", "cik", "abonelikten"]):
        return "Bültenden çıkmak istediğinizde bunu yazmanız yeterli; abonelik kaydınız kaldırılabilir. Şimdilik kayıt için yalnızca ad soyad ve e-posta alıyoruz."
    if any(marker in lowered for marker in ["gizlilik", "veri", "email", "e-posta", "eposta"]):
        return "Bülten kaydı için ad soyad ve e-posta adresinizi alıyoruz. Bu bilgiler bülten gönderimi ve kayıt takibi için kullanılır; yatırım tavsiyesi ya da kişisel portföy takibi amacıyla kullanılmaz."
    if any(marker in lowered for marker in ["sıklık", "siklik", "ne zaman", "hangi gün", "hangi gun"]):
        return "Bülten düzenli ekonomi özeti mantığında tasarlandı: önemli piyasa gelişmeleri, makro veri gündemi ve öne çıkan risk başlıkları kısa ve okunabilir şekilde paylaşılır."
    return (
        "Bültenimiz ekonomi ve finans gündemini kısa, anlaşılır ve uygulanabilir şekilde takip etmek isteyenler için hazırlanır. "
        "İçerikte piyasa özeti, önemli makro veriler, merkez bankası gündemi, altın-döviz-kripto gibi başlıklarda genel görünüm, "
        "öne çıkan haberlerin olası etkileri ve haftalık takip listesi yer alabilir. Kesin yatırım tavsiyesi vermez; amaç gündemi daha hızlı anlamanıza yardımcı olmaktır. "
        "Kayıt olmak isterseniz tam adınızı ve e-posta adresinizi paylaşmanız yeterli."
    )


def _is_newsletter_followup(agent: EconomyAgent, chat_id: str | None) -> bool:
    memory = getattr(agent, "memory", None)
    snapshot = getattr(memory, "snapshot", None)
    if not callable(snapshot):
        return False
    for previous in reversed(snapshot(chat_id)[-4:]):
        if getattr(previous, "role", None) == "model" and _asks_for_newsletter_contact(getattr(previous, "text", "")):
            return True
    return False


def _asks_for_newsletter_contact(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ["bülten", "bulten"]) and any(
        marker in lowered for marker in ["e-posta", "eposta", "email", "adınızı", "adinizi"]
    )


def _extract_email(text: str) -> str | None:
    match = re.search(r"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    return match.group(0).lower() if match else None


def _extract_newsletter_name(text: str, email: str | None) -> str | None:
    candidate = text
    if email:
        candidate = candidate.replace(email, " ")
    candidate = re.sub(r"\b(e-?posta(?:m| adresim)?|email(?:im)?|mail(?:im)?|adresim|ad[ıi]m|ismim)\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b(b[üu]lten(?:inize|e)?|kay[ıi]t|kaydolmak|istiyorum|abone|olmak|olay[ıi]m|beni|l[üu]tfen)\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü\s'-]", " ", candidate)
    words = [word.strip(" '-") for word in candidate.split() if word.strip(" '-")]
    if len(words) < 2:
        return None
    return " ".join(words[:4])


def _remember_agent_exchange(agent: EconomyAgent, chat_id: str | None, user_text: str, reply_text: str) -> None:
    memory = getattr(agent, "memory", None)
    remember = getattr(memory, "remember_exchange", None)
    if callable(remember):
        remember(chat_id, user_text, reply_text)


def _handle_legacy_media_request(
    text: str,
    message: dict[str, Any],
    chat_id: int | str,
    telegram: TelegramClient,
    price_chart: PriceChartTool | None,
    visual_generator: EconomyVisualGenerator | None,
    agent: EconomyAgent,
) -> bool:
    if price_chart and _handle_price_chart_request(text, message, chat_id, telegram, price_chart, agent):
        return True
    if visual_generator and _handle_visual_request(text, message, chat_id, telegram, visual_generator):
        return True
    return False


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


def _media_interpretation_payload(
    agent: EconomyAgent,
    text: str,
    message: dict[str, Any],
    chat_id: int | str,
    price_chart: PriceChartTool | None,
    visual_generator: EconomyVisualGenerator | None,
) -> dict[str, Any] | object | None:
    if not price_chart and not visual_generator:
        return _MEDIA_INTERPRETATION_SKIPPED
    if not _media_interpreter_available(agent):
        return _MEDIA_INTERPRETATION_UNAVAILABLE

    visual_chat_id = _memory_chat_id(message, chat_id)
    if not _should_interpret_media_request(text, message, visual_generator, visual_chat_id):
        return _MEDIA_INTERPRETATION_SKIPPED

    return _interpret_media_request(agent, text, message, chat_id, visual_generator)


def _media_interpreter_available(agent: EconomyAgent) -> bool:
    interpreter = getattr(agent, "interpret_media_request", None)
    if not callable(interpreter):
        return False
    settings = getattr(agent, "settings", None)
    return settings is None or bool(getattr(settings, "google_api_key", None))


def _should_interpret_media_request(
    text: str,
    message: dict[str, Any],
    visual_generator: EconomyVisualGenerator | None,
    visual_chat_id: str | None,
) -> bool:
    if _largest_photo_file_id(message):
        return True
    if _has_media_language(text):
        return True
    if visual_generator and _has_visual_context(visual_generator, visual_chat_id):
        return _is_visual_followup(visual_generator, text)
    return False


def _has_media_language(text: str) -> bool:
    lowered = text.lower()
    media_markers = [
        "grafik",
        "grafiği",
        "grafigi",
        "chart",
        "mum",
        "candle",
        "candlestick",
        "görsel",
        "gorsel",
        "resim",
        "foto",
        "fotoğraf",
        "fotograf",
        "image",
        "picture",
        "infografik",
        "şema",
        "sema",
        "poster",
        "kapak",
        "dergi",
        "illustration",
        "ilüstrasyon",
        "ilustrasyon",
        "çiz",
        "ciz",
        "draw",
    ]
    return any(marker in lowered for marker in media_markers)


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
