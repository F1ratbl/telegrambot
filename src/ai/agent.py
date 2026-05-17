from __future__ import annotations

import logging
import json
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.ai.prompts import SYSTEM_PROMPT
from src.ai.tool_declarations import FUNCTION_DECLARATIONS
from src.bot.memory import InMemoryConversationMemory
from src.config import Settings
from src.tools.knowledge_base import KnowledgeBaseTool
from src.tools.market_data import MarketDataClient
from src.tools.news import NewsSearchClient


logger = logging.getLogger(__name__)


START_MESSAGE = (
    "Merhaba! Ben ekonomi ve finans alaninda size yardimci olmak icin buradayim. "
    "Piyasalar, yatirimlar veya ekonomiyle ilgili bir sorunuz varsa yazabilirsiniz."
)
NO_TEXT_FALLBACK = "Cevap olusturamadim."


class GeminiTemporarilyUnavailableError(RuntimeError):
    pass


class EconomyAgent:
    def __init__(
        self,
        settings: Settings,
        market_data: MarketDataClient,
        knowledge_base: KnowledgeBaseTool,
        memory: InMemoryConversationMemory,
        news_search: NewsSearchClient | None = None,
    ) -> None:
        self.settings = settings
        self.market_data = market_data
        self.knowledge_base = knowledge_base
        self.memory = memory
        self.news_search = news_search or NewsSearchClient(settings)
        self._client = None
        self._media_interpretation_disabled_until = 0.0
        self._chart_interpretation_disabled_until = 0.0

    def reply(self, user_message: str, chat_id: str | None = None, user_name: str | None = None) -> str:
        if not user_message.strip():
            return "Ekonomiyle ilgili sorunuzu yazarsaniz yardimci olayim."
        if user_message.strip().lower() == "/start":
            return START_MESSAGE
        clean_message = user_message.strip()
        if self._is_news_question(clean_message):
            answer = self._answer_news_question(clean_message, chat_id)
            self.memory.remember_exchange(chat_id, clean_message, answer)
            return answer
        if not self.settings.google_api_key:
            return (
                "Gemini API anahtari henuz tanimli degil. GOOGLE_API_KEY veya "
                "GEMINI_API_KEY ortam degiskenini ekledikten sonra cevap uretebilirim."
            )

        try:
            remembered_name = self._remember_user_name(chat_id, clean_message)
            active_name = remembered_name or self.memory.get_preferred_name(chat_id)
            return self._reply_with_gemini(clean_message, chat_id, active_name)
        except GeminiTemporarilyUnavailableError:
            return (
                "Gemini su an yogun gorunuyor. Bir kac saniye sonra ayni soruyu tekrar "
                "gonderirseniz devam edebilirim."
            )
        except Exception:
            logger.exception("Gemini agent failed.")
            return (
                "Su an ekonomi asistaninin model veya veri baglantisinda bir sorun var. "
                "Biraz sonra tekrar deneyebilir misiniz?"
            )

    def _reply_with_gemini(self, user_message: str, chat_id: str | None, user_name: str | None) -> str:
        prefetched_market = self._prefetch_market_snapshot(user_message, chat_id)
        if prefetched_market is not None:
            answer = self._reply_with_prefetched_market(user_message, chat_id, user_name, prefetched_market)
            self.memory.remember_exchange(chat_id, user_message, answer)
            return answer

        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.settings.google_api_key)

        contents = self._build_contents(types, user_message, chat_id, user_name)
        config = self._build_config(types, include_tools=True)

        for _ in range(self.settings.gemini_max_tool_rounds):
            response = self._generate_content_with_retry(
                model=self.settings.gemini_model,
                contents=contents,
                config=config,
            )
            function_calls = self._extract_function_calls(response)
            if not function_calls:
                answer = self._extract_text(response)
                self.memory.remember_exchange(chat_id, user_message, answer)
                return answer

            contents.append(response.candidates[0].content)
            response_parts = []
            for function_call in function_calls:
                tool_result = self._execute_tool(function_call, chat_id=chat_id)
                response_parts.append(self._function_response_part(types, function_call, tool_result))
            contents.append(types.Content(role="user", parts=response_parts))

        final_response = self._generate_content_with_retry(
            model=self.settings.gemini_model,
            contents=contents,
            config=self._build_config(types, include_tools=False),
        )
        answer = self._extract_text(final_response)
        self.memory.remember_exchange(chat_id, user_message, answer)
        return answer

    def _reply_with_prefetched_market(
        self,
        user_message: str,
        chat_id: str | None,
        user_name: str | None,
        market_snapshot: dict[str, Any],
    ) -> str:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.settings.google_api_key)

        contents = self._build_contents(
            types,
            user_message,
            chat_id,
            user_name,
            market_snapshot=market_snapshot,
        )
        response = self._generate_content_with_retry(
            model=self.settings.gemini_model,
            contents=contents,
            config=self._build_config(types, include_tools=False),
        )
        return self._extract_text(response)

    def interpret_chart_request(self, user_message: str, chat_id: str | None = None) -> dict[str, Any] | None:
        if not user_message.strip() or not self.settings.google_api_key:
            return None
        if time.monotonic() < self._chart_interpretation_disabled_until:
            return None

        try:
            from google import genai
            from google.genai import types

            if self._client is None:
                self._client = genai.Client(api_key=self.settings.google_api_key)

            response = self._generate_content_with_retry(
                model=self.settings.gemini_model,
                contents=[self._chart_interpretation_prompt(user_message, chat_id)],
                config=self._build_chart_interpretation_config(types),
            )
            return self._parse_chart_interpretation(self._extract_text_or_none(response))
        except GeminiTemporarilyUnavailableError as exc:
            self._chart_interpretation_disabled_until = time.monotonic() + 60
            logger.warning("Gemini chart request interpretation temporarily unavailable: %s", exc)
            return None
        except Exception:
            logger.exception("Gemini chart request interpretation failed.")
            return None

    def interpret_media_request(
        self,
        user_message: str,
        chat_id: str | None = None,
        has_reference_image: bool = False,
        has_visual_context: bool = False,
    ) -> dict[str, Any] | None:
        if not user_message.strip() or not self.settings.google_api_key:
            return None
        if time.monotonic() < self._media_interpretation_disabled_until:
            return {"intent": "unavailable"}

        try:
            from google import genai
            from google.genai import types

            if self._client is None:
                self._client = genai.Client(api_key=self.settings.google_api_key)

            response = self._generate_content_with_retry(
                model=self.settings.gemini_model,
                contents=[
                    self._media_interpretation_prompt(
                        user_message=user_message,
                        chat_id=chat_id,
                        has_reference_image=has_reference_image,
                        has_visual_context=has_visual_context,
                    )
                ],
                config=self._build_media_interpretation_config(types),
            )
            return self._parse_media_interpretation(self._extract_text_or_none(response))
        except GeminiTemporarilyUnavailableError as exc:
            self._media_interpretation_disabled_until = time.monotonic() + 60
            logger.warning("Gemini media request interpretation temporarily unavailable: %s", exc)
            return {"intent": "unavailable"}
        except Exception:
            logger.exception("Gemini media request interpretation failed.")
            return None

    def _media_interpretation_prompt(
        self,
        user_message: str,
        chat_id: str | None,
        has_reference_image: bool,
        has_visual_context: bool,
    ) -> str:
        active_asset = self._infer_active_asset(chat_id, user_message)
        lines = [
            "Kullanici mesajini analiz et ve yalnizca JSON dondur.",
            'intent yalnizca su degerlerden biri olsun: "price_chart", "visual", "visual_edit", "none".',
            "Hazir komut kalibi veya kelime listesi varsayma; kullanicinin dogal niyetini yorumla.",
            "price_chart: Kullanici finansal varlik icin fiyat grafigi, mum grafigi, tarihsel performans grafigi veya chart istiyorsa.",
            "visual: Kullanici yeni bir gorsel, infografik, sema, ilustrasyon, kapak veya dergi/editorial gorseli istiyorsa.",
            "visual_edit: Kullanici yuklenen ya da once uretilen gorselde renk, kiyafet, nesne, arka plan, stil veya kompozisyon degisikligi istiyorsa.",
            "none: Normal soru, haber yorumu, fiyat sorusu, sohbet veya gorsel/grafik uretimi istemeyen mesaj.",
            "Kavramsal infografik/sema istegini price_chart yapma; gercek fiyat verisi grafigi istenmedikce visual sec.",
            "Gorsel baglami varsa kisa talimatlari da yorumla: 'kravati pembe yap', 'takimi kirmizi olsun', 'arka plani degistir' -> visual_edit.",
            "price_chart icin symbol standart kisa kod olsun: AMD, NVDA, AAPL, TSLA, MSFT, SP500, NASDAQ, DAX, BIST100, GOLD, SILVER, BRENT, BTCUSD, ETHUSD, USDTRY, EURTRY.",
            "range_days kullanici acikca gun araligi verirse sayi olsun; yoksa null.",
            "interval_hours kullanici mum/veri araligi verirse saat sayisi olsun; yoksa null.",
            "period range_days yoksa 7d, 1mo, 3mo, 6mo veya 1y degerlerinden biri olsun. Emin degilsen 1mo.",
            "visual ve visual_edit icin request_text kullanicinin istegini uygulanabilir tek talimat olarak yaz; bos birakma.",
            "visual_edit icin use_reference_image true olsun.",
            (
                'JSON bicimi: {"intent":"price_chart","symbol":"AMD","range_days":4,'
                '"interval_hours":4,"period":"7d","request_text":null,"use_reference_image":false}'
            ),
        ]
        if active_asset:
            lines.append(f"Aktif varlik baglami: {active_asset}")
        lines.append(f"Yuklenen/cevaplanan gorsel var: {str(has_reference_image).lower()}")
        lines.append(f"Once uretilmis gorsel baglami var: {str(has_visual_context).lower()}")
        lines.append(f"Kullanici mesaji: {user_message}")
        return "\n".join(lines)

    def _build_media_interpretation_config(self, types: Any) -> Any:
        kwargs: dict[str, Any] = {
            "system_instruction": "Finans botunda gorsel, gorsel duzenleme ve fiyat grafigi niyetlerini JSON'a ceviren bir yorumlayicisin.",
            "max_output_tokens": 450,
            "response_mime_type": "application/json",
        }
        if self.settings.gemini_temperature is not None:
            kwargs["temperature"] = 0
        try:
            return types.GenerateContentConfig(**kwargs)
        except TypeError:
            kwargs.pop("response_mime_type", None)
            return types.GenerateContentConfig(**kwargs)

    def _parse_media_interpretation(self, text: str | None) -> dict[str, Any] | None:
        return self._parse_json_interpretation(text, "media")

    def _chart_interpretation_prompt(self, user_message: str, chat_id: str | None) -> str:
        active_asset = self._infer_active_asset(chat_id, user_message)
        lines = [
            "Kullanici mesajini analiz et ve yalnizca JSON dondur.",
            "Gorev: Kullanici bir finansal varlik icin fiyat grafigi/grafik cizimi istiyor mu belirle.",
            "Dogal Turkce ve Ingilizce ifadeleri yorumla; hazir komut kalibi varsayma.",
            "Grafik istegi degilse veya varlik/sembol anlasilmiyorsa is_chart_request false olsun.",
            "Sembol alaninda standart kisa kod kullan: AMD, NVDA, AAPL, TSLA, MSFT, SP500, NASDAQ, DAX, BIST100, GOLD, SILVER, BRENT, BTCUSD, ETHUSD, USDTRY, EURTRY.",
            "range_days kullanici acikca gun araligi verirse sayi olsun: 'son 4 gun', '4 gundeki' -> 4. Yoksa null.",
            "interval_hours kullanici mum/veri araligi verirse saat sayisi olsun: '4 saatlik' -> 4. Yoksa null.",
            "period range_days yoksa 7d, 1mo, 3mo, 6mo veya 1y degerlerinden biri olsun. Emin degilsen 1mo.",
            'JSON bicimi: {"is_chart_request": true, "symbol": "AMD", "range_days": 4, "interval_hours": 4, "period": "7d"}',
        ]
        if active_asset:
            lines.append(f"Aktif varlik baglami: {active_asset}")
        lines.append(f"Kullanici mesaji: {user_message}")
        return "\n".join(lines)

    def _build_chart_interpretation_config(self, types: Any) -> Any:
        kwargs: dict[str, Any] = {
            "system_instruction": "Finans sohbetlerinde grafik isteklerini yapilandirilmis JSON'a ceviren bir yorumlayicisin.",
            "max_output_tokens": 300,
            "response_mime_type": "application/json",
        }
        if self.settings.gemini_temperature is not None:
            kwargs["temperature"] = 0
        try:
            return types.GenerateContentConfig(**kwargs)
        except TypeError:
            kwargs.pop("response_mime_type", None)
            return types.GenerateContentConfig(**kwargs)

    def _parse_chart_interpretation(self, text: str | None) -> dict[str, Any] | None:
        return self._parse_json_interpretation(text, "chart")

    def _parse_json_interpretation(self, text: str | None, label: str) -> dict[str, Any] | None:
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Gemini %s interpretation returned invalid JSON: %s", label, cleaned[:300])
            return None
        return payload if isinstance(payload, dict) else None

    def _generate_content_with_retry(self, model: str, contents: list[Any], config: Any) -> Any:
        attempts = max(1, self.settings.gemini_retry_attempts)
        last_exception: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                return self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                last_exception = exc
                if not self._is_retryable_gemini_error(exc) or attempt >= attempts:
                    break
                delay = self._retry_delay_seconds(attempt)
                logger.warning(
                    "Gemini temporary failure on attempt %s/%s, retrying in %.2fs: %s",
                    attempt,
                    attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)

        if last_exception and self._is_retryable_gemini_error(last_exception):
            raise GeminiTemporarilyUnavailableError(str(last_exception)) from last_exception
        if last_exception:
            raise last_exception
        raise RuntimeError("Gemini generate_content returned without a response or exception.")

    def _retry_delay_seconds(self, attempt: int) -> float:
        base_delay = max(0.1, self.settings.gemini_retry_base_delay_seconds)
        return base_delay * (2 ** (attempt - 1))

    def _is_retryable_gemini_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code in {429, 500, 503, 504}:
            return True

        message = str(exc).upper()
        retry_markers = [
            "503",
            "UNAVAILABLE",
            "HIGH DEMAND",
            "429",
            "RESOURCE_EXHAUSTED",
            "TIMEOUT",
            "INTERNAL",
        ]
        return any(marker in message for marker in retry_markers)

    def _build_config(self, types: Any, include_tools: bool) -> Any:
        kwargs: dict[str, Any] = {
            "system_instruction": SYSTEM_PROMPT,
            "max_output_tokens": self.settings.gemini_max_output_tokens,
        }
        if include_tools:
            kwargs["tools"] = [types.Tool(function_declarations=FUNCTION_DECLARATIONS)]
        if self.settings.gemini_temperature is not None:
            kwargs["temperature"] = self.settings.gemini_temperature
        if self.settings.gemini_thinking_level and hasattr(types, "ThinkingConfig"):
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.settings.gemini_thinking_level
            )
        return types.GenerateContentConfig(**kwargs)

    def _build_contents(
        self,
        types: Any,
        user_message: str,
        chat_id: str | None,
        user_name: str | None,
        market_snapshot: dict[str, Any] | None = None,
    ) -> list[Any]:
        contents = []
        for message in self.memory.snapshot(chat_id):
            contents.append(
                types.Content(
                    role=message.role,
                    parts=[types.Part(text=message.text)],
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=self._current_turn_text(
                            user_message,
                            chat_id,
                            user_name,
                            market_snapshot=market_snapshot,
                        )
                    )
                ],
            )
        )
        return contents

    def _current_turn_text(
        self,
        user_message: str,
        chat_id: str | None,
        user_name: str | None,
        market_snapshot: dict[str, Any] | None = None,
    ) -> str:
        try:
            now = datetime.now(ZoneInfo(self.settings.timezone))
        except ZoneInfoNotFoundError:
            now = datetime.utcnow()
        timestamp = now.strftime("%Y-%m-%d %H:%M %Z").strip()
        active_asset = self._infer_active_asset(chat_id, user_message)
        lines = [f"Tarih/saat: {timestamp}"]
        if user_name:
            lines.append(f"Kullanici ad tercihi: {user_name}")
        else:
            lines.append("Kullanici ad tercihi: bilinmiyor")
        if active_asset:
            lines.append(f"Aktif varlik baglami: {active_asset}")
        previous_user_message = self._previous_user_message(chat_id)
        if previous_user_message and self._is_context_followup_message(user_message.lower()):
            lines.append(
                "Bu mesaj onceki konunun devami gibi gorunuyor. Onceki kullanici "
                f"mesaji: {previous_user_message}"
            )
        lines.append(f"Birincil cevap tercihi: {self._response_preference_hint(user_message)}")
        if market_snapshot is not None:
            lines.append(
                "Guncel market verisi asagida verildi. Fiyat cevabinda yalnizca bu "
                "veriyi kullan; fiyat uydurma, tarih sorma, kodun hazir cumlesi gibi "
                "degil dogal bir sohbet cevabi yaz."
            )
            lines.append(json.dumps(self._compact_market_snapshot(market_snapshot), ensure_ascii=False))
        lines.append(f"Kullanici mesaji: {user_message}")
        return "\n".join(lines)

    def _compact_market_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        quotes = []
        for quote in (snapshot.get("quotes") or [])[:8]:
            quotes.append(
                {
                    "symbol": quote.get("symbol"),
                    "requested_symbol": quote.get("requested_symbol"),
                    "name": quote.get("name"),
                    "price": quote.get("price"),
                    "previous_close": quote.get("previous_close"),
                    "change": quote.get("change"),
                    "change_percent": quote.get("change_percent"),
                    "currency": quote.get("currency"),
                    "market_time": quote.get("market_time"),
                    "timezone": quote.get("timezone"),
                }
            )
        return {
            "provider": snapshot.get("provider"),
            "fetched_at": snapshot.get("fetched_at"),
            "note": snapshot.get("note"),
            "quotes": quotes,
            "derived_metrics": snapshot.get("derived_metrics") or {},
            "errors": snapshot.get("errors") or [],
        }

    def _remember_user_name(self, chat_id: str | None, user_message: str) -> str | None:
        detected_name = self._extract_explicit_name(user_message)
        if detected_name:
            self.memory.set_preferred_name(chat_id, detected_name)
        return detected_name

    def _extract_explicit_name(self, text: str) -> str | None:
        patterns = [
            r"\bbenim adim\s+([A-Za-zCÇGĞIİÖŞUÜçğıöşü]{2,24})\b",
            r"\badim\s+([A-Za-zCÇGĞIİÖŞUÜçğıöşü]{2,24})\b",
            r"\bismim\s+([A-Za-zCÇGĞIİÖŞUÜçğıöşü]{2,24})\b",
            r"\bbana\s+([A-Za-zCÇGĞIİÖŞUÜçğıöşü]{2,24})\s+de\b",
            r"\bbana\s+([A-Za-zCÇGĞIİÖŞUÜçğıöşü]{2,24})\s+diyebilirsin\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._normalize_name(match.group(1))
        return None

    def _normalize_name(self, name: str) -> str:
        clean = name.strip(" .,!?:;")
        if not clean:
            return clean
        return clean[:1].upper() + clean[1:].lower()

    def _response_preference_hint(self, user_message: str) -> str:
        lowered = user_message.lower()
        if self._is_context_followup_message(lowered):
            return "Bu mesaj onceki sorunun devami olabilir. Onceki konuyu koru; kullanici degerlendirme istiyorsa ayni haber/varlik uzerinden iyi-kotu, olumlu-olumsuz ve riskleri anlat."
        if any(token in lowered for token in ["ons", "ounce"]):
            return "Kullanici ons odakli soruyor. Ozellikle baska bir para birimi istemedikce ons fiyatini USD cinsinden ver."
        if any(token in lowered for token in ["gram", "kilo", "kilosu", "kg", "kilogram"]):
            return "Kullanici gram veya kilogram odakli soruyor. Ozellikle baska bir para birimi istemedikce yerel fiyatlari TL cinsinden ver."
        if any(token in lowered for token in ["varil", "ton", "tonu"]):
            return "Kullanici emtia birimi soruyor. Ozellikle baska bir para birimi istemedikce ilgili uluslararasi fiyat birimiyle cevap ver."
        if any(token in lowered for token in ["tl", "try", "lira"]):
            return "Sonucu once TL cinsinden ver."
        if any(token in lowered for token in ["usd", "dolar", "dollar"]):
            return "Sonucu once USD cinsinden ver."
        if self._looks_turkish(user_message):
            return "Kullanici Turkce yaziyor. Ozellikle baska bir sey istemedikce yerel fiyat tercihi TL ve altinda varsayilan format gram/TL."
        return "Kullanici Ingilizce veya Turkce disi yaziyor. Ozellikle baska bir sey istemedikce fiyat tercihi USD."

    def _is_news_question(self, user_message: str) -> bool:
        lowered = self._normalize_news_text(user_message).lower()
        explanation_markers = [
            "haberi açıkla",
            "haberi acikla",
            "haberini açıkla",
            "haberini acikla",
            "haberi anlat",
            "haberini anlat",
            "haberi yorumla",
            "haberini yorumla",
            "haberi ne oluyor",
            "haber ne oluyor",
            "ne anlama geliyor",
            "ne anlama gelir",
        ]
        if any(marker in lowered for marker in explanation_markers):
            return False

        explicit_news_markers = [
            "son haberler",
            "haberleri neler",
            "haberler neler",
            "haberleri ver",
            "haberleri getir",
            "haberleri çek",
            "haberleri cek",
            "haber çek",
            "haber cek",
            "haberlere bak",
            "haberlerine bak",
            "haberleri kontrol",
            "güncel haberler",
            "guncel haberler",
            "son gelişmeler",
            "son gelismeler",
        ]
        if any(marker in lowered for marker in explicit_news_markers):
            return True
        return False

    def _answer_news_question(self, user_message: str, chat_id: str | None) -> str:
        query = self._news_query_for_message(user_message, chat_id)
        snapshot = self.news_search.search(query, limit=self.settings.news_max_items)
        return self._format_news_snapshot(snapshot, heading=f"{query} icin son haberler:")

    def _news_query_for_message(self, user_message: str, chat_id: str | None) -> str:
        normalized_message = self._normalize_news_text(user_message)
        asset = self._infer_active_asset(chat_id, normalized_message)
        asset_queries = {
            "altin": "altin",
            "gumus": "gumus",
            "petrol": "petrol",
            "bitcoin": "bitcoin",
            "ethereum": "ethereum",
            "bist100": "BIST 100",
            "nasdaq": "Nasdaq",
            "sp500": "S&P 500",
            "usdtry": "dolar TL",
            "eurtry": "euro TL",
            "amd": "AMD",
            "nvda": "Nvidia",
            "aapl": "Apple",
            "tsla": "Tesla",
            "msft": "Microsoft",
        }
        if asset:
            return asset_queries.get(asset, asset)

        cleaned = re.sub(
            (
                r"\b(haberlerine|haberleri|haberlere|haberler|haber|son|gelişmeler|"
                r"gelismeler|gündem|gundem|neler|nedir|ne|hakkında|hakkinda|"
                r"bakabilir|bakar|bak|çeker|ceker|çek|cek|getir|misin|mısın|"
                r"musun|müsün|var mı|var mi|ilgili|bana|güncel|guncel|bugün|"
                r"bugun|bugünkü|bugunku|söyler|soyler|söyle|soyle|anlat|"
                r"verir|ver|lütfen|lutfen|durum|oldu|oluyor|olan|açıklandı|"
                r"aciklandi)\b"
            ),
            " ",
            normalized_message,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?")
        if not cleaned or self._is_generic_news_phrase(cleaned):
            return "ekonomi piyasalar"
        return self._normalize_news_query(cleaned)

    def _normalize_news_text(self, text: str) -> str:
        normalized = re.sub(r"\bahber", "haber", text, flags=re.IGNORECASE)
        normalized = re.sub(r"\bahbar", "haber", normalized, flags=re.IGNORECASE)
        return normalized

    def _normalize_news_query(self, text: str) -> str:
        replacements = {
            "abd": "ABD",
            "fed": "Fed",
            "tcmb": "TCMB",
            "tüik": "TÜİK",
            "tuik": "TÜİK",
            "ecb": "ECB",
        }
        words = []
        for word in text.split():
            clean = word.strip(" .,!?:;")
            words.append(replacements.get(clean.lower(), clean))
        return " ".join(word for word in words if word)

    def _is_generic_news_phrase(self, text: str) -> bool:
        generic_words = {
            "haber",
            "haberler",
            "haberlere",
            "haberleri",
            "bak",
            "bakar",
            "cek",
            "ceker",
            "çek",
            "çeker",
            "getir",
            "misin",
            "mısın",
            "musun",
            "müsün",
            "son",
            "guncel",
            "güncel",
            "piyasa",
            "piyasalar",
            "piyasalarda",
            "ekonomi",
            "ekonomisi",
            "ekonomide",
        }
        words = [word.strip(" .,!?:;").lower() for word in text.split()]
        return bool(words) and all(word in generic_words for word in words)

    def _is_macro_news_status_question(self, lowered_message: str) -> bool:
        if self._extract_asset_label(lowered_message):
            return False
        topic_markers = [
            "abd",
            "amerika",
            "avrupa",
            "çin",
            "cin",
            "enflasyon",
            "faiz",
            "fed",
            "tcmb",
            "merkez bank",
            "ecb",
            "tüik",
            "tuik",
            "resesyon",
            "büyüme",
            "buyume",
            "işsizlik",
            "issizlik",
            "istihdam",
            "cari açık",
            "cari acik",
            "bütçe",
            "butce",
            "tahvil",
            "kredi",
            "konut",
            "enerji",
            "ihracat",
            "ithalat",
            "ekonomi",
            "piyasa",
            "piyasalar",
        ]
        status_markers = [
            "güncel",
            "guncel",
            "son durum",
            "son gelişme",
            "son gelisme",
            "neler oluyor",
            "ne oluyor",
            "ne oldu",
            "gündem",
            "gundem",
            "açıklandı",
            "aciklandi",
            "karar",
        ]
        return self._mentions_any(lowered_message, topic_markers) and self._mentions_any(
            lowered_message,
            status_markers,
        )

    def _format_news_snapshot(self, snapshot: dict[str, Any], heading: str) -> str:
        items = snapshot.get("items") or []
        if not items:
            return "Su an ilgili konuda guncel haber cekemedim. Biraz sonra tekrar deneyebiliriz."

        lines = [heading]
        for index, item in enumerate(items[: self.settings.news_max_items], start=1):
            title = item.get("title") or "Baslik yok"
            summary = item.get("summary") or "Bu haber ilgili piyasadaki son gelismeye deginiyor."
            link = item.get("link") or ""
            source = item.get("source") or self._link_label(link)
            lines.append(f"{index}. {title}")
            lines.append(f"Ozet: {summary}")
            if link:
                lines.append(f"Kaynak: <a href=\"{link}\">{source}</a>")
        return "\n".join(lines)

    def _link_label(self, link: str) -> str:
        if not link:
            return "Haberi oku"
        host = urlparse(link).netloc.replace("www.", "")
        return host or "Haberi oku"

    def _news_for_large_market_move(
        self,
        user_message: str,
        snapshot: dict[str, Any],
        chat_id: str | None,
    ) -> dict[str, Any] | None:
        main_quote = self._first_non_currency_quote(snapshot)
        if not main_quote:
            main_quote = self._find_quote(snapshot, "USDTRY=X") or self._find_quote(snapshot, "EURTRY=X")
        if not main_quote:
            return None

        change_percent = _first_numeric_value(main_quote.get("change_percent"))
        if change_percent is None:
            return None
        if abs(change_percent) < self.settings.news_move_threshold_percent:
            return None

        query = self._news_query_for_market_quote(main_quote, user_message, chat_id)
        snapshot = self.news_search.search(query, limit=3)
        if snapshot.get("status") != "ok":
            return None
        return snapshot

    def _news_query_for_market_quote(
        self,
        quote: dict[str, Any],
        user_message: str,
        chat_id: str | None,
    ) -> str:
        asset = self._infer_active_asset(chat_id, user_message)
        if asset:
            return self._news_query_for_message(asset, chat_id=None)
        name = quote.get("name") or quote.get("symbol") or "piyasa"
        return str(name)

    def _prefetch_market_snapshot(self, user_message: str, chat_id: str | None) -> dict[str, Any] | None:
        symbols = self._extract_prefetch_symbols(user_message, chat_id)
        if not symbols:
            return None
        result = self.market_data.get_snapshot(symbols=symbols)
        if result.get("status") != "ok" or not result.get("quotes"):
            return None
        return result

    def _extract_prefetch_symbols(self, user_message: str, chat_id: str | None) -> list[str]:
        lowered = user_message.lower()
        active_asset = self._infer_active_asset(chat_id, user_message)
        symbols: list[str] = []

        if self._mentions_any(lowered, ["dolar/tl", "usdtry", "usd/try", "dolar kaç tl", "dolar kac tl"]):
            symbols.append("USDTRY")
        elif (
            "dolar" in lowered
            and self._mentions_any(lowered, ["tl", "try", "lira"])
            and not self._is_context_followup_message(lowered)
        ):
            symbols.append("USDTRY")

        if self._mentions_any(lowered, ["euro/tl", "eurtry", "eur/tl", "euro kaç tl", "euro kac tl"]):
            symbols.append("EURTRY")
        elif "euro" in lowered and self._mentions_any(lowered, ["tl", "try", "lira"]):
            symbols.append("EURTRY")

        if self._mentions_any(lowered, ["altın", "altin", "gold", "xau"]) or active_asset == "altin":
            symbols.append("GOLD")
        if self._mentions_any(lowered, ["gümüş", "gumus", "silver", "xag"]) or active_asset == "gumus":
            symbols.append("SILVER")
        if self._mentions_any(lowered, ["brent", "petrol", "varil", "oil"]) or active_asset == "petrol":
            symbols.append("BRENT")
        if self._mentions_any(lowered, ["wti"]):
            symbols.append("WTI")
        if self._mentions_any(lowered, ["bitcoin", "btc"]) or active_asset == "bitcoin":
            symbols.append("BTCUSD")
        if self._mentions_any(lowered, ["ethereum", "eth"]) or active_asset == "ethereum":
            symbols.append("ETHUSD")
        if self._mentions_any(lowered, ["nasdaq"]):
            symbols.append("NASDAQ")
        if self._mentions_any(lowered, ["s&p", "sp500", "s&p 500"]):
            symbols.append("SP500")
        if self._mentions_any(lowered, ["bist", "xu100", "bist100"]):
            symbols.append("BIST100")
        if self._mentions_any(lowered, ["amd"]):
            symbols.append("AMD")
        if self._mentions_any(lowered, ["nvidia", "nvda"]):
            symbols.append("NVDA")
        if self._mentions_any(lowered, ["apple", "aapl"]):
            symbols.append("AAPL")
        if self._mentions_any(lowered, ["tesla", "tsla"]):
            symbols.append("TSLA")
        if self._mentions_any(lowered, ["microsoft", "msft"]):
            symbols.append("MSFT")
        if active_asset in {"nasdaq", "sp500", "bist100", "amd", "nvda", "aapl", "tsla", "msft"}:
            symbols.append(active_asset.upper())

        if not symbols:
            return []

        price_markers = [
            "fiyat",
            "kaç",
            "kac",
            "ne kadar",
            "kaç tl",
            "kac tl",
            "kaç dolar",
            "kac dolar",
            "kaç usd",
            "kac usd",
            "anlık",
            "anlik",
            "güncel",
            "guncel",
            "kaç lira",
            "kac lira",
            "kaç euro",
            "kac euro",
            "gram",
            "ons",
            "ounce",
            "son durum",
            "ne durumda",
            "durum ne",
            "seyir",
            "seyrediyor",
            "seviye",
            "seviyesi",
            "islem",
            "işlem",
            "kilo",
            "kilosu",
            "kg",
            "varil",
            "tonu",
            "ton",
        ]
        if not any(marker in lowered for marker in price_markers) and len(lowered.split()) > 3:
            return []
        return list(dict.fromkeys(symbols))

    def _is_broad_market_status_question(self, lowered_message: str) -> bool:
        return self._mentions_any(
            lowered_message,
            [
                "son durum",
                "ne durumda",
                "durum ne",
                "piyasa",
                "guncel durum",
                "güncel durum",
                "seyir",
                "seyrediyor",
            ],
        )

    def _mentions_any(self, text: str, options: list[str]) -> bool:
        return any(option in text for option in options)

    def _looks_turkish(self, text: str) -> bool:
        lowered = text.lower()
        turkish_markers = [
            "altin",
            "fiyat",
            "kadar",
            "gram",
            "ons",
            "tl",
            "lira",
            "ne",
            "bugun",
            "su an",
            "kaç",
            "guncel",
        ]
        turkish_chars = any(char in lowered for char in "çğıöşü")
        marker_match = sum(1 for marker in turkish_markers if marker in lowered)
        return turkish_chars or marker_match >= 2

    def _infer_active_asset(self, chat_id: str | None, user_message: str) -> str | None:
        current_asset = self._extract_asset_label(user_message)
        if current_asset:
            return current_asset

        lowered = user_message.lower().strip()
        if not self._is_context_followup_message(lowered):
            return None

        for message in reversed(self.memory.snapshot(chat_id)):
            if message.text:
                asset = self._extract_asset_label(message.text)
                if asset:
                    return asset
        return None

    def _infer_active_unit(self, chat_id: str | None, user_message: str) -> str | None:
        current_unit = self._extract_unit_label(user_message)
        if current_unit:
            return current_unit

        lowered = user_message.lower().strip()
        if not self._is_context_followup_message(lowered):
            return None

        for message in reversed(self.memory.snapshot(chat_id)):
            if message.text:
                unit = self._extract_unit_label(message.text)
                if unit:
                    return unit
        return None

    def _is_short_followup_message(self, lowered_message: str) -> bool:
        canonical = lowered_message.strip().replace("?", "")
        return canonical in {
            "gram",
            "ons",
            "ounce",
            "tl",
            "try",
            "usd",
            "dolar",
            "lira",
            "kilo",
            "kilosu",
            "kg",
            "kilogram",
            "varil",
            "tonu",
            "ton",
            "adet",
            "lot",
        }

    def _is_context_followup_message(self, lowered_message: str) -> bool:
        canonical = lowered_message.strip().replace("?", "")
        if self._is_short_followup_message(canonical):
            return True
        followup_markers = [
            "iyi mi",
            "kötü mü",
            "kotu mu",
            "iyi bir şey mi",
            "iyi bir sey mi",
            "kötü bir şey mi",
            "kotu bir sey mi",
            "olumlu mu",
            "olumsuz mu",
            "pozitif mi",
            "negatif mi",
            "riskli mi",
            "ne anlama geliyor",
            "ne anlama gelir",
            "bu ne demek",
            "bu ne demek oluyor",
            "bunun etkisi ne",
            "etkisi ne olur",
            "hisseye etkisi ne",
            "yatırımcı için ne demek",
            "yatirimci icin ne demek",
            "kaç tl ediyor",
            "kac tl ediyor",
            "tl ediyor",
            "kaç lira ediyor",
            "kac lira ediyor",
            "lira ediyor",
            "tl karşılığı",
            "tl karsiligi",
            "lira karşılığı",
            "lira karsiligi",
            "kaç dolar ediyor",
            "kac dolar ediyor",
            "dolar ediyor",
            "usd ediyor",
            "usd karşılığı",
            "usd karsiligi",
            "ne ediyor",
            "karşılığı ne",
            "karsiligi ne",
            "haberlere bak",
            "haberlerine bak",
            "haberleri kontrol",
            "haberleri var",
            "son haberler",
        ]
        return any(marker in canonical for marker in followup_markers)

    def _previous_user_message(self, chat_id: str | None) -> str | None:
        for message in reversed(self.memory.snapshot(chat_id)):
            if message.role == "user" and message.text:
                return message.text
        return None

    def _extract_asset_label(self, text: str) -> str | None:
        lowered = text.lower()
        asset_aliases = [
            ("altin", ["altın", "altin", "gold", "xau", "ons altin", "ons altın", "gram altin", "gram altın"]),
            ("gumus", ["gümüş", "gumus", "silver", "xag"]),
            ("petrol", ["petrol", "brent", "wti", "oil"]),
            ("bitcoin", ["bitcoin", "btc"]),
            ("ethereum", ["ethereum", "eth"]),
            ("bist100", ["bist", "bist100", "xu100"]),
            ("nasdaq", ["nasdaq"]),
            ("sp500", ["s&p", "sp500", "s&p 500"]),
            ("usdtry", ["usdtry", "dolar/tl", "usd/try"]),
            ("eurtry", ["eurtry", "euro", "eur/tl"]),
            ("amd", ["amd", "advanced micro devices"]),
            ("nvda", ["nvda", "nvidia"]),
            ("aapl", ["aapl", "apple"]),
            ("tsla", ["tsla", "tesla"]),
            ("msft", ["msft", "microsoft"]),
        ]
        for label, aliases in asset_aliases:
            if any(alias in lowered for alias in aliases):
                return label
        return None

    def _extract_unit_label(self, text: str) -> str | None:
        lowered = text.lower()
        unit_aliases = [
            ("ons", ["ons", "ounce"]),
            ("gram", ["gram"]),
            ("kilo", ["kilo", "kilosu", "kg", "kilogram"]),
            ("varil", ["varil"]),
            ("ton", ["ton", "tonu"]),
        ]
        for label, aliases in unit_aliases:
            if any(alias in lowered for alias in aliases):
                return label
        return None

    def _extract_function_calls(self, response: Any) -> list[Any]:
        calls = getattr(response, "function_calls", None)
        if calls:
            return list(calls)

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return []
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        return [part.function_call for part in parts if getattr(part, "function_call", None)]

    def _extract_text(self, response: Any) -> str:
        return self._extract_text_or_none(response) or NO_TEXT_FALLBACK

    def _extract_text_or_none(self, response: Any) -> str | None:
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        parts = getattr(candidates[0].content, "parts", None) or []
        chunks = [part.text for part in parts if getattr(part, "text", None)]
        return "\n".join(chunks).strip() or None

    def _first_non_currency_quote(self, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        for quote in snapshot.get("quotes") or []:
            symbol = quote.get("symbol")
            if symbol not in {"USDTRY=X", "EURTRY=X", "EURUSD=X"}:
                return quote
        return None

    def _find_quote(self, snapshot: dict[str, Any], symbol: str) -> dict[str, Any] | None:
        for quote in snapshot.get("quotes") or []:
            if quote.get("symbol") == symbol:
                return quote
        return None

    def _execute_tool(self, function_call: Any, chat_id: str | None = None) -> dict[str, Any]:
        name = getattr(function_call, "name", "")
        args = dict(getattr(function_call, "args", {}) or {})
        try:
            if name == "get_market_snapshot":
                return self.market_data.get_snapshot(symbols=args.get("symbols"))
            if name == "search_knowledge_base":
                return self.knowledge_base.search(
                    query=str(args.get("query", "")),
                    limit=args.get("limit"),
                )
            if name == "subscribe_newsletter":
                return self._subscribe_newsletter(args, chat_id=chat_id)
            return {"status": "error", "message": f"Unknown tool: {name}"}
        except Exception as exc:
            logger.exception("Tool execution failed: %s", name)
            return {"status": "error", "message": str(exc), "tool": name}

    def _subscribe_newsletter(self, args: dict[str, Any], chat_id: str | None) -> dict[str, Any]:
        full_name = str(args.get("full_name") or "").strip()
        email = str(args.get("email") or "").strip().lower()
        consent_text = str(args.get("consent_text") or "").strip()
        if not full_name:
            return {"status": "missing_fields", "message": "Ad soyad eksik.", "missing": ["full_name"]}
        if not _looks_like_email(email):
            return {"status": "missing_fields", "message": "Gecerli email adresi eksik.", "missing": ["email"]}
        webhook_url = self.settings.zapier_newsletter_webhook_url
        if not webhook_url:
            return {
                "status": "not_configured",
                "message": "ZAPIER_NEWSLETTER_WEBHOOK_URL ortam degiskeni tanimli degil.",
            }

        import requests

        payload = {
            "full_name": full_name,
            "email": email,
            "consent_text": consent_text,
            "source": "telegram",
            "chat_id": chat_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=max(3.0, self.settings.request_timeout_seconds),
        )
        if response.status_code >= 400:
            return {
                "status": "error",
                "message": f"Zapier webhook {response.status_code} dondu.",
                "provider": "zapier",
            }
        return {
            "status": "ok",
            "message": "Bulten kaydi Zapier endpointine gonderildi.",
            "email": email,
            "full_name": full_name,
        }

    def subscribe_newsletter(
        self,
        full_name: str,
        email: str,
        consent_text: str = "",
        chat_id: str | None = None,
    ) -> dict[str, Any]:
        return self._subscribe_newsletter(
            {
                "full_name": full_name,
                "email": email,
                "consent_text": consent_text,
            },
            chat_id=chat_id,
        )

    def _function_response_part(self, types: Any, function_call: Any, result: dict[str, Any]) -> Any:
        kwargs: dict[str, Any] = {
            "name": function_call.name,
            "response": {"result": result},
        }
        call_id = getattr(function_call, "id", None)
        if call_id:
            kwargs["id"] = call_id
        try:
            return types.Part.from_function_response(**kwargs)
        except TypeError:
            kwargs.pop("id", None)
            return types.Part.from_function_response(**kwargs)


def _first_numeric_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _looks_like_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))
