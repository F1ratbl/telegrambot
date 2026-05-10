from __future__ import annotations

import logging
import re
import time
from datetime import datetime
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
            answer = self._market_snapshot_direct_answer(user_message, prefetched_market, chat_id)
            news_snapshot = self._news_for_large_market_move(user_message, prefetched_market, chat_id)
            if news_snapshot:
                answer = f"{answer}\n\n{self._format_news_snapshot(news_snapshot, heading='Hareket belirgin oldugu icin son haberlerden bazilari:')}"
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
                tool_result = self._execute_tool(function_call)
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
                parts=[types.Part(text=self._current_turn_text(user_message, chat_id, user_name))],
            )
        )
        return contents

    def _current_turn_text(self, user_message: str, chat_id: str | None, user_name: str | None) -> str:
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
        lines.append(f"Birincil cevap tercihi: {self._response_preference_hint(user_message)}")
        lines.append(f"Kullanici mesaji: {user_message}")
        return "\n".join(lines)

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
        if self._is_short_followup_message(lowered):
            return "Bu mesaj onceki sorunun kisa bir devami olabilir. Onceki varlik baglamini koru ve kullanicinin istedigi birim veya olcuye gore cevap ver."
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
        lowered = user_message.lower()
        if "neden" in lowered and any(marker in lowered for marker in ["yükseldi", "yukseldi", "düştü", "dustu"]):
            return True
        news_markers = [
            "haber",
            "haberler",
            "son gelişmeler",
            "son gelismeler",
            "gündem",
            "gundem",
            "ne oldu",
            "neden yükseldi",
            "neden yukseldi",
            "neden düştü",
            "neden dustu",
        ]
        return any(marker in lowered for marker in news_markers)

    def _answer_news_question(self, user_message: str, chat_id: str | None) -> str:
        query = self._news_query_for_message(user_message, chat_id)
        snapshot = self.news_search.search(query, limit=self.settings.news_max_items)
        return self._format_news_snapshot(snapshot, heading=f"{query} icin son haberler:")

    def _news_query_for_message(self, user_message: str, chat_id: str | None) -> str:
        asset = self._infer_active_asset(chat_id, user_message)
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
                r"musun|müsün|var mı|var mi|ilgili|bana)\b"
            ),
            " ",
            user_message,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?")
        if not cleaned or self._is_generic_news_phrase(cleaned):
            return "ekonomi piyasalar"
        return cleaned

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
            "ekonomi",
        }
        words = [word.strip(" .,!?:;").lower() for word in text.split()]
        return bool(words) and all(word in generic_words for word in words)

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

    def _market_snapshot_direct_answer(
        self,
        user_message: str,
        snapshot: dict[str, Any],
        chat_id: str | None = None,
    ) -> str:
        lowered = user_message.lower()
        derived = snapshot.get("derived_metrics") or {}
        quotes = snapshot.get("quotes") or []
        active_unit = self._infer_active_unit(chat_id, user_message)
        wants_try = self._mentions_any(lowered, ["tl", "try", "lira"])
        wants_usd = self._mentions_any(lowered, ["usd", "dolar", "dollar"])
        gold_quote = self._find_quote(snapshot, "GC=F")
        gold_request = bool(gold_quote) and (
            self._mentions_any(lowered, ["altın", "altin", "gold", "xau"])
            or self._mentions_any(lowered, ["gram", "ons", "ounce", "kilo", "kilosu", "kg"])
            or active_unit in {"ons", "gram", "kilo"}
        )

        if gold_request:
            if self._is_broad_market_status_question(lowered):
                gram_try = _first_numeric_value(derived.get("gold_gram_try_estimate"))
                ounce_usd = _first_numeric_value(derived.get("gold_ounce_usd"))
                if ounce_usd is None:
                    ounce_usd = _first_numeric_value(gold_quote.get("price"))
                if gram_try is not None and ounce_usd is not None:
                    return (
                        "Son erisilebilir veriye gore altinda gram fiyat yaklasik "
                        f"{self._format_number(gram_try)} TL, ons fiyat ise yaklasik "
                        f"{self._format_number(ounce_usd)} USD seviyesinde."
                    )
                if gram_try is not None:
                    return f"Son erisilebilir veriye gore altinda gram fiyat yaklasik {self._format_number(gram_try)} TL seviyesinde."
                if ounce_usd is not None:
                    return f"Son erisilebilir veriye gore altinda ons fiyat yaklasik {self._format_number(ounce_usd)} USD seviyesinde."
            if active_unit == "ons" and wants_try:
                value = _first_numeric_value(derived.get("gold_ounce_try_estimate"))
                if value is not None:
                    return f"Altinin ons fiyati su an yaklasik {self._format_number(value)} TL seviyesinde."
            if active_unit == "ons" and not wants_try:
                value = _first_numeric_value(derived.get("gold_ounce_usd"))
                if value is None:
                    value = _first_numeric_value(gold_quote.get("price"))
                if value is not None:
                    return f"Altinin ons fiyati su an yaklasik {self._format_number(value)} USD seviyesinde."
            if active_unit == "gram" or self._mentions_any(lowered, ["gram"]) or (wants_try and not wants_usd):
                value = _first_numeric_value(derived.get("gold_gram_try_estimate"))
                if value is not None:
                    return f"Gram altin su an yaklasik {self._format_number(value)} TL seviyesinde."
            if self._mentions_any(lowered, ["ons", "ounce", "usd", "dolar"]):
                value = _first_numeric_value(derived.get("gold_ounce_usd"))
                if value is None:
                    value = _first_numeric_value(gold_quote.get("price"))
                if value is not None:
                    return f"Altinin ons fiyati su an yaklasik {self._format_number(value)} USD seviyesinde."
            value = _first_numeric_value(derived.get("gold_gram_try_estimate"))
            if value is not None:
                return f"Gram altin su an yaklasik {self._format_number(value)} TL seviyesinde."

        if self._mentions_any(lowered, ["euro", "eurtry", "eur/tl", "euro/tl"]):
            quote = self._find_quote(snapshot, "EURTRY=X")
            value = _first_numeric_value((quote or {}).get("price"))
            if value is not None:
                return f"Euro/TL su an yaklasik {self._format_number(value)} seviyesinde."

        main_quote = self._first_non_currency_quote(snapshot)
        if main_quote:
            try_value = self._convert_quote_to_try(main_quote, snapshot)
            usd_value = self._convert_quote_to_usd(main_quote, snapshot)
            if wants_try and try_value is not None:
                name = main_quote.get("name") or main_quote.get("symbol") or "Bu varlik"
                return f"{name} TL karsiligi su an yaklasik {self._format_number(try_value)} TL seviyesinde."
            if wants_usd and usd_value is not None:
                name = main_quote.get("name") or main_quote.get("symbol") or "Bu varlik"
                return f"{name} dolar karsiligi su an yaklasik {self._format_number(usd_value)} USD seviyesinde."
            value = _first_numeric_value(main_quote.get("price"))
            if value is not None:
                currency = main_quote.get("currency") or ""
                suffix = f" {currency}" if currency else ""
                name = main_quote.get("name") or main_quote.get("symbol") or "Bu varlik"
                return f"{name} su an yaklasik {self._format_number(value)}{suffix} seviyesinde."

        if self._mentions_any(lowered, ["dolar", "usdtry", "usd/try", "dolar/tl"]):
            quote = self._find_quote(snapshot, "USDTRY=X")
            value = _first_numeric_value((quote or {}).get("price"))
            if value is not None:
                return f"Dolar/TL su an yaklasik {self._format_number(value)} seviyesinde."

        usable_quotes = [quote for quote in quotes if _first_numeric_value(quote.get("price")) is not None]
        if usable_quotes:
            quote = usable_quotes[0]
            value = _first_numeric_value(quote.get("price"))
            currency = quote.get("currency") or ""
            name = quote.get("name") or quote.get("symbol") or "Bu varlik"
            suffix = f" {currency}" if currency else ""
            return f"{name} su an yaklasik {self._format_number(value)}{suffix} seviyesinde."

        return "Su an piyasa verisine ulasamadim. Biraz sonra tekrar deneyebilir misiniz?"

    def _first_non_currency_quote(self, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        for quote in snapshot.get("quotes") or []:
            symbol = quote.get("symbol")
            if symbol not in {"USDTRY=X", "EURTRY=X", "EURUSD=X"}:
                return quote
        return None

    def _convert_quote_to_try(self, quote: dict[str, Any], snapshot: dict[str, Any]) -> float | None:
        value = _first_numeric_value(quote.get("price"))
        if value is None:
            return None

        currency = (quote.get("currency") or "").upper()
        symbol = quote.get("symbol")
        if currency == "TRY" or str(symbol).endswith(".IS"):
            return value
        if currency != "USD":
            return None

        usdtry_quote = self._find_quote(snapshot, "USDTRY=X")
        usdtry = _first_numeric_value((usdtry_quote or {}).get("price"))
        if usdtry is None:
            return None
        return value * usdtry

    def _convert_quote_to_usd(self, quote: dict[str, Any], snapshot: dict[str, Any]) -> float | None:
        value = _first_numeric_value(quote.get("price"))
        if value is None:
            return None

        currency = (quote.get("currency") or "").upper()
        if currency == "USD":
            return value
        if currency != "TRY":
            return None

        usdtry_quote = self._find_quote(snapshot, "USDTRY=X")
        usdtry = _first_numeric_value((usdtry_quote or {}).get("price"))
        if usdtry in (None, 0):
            return None
        return value / usdtry

    def _find_quote(self, snapshot: dict[str, Any], symbol: str) -> dict[str, Any] | None:
        for quote in snapshot.get("quotes") or []:
            if quote.get("symbol") == symbol:
                return quote
        return None

    def _format_number(self, value: float | int | None) -> str:
        if value is None:
            return ""
        numeric = float(value)
        if abs(numeric) >= 1000:
            text = f"{numeric:,.2f}"
        elif abs(numeric) >= 100:
            text = f"{numeric:.2f}"
        else:
            text = f"{numeric:.4f}".rstrip("0").rstrip(".")
        return text.replace(",", "X").replace(".", ",").replace("X", ".")

    def _execute_tool(self, function_call: Any) -> dict[str, Any]:
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
            return {"status": "error", "message": f"Unknown tool: {name}"}
        except Exception as exc:
            logger.exception("Tool execution failed: %s", name)
            return {"status": "error", "message": str(exc), "tool": name}

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
