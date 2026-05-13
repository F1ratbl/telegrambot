from src.tools.market_data import calculate_change, normalize_symbol, MarketDataClient, MarketQuote
from src.tools.charting import PriceChartTool
from src.tools.news import NewsItem, NewsSearchClient, _filter_relevant_items, _rss_query_candidates
from src.tools.visual_generation import EconomyVisualGenerator
from src.bot.telegram import TelegramClient, _sanitize_telegram_text
from src.bot.webhook import _handle_update
from src.bot.memory import InMemoryConversationMemory
from src.audio.speech import SpeechServiceError
from src.ai.agent import EconomyAgent, START_MESSAGE
from src.config import Settings


class _DummyTool:
    pass


class _FakeNews:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, limit: int | None = None) -> dict:
        self.queries.append(query)
        return {
            "status": "ok",
            "provider": "fake",
            "query": query,
            "items": [
                {
                    "title": f"{query} piyasasinda son gelisme",
                    "summary": f"{query} hakkindaki haber fiyatlamayi etkileyen son gelismeye odaklaniyor.",
                    "link": "https://example.com/news",
                }
            ],
        }


class _FakeMarket:
    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot

    def get_snapshot(self, symbols=None) -> dict:
        return self.snapshot


class _FakeHttpResponse:
    def __init__(
        self,
        status_code: int,
        text: str = "",
        data: dict | None = None,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._data = data or {}
        self.content = content
        self.headers: dict[str, str] = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} error")

    def json(self) -> dict:
        return self._data


class _FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.candidates = []
        self.function_calls = []


class _RecordingGeminiModels:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeGeminiResponse(self.reply)


class _FakeGeminiClient:
    def __init__(self, reply: str) -> None:
        self.models = _RecordingGeminiModels(reply)


class _FailingImageModels:
    def generate_images(self, model, prompt, config):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


class _FailingImageClient:
    models = _FailingImageModels()


class _FakeGeneratedImageBytes:
    image_bytes = b"imagen-bytes"


class _FakeGeneratedImage:
    image = _FakeGeneratedImageBytes()


class _FakeGenerateImagesResponse:
    generated_images = [_FakeGeneratedImage()]


class _RecordingImageModels:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_images(self, model, prompt, config):
        self.calls.append({"model": model, "prompt": prompt, "config": config})
        return _FakeGenerateImagesResponse()


class _RecordingImageClient:
    def __init__(self) -> None:
        self.models = _RecordingImageModels()


class _FakeAgent:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.chat_ids: list[str | None] = []

    def reply(self, user_message: str, chat_id: str | None = None) -> str:
        self.messages.append(user_message)
        self.chat_ids.append(chat_id)
        return f"cevap: {user_message}"


class _FakeTelegram:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.voices: list[dict] = []
        self.photos: list[dict] = []
        self.downloaded_file_ids: list[str] = []

    def send_message(self, chat_id, text, reply_to_message_id=None) -> None:
        self.messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "reply_to_message_id": reply_to_message_id,
            }
        )

    def send_voice(self, chat_id, audio, reply_to_message_id=None, caption=None) -> None:
        self.voices.append(
            {
                "chat_id": chat_id,
                "audio": audio,
                "reply_to_message_id": reply_to_message_id,
                "caption": caption,
            }
        )

    def send_photo(self, chat_id, image, reply_to_message_id=None, caption=None) -> None:
        self.photos.append(
            {
                "chat_id": chat_id,
                "image": image,
                "reply_to_message_id": reply_to_message_id,
                "caption": caption,
            }
        )

    def download_file(self, file_id: str) -> bytes:
        self.downloaded_file_ids.append(file_id)
        return b"voice-bytes"


class _FakeSTT:
    enabled = True

    def transcribe(self, audio: bytes, mimetype: str | None = None) -> str:
        assert audio == b"voice-bytes"
        assert mimetype == "audio/ogg"
        return "altin kac tl"


class _FakeTTS:
    enabled = True

    def synthesize(self, text: str) -> bytes:
        assert text == "cevap: altin kac tl"
        return b"mp3-bytes"


class _FakeChartTool:
    def parse_request(self, text: str):
        return {"symbol": "GOLD"} if "grafik" in text.lower() else None

    def create_price_chart(self, request):
        return b"chart-bytes", "Altin grafigi"


class _FakeVisualGenerator:
    def parse_request(self, text: str):
        return text if "infografik" in text.lower() else None

    def generate(self, request_text: str):
        return b"visual-bytes", "Ekonomi gorseli"


class _FailingTTS:
    enabled = True

    def synthesize(self, text: str) -> bytes:
        assert text == "cevap: altin kac tl"
        raise SpeechServiceError(
            "ElevenLabs",
            402,
            "Free users cannot use library voices via the API.",
        )


class _DisabledVoice:
    enabled = False


class _RecordingTelegramClient(TelegramClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.uploads: list[dict] = []

    def _post_file(self, method, payload, files):  # type: ignore[override]
        self.uploads.append({"method": method, "payload": payload, "files": files})
        return {"ok": True}


class _FakeTypes:
    class Part:
        def __init__(self, text: str) -> None:
            self.text = text

    class Content:
        def __init__(self, role: str, parts: list) -> None:
            self.role = role
            self.parts = parts


def _joined_gemini_content_text(contents: list) -> str:
    chunks = []
    for content in contents:
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def test_normalize_symbol_aliases() -> None:
    assert normalize_symbol("BIST 100") == "XU100.IS"
    assert normalize_symbol("s&p 500") == "^GSPC"
    assert normalize_symbol("usdtry") == "USDTRY=X"


def test_calculate_change() -> None:
    change, change_percent = calculate_change(110, 100)
    assert change == 10
    assert change_percent == 10


def test_calculate_change_without_previous_close() -> None:
    assert calculate_change(110, None) == (None, None)


def test_sanitize_telegram_text_removes_markdown_stars() -> None:
    text = """
    Guncel veriler:
    * Altin **4695,30 USD**.
    * Dolar/TL **45,21**.
    """
    sanitized = _sanitize_telegram_text(text)
    assert "**" not in sanitized
    assert "* " not in sanitized
    assert "Altin 4695,30 USD." in sanitized


def test_sanitize_telegram_text_preserves_safe_html_links() -> None:
    text = 'Kaynak: <a href="https://example.com/news?a=1&b=2">bicpara.com</a> <script>'
    sanitized = _sanitize_telegram_text(text)
    assert '<a href="https://example.com/news?a=1&amp;b=2">bicpara.com</a>' in sanitized
    assert "&lt;script&gt;" in sanitized


def test_webhook_text_message_returns_text_reply() -> None:
    agent = _FakeAgent()
    telegram = _FakeTelegram()
    handled = _handle_update(
        {
            "message": {
                "message_id": 7,
                "chat": {"id": 123},
                "from": {"id": 456, "first_name": "Firat"},
                "text": "dolar kac tl",
            }
        },
        agent,  # type: ignore[arg-type]
        telegram,  # type: ignore[arg-type]
    )
    assert handled is True
    assert agent.messages == ["dolar kac tl"]
    assert agent.chat_ids == ["123:456"]
    assert telegram.messages[0]["text"] == "cevap: dolar kac tl"
    assert telegram.voices == []


def test_webhook_voice_message_returns_voice_reply() -> None:
    agent = _FakeAgent()
    telegram = _FakeTelegram()
    handled = _handle_update(
        {
            "message": {
                "message_id": 8,
                "chat": {"id": 123},
                "from": {"id": 456, "first_name": "Firat"},
                "voice": {"file_id": "voice-file-id", "mime_type": "audio/ogg"},
            }
        },
        agent,  # type: ignore[arg-type]
        telegram,  # type: ignore[arg-type]
        _FakeSTT(),  # type: ignore[arg-type]
        _FakeTTS(),  # type: ignore[arg-type]
    )
    assert handled is True
    assert telegram.downloaded_file_ids == ["voice-file-id"]
    assert agent.messages == ["altin kac tl"]
    assert agent.chat_ids == ["123:456"]
    assert telegram.voices[0]["audio"] == b"mp3-bytes"
    assert telegram.voices[0]["reply_to_message_id"] == 8
    assert telegram.messages == []


def test_webhook_chart_request_returns_photo_without_agent() -> None:
    agent = _FakeAgent()
    telegram = _FakeTelegram()
    handled = _handle_update(
        {
            "message": {
                "message_id": 9,
                "chat": {"id": 123},
                "from": {"id": 456},
                "text": "altin son 1 ay grafik çiz",
            }
        },
        agent,  # type: ignore[arg-type]
        telegram,  # type: ignore[arg-type]
        price_chart=_FakeChartTool(),  # type: ignore[arg-type]
    )
    assert handled is True
    assert agent.messages == []
    assert telegram.photos[0]["image"] == b"chart-bytes"
    assert telegram.photos[0]["caption"] == "Altin grafigi"


def test_webhook_visual_request_returns_photo_without_agent() -> None:
    agent = _FakeAgent()
    telegram = _FakeTelegram()
    handled = _handle_update(
        {
            "message": {
                "message_id": 10,
                "chat": {"id": 123},
                "from": {"id": 456},
                "text": "bedelli sermaye artırımı infografik oluştur",
            }
        },
        agent,  # type: ignore[arg-type]
        telegram,  # type: ignore[arg-type]
        visual_generator=_FakeVisualGenerator(),  # type: ignore[arg-type]
    )
    assert handled is True
    assert agent.messages == []
    assert telegram.photos[0]["image"] == b"visual-bytes"
    assert telegram.photos[0]["caption"] == "Ekonomi gorseli"


def test_visual_generator_falls_back_when_gemini_image_quota_fails() -> None:
    generator = EconomyVisualGenerator(Settings(google_api_key="test"))
    generator._client = _FailingImageClient()

    image, caption = generator.generate("bedelli sermaye artırımı infografik oluştur")

    assert caption == "Ekonomi semasi"
    assert image.startswith(b"\x89PNG")


def test_visual_generator_uses_imagen_generate_images() -> None:
    client = _RecordingImageClient()
    generator = EconomyVisualGenerator(
        Settings(
            google_api_key="test",
            gemini_image_model="imagen-4.0-ultra-generate-001",
        )
    )
    generator._client = client

    image, caption = generator.generate("enflasyon faiz ilişkisini infografik yap")

    assert image == b"imagen-bytes"
    assert caption == "Ekonomi gorseli"
    assert client.models.calls[0]["model"] == "imagen-4.0-ultra-generate-001"
    assert "enflasyon faiz" in client.models.calls[0]["prompt"]


def test_visual_generator_uses_huggingface_before_imagen(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return _FakeHttpResponse(
            200,
            content=b"hf-image-bytes",
            headers={"content-type": "image/png"},
        )

    monkeypatch.setattr("requests.post", fake_post)
    client = _RecordingImageClient()
    generator = EconomyVisualGenerator(
        Settings(
            google_api_key="test",
            huggingface_api_key="hf_test",
            huggingface_image_model="black-forest-labs/FLUX.1-schnell",
        )
    )
    generator._client = client

    image, caption = generator.generate("hisse bölünmesini görselle anlat")

    assert image == b"hf-image-bytes"
    assert caption == "Ekonomi gorseli"
    assert calls[0]["headers"]["Authorization"] == "Bearer hf_test"
    assert "black-forest-labs/FLUX.1-schnell" in calls[0]["url"]
    assert client.models.calls == []


def test_webhook_voice_message_falls_back_to_text_when_tts_fails() -> None:
    agent = _FakeAgent()
    telegram = _FakeTelegram()
    handled = _handle_update(
        {
            "message": {
                "message_id": 8,
                "chat": {"id": 123},
                "voice": {"file_id": "voice-file-id", "mime_type": "audio/ogg"},
            }
        },
        agent,  # type: ignore[arg-type]
        telegram,  # type: ignore[arg-type]
        _FakeSTT(),  # type: ignore[arg-type]
        _FailingTTS(),  # type: ignore[arg-type]
    )
    assert handled is True
    assert telegram.voices == []
    assert "secili ElevenLabs sesi" in telegram.messages[0]["text"]
    assert "cevap: altin kac tl" in telegram.messages[0]["text"]


def test_webhook_voice_message_warns_when_voice_config_missing() -> None:
    agent = _FakeAgent()
    telegram = _FakeTelegram()
    handled = _handle_update(
        {
            "message": {
                "message_id": 9,
                "chat": {"id": 123},
                "voice": {"file_id": "voice-file-id", "mime_type": "audio/ogg"},
            }
        },
        agent,  # type: ignore[arg-type]
        telegram,  # type: ignore[arg-type]
        _DisabledVoice(),  # type: ignore[arg-type]
        _DisabledVoice(),  # type: ignore[arg-type]
    )
    assert handled is True
    assert "DEEPGRAM_API_KEY" in telegram.messages[0]["text"]
    assert telegram.voices == []


def test_telegram_sends_mp3_tts_as_audio_file() -> None:
    client = _RecordingTelegramClient(
        Settings(
            telegram_bot_token="token",
            elevenlabs_output_format="mp3_44100_128",
        )
    )
    client.send_voice(chat_id=123, audio=b"mp3")
    assert client.uploads[0]["method"] == "sendAudio"
    assert "audio" in client.uploads[0]["files"]


def test_telegram_sends_opus_tts_as_voice_note() -> None:
    client = _RecordingTelegramClient(
        Settings(
            telegram_bot_token="token",
            elevenlabs_output_format="opus_48000_32",
        )
    )
    client.send_voice(chat_id=123, audio=b"opus")
    assert client.uploads[0]["method"] == "sendVoice"
    assert "voice" in client.uploads[0]["files"]


def test_telegram_sends_photo() -> None:
    client = _RecordingTelegramClient(Settings(telegram_bot_token="token"))
    client.send_photo(chat_id=123, image=b"png", caption="Grafik")
    assert client.uploads[0]["method"] == "sendPhoto"
    assert client.uploads[0]["payload"]["caption"] == "Grafik"
    assert "photo" in client.uploads[0]["files"]


def test_price_chart_falls_back_to_stooq_when_yahoo_is_rate_limited(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "finance.yahoo.com" in url:
            return _FakeHttpResponse(429)
        return _FakeHttpResponse(
            200,
            text=(
                "Date,Open,High,Low,Close,Volume\n"
                "2026-05-01,100,110,99,105,0\n"
                "2026-05-02,105,112,101,111,0\n"
            ),
        )

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("src.tools.charting.time.sleep", lambda _: None)

    tool = PriceChartTool(Settings())
    points = tool._fetch_history("ALTIN", "1mo")

    assert len(points) == 2
    assert points[-1][1] == 111
    assert any("finance.yahoo.com" in call for call in calls)
    assert any("stooq.com" in call for call in calls)


def test_memory_stores_name_only_when_explicitly_provided() -> None:
    memory = InMemoryConversationMemory()
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    assert memory.get_preferred_name("1") is None
    agent._remember_user_name("1", "merhaba")
    assert memory.get_preferred_name("1") is None
    agent._remember_user_name("1", "benim adim firat")
    assert memory.get_preferred_name("1") == "Firat"


def test_context_followup_mentions_previous_user_message_to_gemini() -> None:
    memory = InMemoryConversationMemory()
    memory.remember_exchange(
        "chat-1",
        "IREN Limited şirketinin 2,6 milyar dolarlık tahvil ihracı açıklaması ne oluyor",
        "IREN tahvil ihracıyla fon sağlıyor.",
    )
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    contents = agent._build_contents(
        _FakeTypes,
        "iyi bir şey mi kötü bir şey mi",
        "chat-1",
        None,
    )
    content_text = _joined_gemini_content_text(contents)
    assert "Bu mesaj onceki konunun devami gibi gorunuyor" in content_text
    assert "IREN Limited şirketinin 2,6 milyar dolarlık tahvil ihracı" in content_text
    assert "iyi-kotu" in content_text


def test_gold_snapshot_builds_try_and_gram_estimates() -> None:
    client = MarketDataClient(Settings())
    quotes = [
        MarketQuote("ALTIN", "GC=F", "Gold Futures", 3100.0, 3000.0, 100.0, 3.33, "USD", None, None, None),
        MarketQuote("USDTRY", "USDTRY=X", "USD/TRY", 38.5, 38.0, 0.5, 1.31, "TRY", None, None, None),
    ]
    derived = client._build_derived_metrics(quotes, ["ALTIN"])
    assert derived["gold_ounce_usd"] == 3100.0
    assert derived["usdtry"] == 38.5
    assert derived["gold_gram_try_estimate"] > 0


def test_snapshot_expands_usd_priced_assets_with_usdtry() -> None:
    client = MarketDataClient(Settings())
    assert client._expand_requested_symbols(["NASDAQ"]) == ["NASDAQ", "USDTRY"]
    assert client._expand_requested_symbols(["SP500"]) == ["SP500", "USDTRY"]
    assert client._expand_requested_symbols(["BIST100"]) == ["BIST100", "USDTRY"]


def test_short_followup_infers_gold_context_from_memory() -> None:
    memory = InMemoryConversationMemory()
    memory.remember_exchange("chat-1", "altin fiyatı ne kadar", "Altin su an ...")
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    assert agent._infer_active_asset("chat-1", "gram") == "altin"


def test_short_kilo_followup_infers_gold_context_from_memory() -> None:
    memory = InMemoryConversationMemory()
    memory.remember_exchange("chat-1", "altin gram fiyatı nedir", "Altin gram fiyatı su an ...")
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    assert agent._infer_active_asset("chat-1", "kilosu") == "altin"


def test_start_message_is_handled_without_gemini() -> None:
    memory = InMemoryConversationMemory()
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    assert agent.reply("/start", chat_id="chat-1") == START_MESSAGE


def test_prefetch_symbols_detects_usdtry_question() -> None:
    memory = InMemoryConversationMemory()
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    assert agent._extract_prefetch_symbols("dolar kaç tl", "chat-1") == ["USDTRY"]


def test_prefetch_symbols_uses_active_asset_for_short_followup() -> None:
    memory = InMemoryConversationMemory()
    memory.remember_exchange("chat-1", "altin gram fiyati nedir", "Altin gram fiyati...")
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    assert agent._extract_prefetch_symbols("ons", "chat-1") == ["GOLD"]


def test_prefetch_symbols_uses_active_asset_for_try_conversion_followup() -> None:
    memory = InMemoryConversationMemory()
    memory.remember_exchange("chat-1", "ons altin ne kadar", "Altinin ons fiyati...")
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    assert agent._extract_prefetch_symbols("kaç tl ediyor", "chat-1") == ["GOLD"]


def test_prefetch_symbols_detects_broad_gold_status_question() -> None:
    memory = InMemoryConversationMemory()
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    assert agent._extract_prefetch_symbols("Altınla ilgili son durum nedir", "chat-1") == ["GOLD"]


def test_fetch_quote_falls_back_to_chart_when_quote_endpoint_fails() -> None:
    client = MarketDataClient(Settings())

    def fake_quote(symbol: str, requested_symbol: str) -> MarketQuote | None:
        raise RuntimeError("401 unauthorized")

    def fake_chart(symbol: str, requested_symbol: str) -> MarketQuote:
        return MarketQuote(
            requested_symbol=requested_symbol,
            symbol=symbol,
            name="Gold Futures",
            price=3200.0,
            previous_close=3150.0,
            change=50.0,
            change_percent=1.58,
            currency="USD",
            exchange="COMEX",
            market_time=None,
            timezone=None,
        )

    client._fetch_quote_from_quote_endpoint = fake_quote  # type: ignore[method-assign]
    client._fetch_quote_from_chart_endpoint = fake_chart  # type: ignore[method-assign]

    quote = client._fetch_quote("GOLD")
    assert quote.symbol == "GC=F"
    assert quote.price == 3200.0


def test_prefetched_market_snapshot_is_sent_to_gemini_for_reply() -> None:
    memory = InMemoryConversationMemory()
    market = _FakeMarket(
        {
            "status": "ok",
            "provider": "fake market",
            "fetched_at": "2026-05-11T09:00:00+00:00",
            "quotes": [
                {
                    "requested_symbol": "GOLD",
                    "symbol": "GC=F",
                    "name": "Gold Futures",
                    "price": 3200.0,
                    "currency": "USD",
                },
                {
                    "requested_symbol": "USDTRY",
                    "symbol": "USDTRY=X",
                    "name": "USD/TRY",
                    "price": 40.0,
                    "currency": "TRY",
                },
            ],
            "derived_metrics": {
                "gold_ounce_usd": 3200.0,
                "gold_gram_try_estimate": 4115.3,
            },
        }
    )
    agent = EconomyAgent(Settings(google_api_key="test"), market, _DummyTool(), memory)
    fake_gemini = _FakeGeminiClient("Model gram altini dogal bir dille anlatti.")
    agent._client = fake_gemini

    answer = agent.reply("altın kaç tl", chat_id="chat-1")

    assert answer == "Model gram altini dogal bir dille anlatti."
    content_text = _joined_gemini_content_text(fake_gemini.models.calls[0]["contents"])
    assert "Guncel market verisi" in content_text
    assert "gold_gram_try_estimate" in content_text
    assert "4115.3" in content_text
    assert "dogal bir sohbet cevabi" in content_text


def test_extract_text_returns_fallback_only_for_public_helper() -> None:
    memory = InMemoryConversationMemory()
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    response = object()
    assert agent._extract_text_or_none(response) is None
    assert agent._extract_text(response) == "Cevap olusturamadim."


def test_news_question_returns_latest_items_without_gemini() -> None:
    memory = InMemoryConversationMemory()
    news = _FakeNews()
    agent = EconomyAgent(Settings(), _DummyTool(), _DummyTool(), memory, news_search=news)
    answer = agent.reply("altin hakkinda haberler neler", chat_id="chat-1")
    assert "altin icin son haberler:" in answer
    assert "https://example.com/news" in answer
    assert news.queries == ["altin"]


def test_news_followup_uses_previous_amd_context() -> None:
    memory = InMemoryConversationMemory()
    memory.remember_exchange("chat-1", "amd neden bu kadar yukseldi", "AMD hissesi...")
    news = _FakeNews()
    agent = EconomyAgent(Settings(), _DummyTool(), _DummyTool(), memory, news_search=news)
    answer = agent.reply("haberlere bakar misin", chat_id="chat-1")
    assert "AMD icin son haberler:" in answer
    assert news.queries == ["AMD"]


def test_generic_news_request_uses_economy_market_query_without_context() -> None:
    memory = InMemoryConversationMemory()
    news = _FakeNews()
    agent = EconomyAgent(Settings(), _DummyTool(), _DummyTool(), memory, news_search=news)
    answer = agent.reply("haberlere bakar mısın", chat_id="chat-1")
    assert "ekonomi piyasalar icin son haberler:" in answer
    assert news.queries == ["ekonomi piyasalar"]


def test_generic_news_fetch_request_uses_economy_market_query_without_context() -> None:
    memory = InMemoryConversationMemory()
    news = _FakeNews()
    agent = EconomyAgent(Settings(), _DummyTool(), _DummyTool(), memory, news_search=news)
    answer = agent.reply("haber çeker misin", chat_id="chat-1")
    assert "ekonomi piyasalar icin son haberler:" in answer
    assert news.queries == ["ekonomi piyasalar"]


def test_news_question_handles_common_haber_typo() -> None:
    memory = InMemoryConversationMemory()
    news = _FakeNews()
    agent = EconomyAgent(Settings(), _DummyTool(), _DummyTool(), memory, news_search=news)
    answer = agent.reply("güncel ekonomi ahberleri neler", chat_id="chat-1")
    assert "ekonomi piyasalar icin son haberler:" in answer
    assert news.queries == ["ekonomi piyasalar"]


def test_news_query_cleans_economic_topic_request() -> None:
    memory = InMemoryConversationMemory()
    agent = EconomyAgent(Settings(), _DummyTool(), _DummyTool(), memory)
    query = agent._news_query_for_message(
        "abd enflasyon hakkında güncel haberleri söyler misin",
        "chat-1",
    )
    assert query == "ABD enflasyon"


def test_macro_status_question_does_not_fetch_news_without_explicit_request() -> None:
    memory = InMemoryConversationMemory()
    news = _FakeNews()
    agent = EconomyAgent(Settings(), _DummyTool(), _DummyTool(), memory, news_search=news)
    answer = agent.reply("ABD enflasyon son durum ne", chat_id="chat-1")
    assert "Gemini API anahtari" in answer
    assert news.queries == []


def test_generic_market_status_question_does_not_fetch_news_without_explicit_request() -> None:
    memory = InMemoryConversationMemory()
    news = _FakeNews()
    agent = EconomyAgent(Settings(), _DummyTool(), _DummyTool(), memory, news_search=news)
    answer = agent.reply("piyasalarda son durum ne", chat_id="chat-1")
    assert "Gemini API anahtari" in answer
    assert news.queries == []


def test_asset_status_question_stays_out_of_news_without_news_word() -> None:
    memory = InMemoryConversationMemory()
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    assert agent._is_news_question("altın son durum ne") is False


def test_specific_news_explanation_does_not_return_news_list() -> None:
    memory = InMemoryConversationMemory()
    news = _FakeNews()
    agent = EconomyAgent(Settings(), _DummyTool(), _DummyTool(), memory, news_search=news)
    answer = agent.reply(
        "IREN Limited şirketinin 2,6 Milyar Dolarlık Tahvil İhracı haberi ne oluyor açıklar mısın",
        chat_id="chat-1",
    )
    assert "Gemini API anahtari" in answer
    assert news.queries == []


def test_amd_why_moved_question_does_not_fetch_news_without_explicit_request() -> None:
    memory = InMemoryConversationMemory()
    news = _FakeNews()
    agent = EconomyAgent(Settings(), _DummyTool(), _DummyTool(), memory, news_search=news)
    answer = agent.reply("amd neden bu kadar yukseldi", chat_id="chat-1")
    assert "Gemini API anahtari" in answer
    assert news.queries == []


def test_large_market_move_does_not_append_news_without_explicit_request() -> None:
    memory = InMemoryConversationMemory()
    news = _FakeNews()
    market = _FakeMarket(
        {
            "status": "ok",
            "quotes": [
                {
                    "symbol": "^IXIC",
                    "name": "Nasdaq Composite",
                    "price": 24500.0,
                    "change_percent": 3.2,
                    "currency": "USD",
                },
                {
                    "symbol": "USDTRY=X",
                    "name": "USD/TRY",
                    "price": 45.0,
                    "currency": "TRY",
                },
            ],
            "derived_metrics": {},
        }
    )
    agent = EconomyAgent(Settings(google_api_key="test"), market, _DummyTool(), memory, news_search=news)
    agent._client = _FakeGeminiClient("Nasdaq icin modelin dogal piyasa cevabi.")
    answer = agent.reply("nasdaq ne kadar", chat_id="chat-1")
    assert "Nasdaq icin modelin dogal piyasa cevabi." in answer
    assert "Hareket belirgin oldugu icin son haberlerden bazilari:" not in answer
    assert news.queries == []


def test_news_summary_prefers_article_text_over_repeated_title() -> None:
    class FakeNewsSearch(NewsSearchClient):
        def _fetch_article_text(self, link: str) -> tuple[str | None, str]:
            return (
                "https://example.com/amd",
                "AMD hisseleri yapay zeka çiplerine yönelik güçlü talep ve analist hedef fiyat artışlarıyla yükseldi. Şirketin veri merkezi gelirleri yatırımcı beklentilerini destekledi.",
            )

    rss = """
    <rss><channel>
      <item>
        <title>AMD yükseldi - Kaynak</title>
        <link>https://news.google.com/rss/articles/example</link>
        <description>AMD yükseldi - Kaynak</description>
        <source>Kaynak</source>
      </item>
    </channel></rss>
    """
    client = FakeNewsSearch(Settings())
    items = client._parse_items(rss, 1)
    assert items[0].link == "https://example.com/amd"
    assert "yapay zeka çiplerine" in items[0].summary
    assert items[0].summary != items[0].title


def test_news_summary_ignores_generic_google_description() -> None:
    class FakeNewsSearch(NewsSearchClient):
        def _fetch_article_text(self, link: str) -> tuple[str | None, str]:
            return (
                link,
                "Comprehensive up-to-date news coverage, aggregated from sources all over the world by Google News.",
            )

    rss = """
    <rss><channel>
      <item>
        <title>Altın barış iyimserliği ile yükseldi - Kıbrıs Postası</title>
        <link>https://news.google.com/rss/articles/example</link>
        <description>Comprehensive up-to-date news coverage, aggregated from sources all over the world by Google News.</description>
        <source>Kıbrıs Postası</source>
      </item>
    </channel></rss>
    """
    client = FakeNewsSearch(Settings())
    items = client._parse_items(rss, 1)
    assert "Comprehensive up-to-date" not in items[0].summary
    assert "barış iyimserliği" in items[0].summary


def test_news_filter_removes_irrelevant_asset_items() -> None:
    items = [
        NewsItem(
            title="Why Microsoft stock is underperforming today",
            link="https://example.com/msft",
            source="Example",
            published_at=None,
            summary="Microsoft shares lagged the broader market.",
        ),
        NewsItem(
            title="Chip stocks rally after AMD's blowout report",
            link="https://example.com/amd",
            source="Example",
            published_at=None,
            summary="AMD results supported sentiment across semiconductor names.",
        ),
    ]

    filtered = _filter_relevant_items("AMD", items)

    assert len(filtered) == 1
    assert filtered[0].link == "https://example.com/amd"


def test_news_filter_removes_social_sources() -> None:
    items = [
        NewsItem(
            title="Borsada piyasa değeri en yüksek şirketler",
            link="https://instagram.com/example-post",
            source="instagram.com",
            published_at=None,
            summary="Sosyal medya paylaşımı.",
        ),
        NewsItem(
            title="Piyasalarda gün ortası",
            link="https://example.com/news",
            source="Ekonomi Kaynağı",
            published_at=None,
            summary="Piyasalarda gün ortası gelişmeleri takip ediliyor.",
        ),
    ]

    filtered = _filter_relevant_items("ekonomi piyasalar", items)

    assert len(filtered) == 1
    assert filtered[0].link == "https://example.com/news"


def test_turkey_economy_news_query_uses_resilient_candidates() -> None:
    candidates = _rss_query_candidates("türkiye güncel ekonomi")

    assert candidates[0] == "Türkiye ekonomi piyasalar when:7d"
    assert "Türkiye ekonomi when:7d" in candidates
    assert '"türkiye güncel ekonomi"' not in candidates[0]


def test_specific_economic_topic_uses_topic_candidates() -> None:
    candidates = _rss_query_candidates("ABD enflasyon")

    assert candidates[0] == "ABD enflasyon when:7d"
    assert "ABD enflasyon ekonomi when:7d" in candidates
    assert "ABD enflasyon when:30d" in candidates


def test_regional_economy_topic_uses_specific_candidates() -> None:
    candidates = _rss_query_candidates("çin ekonomisi")

    assert candidates[0] == "çin ekonomisi when:7d"
    assert "çin ekonomisi finans piyasalar when:7d" in candidates


def test_news_search_falls_back_when_primary_query_fails() -> None:
    class FallbackNewsSearch(NewsSearchClient):
        def __init__(self) -> None:
            super().__init__(Settings())
            self.queries: list[str] = []

        def _fetch_rss_xml(self, rss_query: str) -> str:
            self.queries.append(rss_query)
            if len(self.queries) == 1:
                raise RuntimeError("503 Server Error")
            return """
            <rss><channel>
              <item>
                <title>Türkiye ekonomisinde güncel gelişmeler - Kaynak</title>
                <link>https://news.google.com/rss/articles/example</link>
                <description>Türkiye ekonomisinde güncel gelişmeler piyasaların odağında.</description>
                <source>Kaynak</source>
              </item>
            </channel></rss>
            """

        def _fetch_article_text(self, link: str) -> tuple[str | None, str]:
            return link, ""

    client = FallbackNewsSearch()
    snapshot = client.search("türkiye güncel ekonomi", limit=1)

    assert snapshot["status"] == "ok"
    assert client.queries[:2] == [
        "Türkiye ekonomi piyasalar when:7d",
        "Türkiye ekonomi when:7d",
    ]
    assert snapshot["items"][0]["title"] == "Türkiye ekonomisinde güncel gelişmeler - Kaynak"
