from src.tools.market_data import calculate_change, normalize_symbol, MarketDataClient, MarketQuote
from src.tools.news import NewsSearchClient
from src.bot.telegram import _sanitize_telegram_text
from src.bot.memory import InMemoryConversationMemory
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


def test_memory_stores_name_only_when_explicitly_provided() -> None:
    memory = InMemoryConversationMemory()
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    assert memory.get_preferred_name("1") is None
    agent._remember_user_name("1", "merhaba")
    assert memory.get_preferred_name("1") is None
    agent._remember_user_name("1", "benim adim firat")
    assert memory.get_preferred_name("1") == "Firat"


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


def test_market_snapshot_direct_answer_answers_gold_try() -> None:
    memory = InMemoryConversationMemory()
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    answer = agent._market_snapshot_direct_answer(
        "altın kaç tl",
        {
            "status": "ok",
            "quotes": [
                {
                    "symbol": "GC=F",
                    "name": "Gold Futures",
                    "price": 3200.0,
                    "currency": "USD",
                },
                {
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
        },
    )
    assert answer == "Gram altin su an yaklasik 4.115,30 TL seviyesinde."


def test_market_snapshot_direct_answer_uses_gold_ounce_for_followup() -> None:
    memory = InMemoryConversationMemory()
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    answer = agent._market_snapshot_direct_answer(
        "onsu ne kadar",
        {
            "status": "ok",
            "quotes": [
                {
                    "symbol": "GC=F",
                    "name": "Gold Futures",
                    "price": 4718.2,
                    "currency": "USD",
                },
                {
                    "symbol": "USDTRY=X",
                    "name": "USD/TRY",
                    "price": 45.364,
                    "currency": "TRY",
                },
            ],
            "derived_metrics": {
                "gold_ounce_usd": 4718.2,
                "gold_gram_try_estimate": 6881.4257,
            },
        },
    )
    assert answer == "Altinin ons fiyati su an yaklasik 4.718,20 USD seviyesinde."


def test_market_snapshot_direct_answer_converts_previous_gold_ounce_to_try() -> None:
    memory = InMemoryConversationMemory()
    memory.remember_exchange("chat-1", "ons altin ne kadar", "Altinin ons fiyati...")
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    answer = agent._market_snapshot_direct_answer(
        "kaç tl ediyor",
        {
            "status": "ok",
            "quotes": [
                {
                    "symbol": "GC=F",
                    "name": "Gold Futures",
                    "price": 4718.2,
                    "currency": "USD",
                },
                {
                    "symbol": "USDTRY=X",
                    "name": "USD/TRY",
                    "price": 45.364,
                    "currency": "TRY",
                },
            ],
            "derived_metrics": {
                "gold_ounce_usd": 4718.2,
                "gold_ounce_try_estimate": 214061.39,
                "gold_gram_try_estimate": 6881.4257,
            },
        },
        chat_id="chat-1",
    )
    assert answer == "Altinin ons fiyati su an yaklasik 214.061,39 TL seviyesinde."


def test_market_snapshot_direct_answer_converts_usd_index_to_try() -> None:
    memory = InMemoryConversationMemory()
    memory.remember_exchange("chat-1", "nasdaq ne kadar", "Nasdaq Composite...")
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    answer = agent._market_snapshot_direct_answer(
        "kaç tl ediyor",
        {
            "status": "ok",
            "quotes": [
                {
                    "symbol": "^IXIC",
                    "name": "Nasdaq Composite",
                    "price": 24500.0,
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
        },
        chat_id="chat-1",
    )
    assert answer == "Nasdaq Composite TL karsiligi su an yaklasik 1.102.500,00 TL seviyesinde."


def test_market_snapshot_direct_answer_keeps_bist_in_try() -> None:
    memory = InMemoryConversationMemory()
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    answer = agent._market_snapshot_direct_answer(
        "bist100 kaç tl",
        {
            "status": "ok",
            "quotes": [
                {
                    "symbol": "XU100.IS",
                    "name": "BIST 100",
                    "price": 12000.5,
                    "currency": "TRY",
                },
            ],
            "derived_metrics": {},
        },
    )
    assert answer == "BIST 100 TL karsiligi su an yaklasik 12.000,50 TL seviyesinde."


def test_market_snapshot_direct_answer_converts_try_index_to_usd() -> None:
    memory = InMemoryConversationMemory()
    memory.remember_exchange("chat-1", "bist100 kaç tl", "BIST 100...")
    agent = EconomyAgent(Settings(google_api_key="test"), _DummyTool(), _DummyTool(), memory)
    answer = agent._market_snapshot_direct_answer(
        "dolar karşılığı ne",
        {
            "status": "ok",
            "quotes": [
                {
                    "symbol": "XU100.IS",
                    "name": "BIST 100",
                    "price": 12000.0,
                    "currency": "TRY",
                },
                {
                    "symbol": "USDTRY=X",
                    "name": "USD/TRY",
                    "price": 40.0,
                    "currency": "TRY",
                },
            ],
            "derived_metrics": {},
        },
        chat_id="chat-1",
    )
    assert answer == "BIST 100 dolar karsiligi su an yaklasik 300,00 USD seviyesinde."


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


def test_amd_news_question_uses_amd_query() -> None:
    memory = InMemoryConversationMemory()
    news = _FakeNews()
    agent = EconomyAgent(Settings(), _DummyTool(), _DummyTool(), memory, news_search=news)
    answer = agent.reply("amd neden bu kadar yukseldi", chat_id="chat-1")
    assert "AMD icin son haberler:" in answer
    assert news.queries == ["AMD"]


def test_large_market_move_appends_news() -> None:
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
    answer = agent.reply("nasdaq ne kadar", chat_id="chat-1")
    assert "Nasdaq Composite su an yaklasik 24.500,00 USD seviyesinde." in answer
    assert "Hareket belirgin oldugu icin son haberlerden bazilari:" in answer
    assert news.queries == ["Nasdaq"]


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
