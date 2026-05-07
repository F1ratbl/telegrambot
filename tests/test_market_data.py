from src.tools.market_data import calculate_change, normalize_symbol, MarketDataClient, MarketQuote
from src.bot.telegram import _sanitize_telegram_text
from src.bot.memory import InMemoryConversationMemory
from src.ai.agent import EconomyAgent
from src.config import Settings


class _DummyTool:
    pass


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
