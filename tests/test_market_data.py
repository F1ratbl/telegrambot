from src.tools.market_data import calculate_change, normalize_symbol
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
