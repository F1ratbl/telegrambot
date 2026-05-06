from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from src.config import Settings


logger = logging.getLogger(__name__)


SYMBOL_ALIASES = {
    "BIST": "XU100.IS",
    "BIST100": "XU100.IS",
    "XU100": "XU100.IS",
    "SP500": "^GSPC",
    "S&P500": "^GSPC",
    "SANDP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "NASDAQCOMPOSITE": "^IXIC",
    "DOW": "^DJI",
    "DOWJONES": "^DJI",
    "RUSSELL2000": "^RUT",
    "DAX": "^GDAXI",
    "FTSE": "^FTSE",
    "FTSE100": "^FTSE",
    "NIKKEI": "^N225",
    "NIKKEI225": "^N225",
    "HANGSENG": "^HSI",
    "VIX": "^VIX",
    "USDTRY": "USDTRY=X",
    "DOLAR": "USDTRY=X",
    "DOLARTRY": "USDTRY=X",
    "EURTRY": "EURTRY=X",
    "EUROTRY": "EURTRY=X",
    "EURUSD": "EURUSD=X",
    "GOLD": "GC=F",
    "ALTIN": "GC=F",
    "XAUUSD": "GC=F",
    "BRENT": "BZ=F",
    "PETROL": "BZ=F",
    "WTI": "CL=F",
    "BTC": "BTC-USD",
    "BTCUSD": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD",
    "ETHUSD": "ETH-USD",
}


DISPLAY_NAMES = {
    "XU100.IS": "BIST 100",
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "^DJI": "Dow Jones",
    "^RUT": "Russell 2000",
    "^GDAXI": "DAX",
    "^FTSE": "FTSE 100",
    "^N225": "Nikkei 225",
    "^HSI": "Hang Seng",
    "^VIX": "VIX",
    "USDTRY=X": "USD/TRY",
    "EURTRY=X": "EUR/TRY",
    "EURUSD=X": "EUR/USD",
    "GC=F": "Gold Futures",
    "BZ=F": "Brent Oil",
    "CL=F": "WTI Oil",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}


@dataclass(frozen=True)
class MarketQuote:
    requested_symbol: str
    symbol: str
    name: str
    price: float | None
    previous_close: float | None
    change: float | None
    change_percent: float | None
    currency: str | None
    exchange: str | None
    market_time: str | None
    timezone: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MarketDataClient:
    yahoo_chart_url = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_snapshot(self, symbols: Iterable[str] | None = None) -> dict[str, Any]:
        requested = _clean_symbols(symbols) or list(self.settings.market_default_symbols)
        requested = requested[:12]
        quotes: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        for requested_symbol in requested:
            try:
                quotes.append(self._fetch_quote(requested_symbol).to_dict())
            except Exception as exc:
                logger.warning("Could not fetch market quote for %s: %s", requested_symbol, exc)
                errors.append({"symbol": requested_symbol, "message": str(exc)})

        status = "ok" if quotes else "error"
        return {
            "status": status,
            "provider": "Yahoo Finance chart endpoint",
            "note": "Data can be delayed or unavailable depending on exchange and provider coverage.",
            "quotes": quotes,
            "errors": errors,
        }

    def _fetch_quote(self, requested_symbol: str) -> MarketQuote:
        import requests

        symbol = normalize_symbol(requested_symbol)
        url = self.yahoo_chart_url.format(symbol=symbol)
        response = requests.get(
            url,
            params={"interval": "1d", "range": "5d"},
            headers={"User-Agent": "telegram-economy-ai/1.0"},
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        chart = payload.get("chart") or {}
        error = chart.get("error")
        if error:
            raise RuntimeError(error.get("description") or str(error))

        results = chart.get("result") or []
        if not results:
            raise RuntimeError("No market data returned.")

        result = results[0]
        meta = result.get("meta") or {}
        price = _first_number(meta.get("regularMarketPrice"), _last_close(result))
        previous_close = _first_number(meta.get("previousClose"), meta.get("chartPreviousClose"))
        change, change_percent = calculate_change(price, previous_close)
        market_time = _timestamp_to_iso(meta.get("regularMarketTime"))

        return MarketQuote(
            requested_symbol=requested_symbol,
            symbol=symbol,
            name=DISPLAY_NAMES.get(symbol, meta.get("shortName") or symbol),
            price=price,
            previous_close=previous_close,
            change=change,
            change_percent=change_percent,
            currency=meta.get("currency"),
            exchange=meta.get("exchangeName") or meta.get("fullExchangeName"),
            market_time=market_time,
            timezone=meta.get("exchangeTimezoneName"),
        )


def normalize_symbol(symbol: str) -> str:
    clean = symbol.strip()
    if not clean:
        return clean
    key = _alias_key(clean)
    return SYMBOL_ALIASES.get(key, clean.upper())


def calculate_change(
    price: float | int | None,
    previous_close: float | int | None,
) -> tuple[float | None, float | None]:
    if price is None or previous_close in (None, 0):
        return None, None
    change = float(price) - float(previous_close)
    change_percent = (change / float(previous_close)) * 100
    return round(change, 4), round(change_percent, 4)


def _clean_symbols(symbols: Iterable[str] | None) -> list[str]:
    if symbols is None:
        return []
    cleaned = []
    for symbol in symbols:
        if isinstance(symbol, str) and symbol.strip():
            cleaned.append(symbol.strip())
    return cleaned


def _alias_key(value: str) -> str:
    replacements = str.maketrans({"İ": "I", "ı": "I", "Ş": "S", "ş": "S"})
    normalized = value.translate(replacements).upper()
    return re.sub(r"[^A-Z0-9&]", "", normalized)


def _first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _last_close(result: dict[str, Any]) -> float | None:
    indicators = result.get("indicators") or {}
    quotes = indicators.get("quote") or []
    if not quotes:
        return None
    closes = quotes[0].get("close") or []
    for value in reversed(closes):
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _timestamp_to_iso(timestamp: Any) -> str | None:
    if not isinstance(timestamp, (int, float)):
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
