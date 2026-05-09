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
    "SILVER": "SI=F",
    "GUMUS": "SI=F",
    "XAGUSD": "SI=F",
    "BRENT": "BZ=F",
    "PETROL": "BZ=F",
    "WTI": "CL=F",
    "BTC": "BTC-USD",
    "BTCUSD": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD",
    "ETHUSD": "ETH-USD",
    "AMD": "AMD",
    "NVDA": "NVDA",
    "NVIDIA": "NVDA",
    "AAPL": "AAPL",
    "APPLE": "AAPL",
    "TSLA": "TSLA",
    "TESLA": "TSLA",
    "MSFT": "MSFT",
    "MICROSOFT": "MSFT",
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
    "SI=F": "Silver Futures",
    "BZ=F": "Brent Oil",
    "CL=F": "WTI Oil",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "AMD": "AMD",
    "NVDA": "Nvidia",
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
}


USD_PRICED_SYMBOLS = {
    "^GSPC",
    "^IXIC",
    "^DJI",
    "^RUT",
    "^VIX",
    "GC=F",
    "SI=F",
    "BZ=F",
    "CL=F",
    "BTC-USD",
    "ETH-USD",
    "AMD",
    "NVDA",
    "AAPL",
    "TSLA",
    "MSFT",
}


TRY_PRICED_SYMBOLS = {
    "XU100.IS",
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
    yahoo_quote_url = "https://query1.finance.yahoo.com/v7/finance/quote"
    yahoo_chart_url = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_snapshot(self, symbols: Iterable[str] | None = None) -> dict[str, Any]:
        requested = _clean_symbols(symbols) or list(self.settings.market_default_symbols)
        requested = requested[:12]
        expanded = self._expand_requested_symbols(requested)
        quotes: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        raw_quotes: list[MarketQuote] = []

        for requested_symbol in expanded:
            try:
                quote = self._fetch_quote(requested_symbol)
                raw_quotes.append(quote)
                quotes.append(quote.to_dict())
            except Exception as exc:
                logger.warning("Could not fetch market quote for %s: %s", requested_symbol, exc)
                errors.append({"symbol": requested_symbol, "message": str(exc)})

        status = "ok" if quotes else "error"
        return {
            "status": status,
            "provider": "Yahoo Finance quote endpoint with chart fallback",
            "note": (
                "Latest available quote data is fetched from Yahoo Finance. "
                "Some markets can still be delayed depending on provider and exchange coverage."
            ),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "quotes": quotes,
            "derived_metrics": self._build_derived_metrics(raw_quotes, requested),
            "errors": errors,
        }

    def _expand_requested_symbols(self, requested: list[str]) -> list[str]:
        expanded = list(requested)
        normalized = {normalize_symbol(symbol) for symbol in requested}
        if (
            normalized.intersection(USD_PRICED_SYMBOLS | TRY_PRICED_SYMBOLS)
            and "USDTRY=X" not in normalized
        ):
            expanded.append("USDTRY")
        return expanded

    def _build_derived_metrics(
        self,
        quotes: list[MarketQuote],
        requested: list[str],
    ) -> dict[str, Any]:
        derived: dict[str, Any] = {}
        by_symbol = {quote.symbol: quote for quote in quotes}
        normalized_requested = {normalize_symbol(symbol) for symbol in requested}

        gold_quote = by_symbol.get("GC=F")
        usdtry_quote = by_symbol.get("USDTRY=X")
        if gold_quote and gold_quote.price is not None:
            derived["gold_ounce_usd"] = round(gold_quote.price, 4)
            if "GC=F" in normalized_requested or any(
                symbol in normalized_requested for symbol in {"USDTRY=X", "EURTRY=X"}
            ):
                ounce_grams = 31.1035
                derived["gold_gram_usd_estimate"] = round(gold_quote.price / ounce_grams, 4)
                if usdtry_quote and usdtry_quote.price is not None:
                    derived["gold_ounce_try_estimate"] = round(gold_quote.price * usdtry_quote.price, 4)
                    derived["gold_gram_try_estimate"] = round(
                        (gold_quote.price / ounce_grams) * usdtry_quote.price,
                        4,
                    )
                    derived["usdtry"] = round(usdtry_quote.price, 4)

        return derived

    def _fetch_quote(self, requested_symbol: str) -> MarketQuote:
        symbol = normalize_symbol(requested_symbol)
        try:
            quote = self._fetch_quote_from_quote_endpoint(symbol, requested_symbol)
            if quote is not None:
                return quote
        except Exception as exc:
            logger.info(
                "Quote endpoint failed for %s, falling back to chart endpoint: %s",
                symbol,
                exc,
            )
        return self._fetch_quote_from_chart_endpoint(symbol, requested_symbol)

    def _fetch_quote_from_quote_endpoint(
        self,
        symbol: str,
        requested_symbol: str,
    ) -> MarketQuote | None:
        import requests

        response = requests.get(
            self.yahoo_quote_url,
            params={"symbols": symbol},
            headers={"User-Agent": "telegram-economy-ai/1.0"},
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        quote_response = payload.get("quoteResponse") or {}
        results = quote_response.get("result") or []
        if not results:
            return None

        result = results[0]
        price = _first_number(
            result.get("regularMarketPrice"),
            result.get("postMarketPrice"),
            result.get("preMarketPrice"),
            result.get("bid"),
            result.get("ask"),
        )
        previous_close = _first_number(
            result.get("regularMarketPreviousClose"),
            result.get("previousClose"),
        )
        change, change_percent = calculate_change(price, previous_close)
        market_time = _timestamp_to_iso(result.get("regularMarketTime"))

        return MarketQuote(
            requested_symbol=requested_symbol,
            symbol=symbol,
            name=DISPLAY_NAMES.get(symbol, result.get("shortName") or symbol),
            price=price,
            previous_close=previous_close,
            change=change,
            change_percent=change_percent,
            currency=result.get("currency"),
            exchange=result.get("fullExchangeName") or result.get("exchange"),
            market_time=market_time,
            timezone=result.get("exchangeTimezoneName"),
        )

    def _fetch_quote_from_chart_endpoint(
        self,
        symbol: str,
        requested_symbol: str,
    ) -> MarketQuote:
        import requests

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
