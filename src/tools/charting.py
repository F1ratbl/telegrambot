from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
import logging
import re
import time
from typing import Any

from src.config import Settings
from src.tools.market_data import DISPLAY_NAMES, normalize_symbol


logger = logging.getLogger(__name__)


PERIOD_ALIASES = {
    "7d": ("7d", "1h", "son 7 gun"),
    "1mo": ("1mo", "1d", "son 1 ay"),
    "3mo": ("3mo", "1d", "son 3 ay"),
    "6mo": ("6mo", "1d", "son 6 ay"),
    "1y": ("1y", "1d", "son 1 yil"),
}


@dataclass(frozen=True)
class ChartRequest:
    symbol: str
    period: str
    custom_days: int | None = None
    interval_hours: int | None = None


@dataclass(frozen=True)
class OHLCPoint:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class PriceChartTool:
    yahoo_chart_urls = (
        "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
    )
    yahoo_spark_urls = (
        "https://query1.finance.yahoo.com/v8/finance/spark",
        "https://query2.finance.yahoo.com/v8/finance/spark",
    )
    nasdaq_historical_url = "https://api.nasdaq.com/api/quote/{symbol}/historical"
    stooq_daily_url = "https://stooq.com/q/d/l/"
    twelve_data_time_series_url = "https://api.twelvedata.com/time_series"
    finnhub_stock_candle_url = "https://finnhub.io/api/v1/stock/candle"
    alpha_vantage_query_url = "https://www.alphavantage.co/query"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/json,text/csv,*/*",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ohlc_cache: dict[str, tuple[float, list[OHLCPoint]]] = {}

    def parse_request(self, text: str) -> ChartRequest | None:
        lowered = text.lower()
        chart_markers = ["grafik", "grafiği", "grafigi", "chart", "çiz", "ciz"]
        if not any(marker in lowered for marker in chart_markers):
            return None

        symbol = self._extract_symbol(text)
        if not symbol:
            return None
        return ChartRequest(
            symbol=symbol,
            period=self._extract_period(lowered),
            custom_days=self._extract_custom_days(lowered),
            interval_hours=self._extract_interval_hours(lowered),
        )

    def request_from_interpretation(self, payload: dict[str, Any] | None) -> ChartRequest | None:
        if not payload or not payload.get("is_chart_request"):
            return None

        raw_symbol = payload.get("symbol")
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            return None

        custom_days = _coerce_int(payload.get("range_days"), minimum=1, maximum=365)
        interval_hours = _coerce_int(payload.get("interval_hours"), minimum=1, maximum=24)
        period = payload.get("period")
        if not isinstance(period, str) or period not in PERIOD_ALIASES:
            period = _period_for_days(custom_days)

        return ChartRequest(
            symbol=raw_symbol.strip().upper(),
            period=period,
            custom_days=custom_days,
            interval_hours=interval_hours,
        )

    def create_price_chart(self, request: ChartRequest) -> tuple[bytes, str]:
        period_label = _period_label(request.period, request.custom_days, request.interval_hours)
        caption = f"{DISPLAY_NAMES.get(normalize_symbol(request.symbol), request.symbol.upper())} {period_label} fiyat grafigi"
        try:
            candles = self._fetch_ohlc_history(
                request.symbol,
                request.period,
                custom_days=request.custom_days,
                interval_hours=request.interval_hours,
            )
            if len(candles) >= 2:
                return self._render_candlestick_chart(
                    request.symbol,
                    request.period,
                    candles,
                    request.custom_days,
                    request.interval_hours,
                ), caption
        except Exception as exc:
            logger.info("OHLC chart data unavailable for %s, falling back to close line chart: %s", request.symbol, exc)

        try:
            points = self._fetch_history(
                request.symbol,
                request.period,
                custom_days=request.custom_days,
                interval_hours=request.interval_hours,
            )
        except Exception as exc:
            logger.warning("Price chart data unavailable for %s: %s", request.symbol, exc)
            return self._render_unavailable_chart(request.symbol, request.period, request.custom_days, request.interval_hours), f"{caption} - veri gecici olarak alinamadi"
        if len(points) < 2:
            return self._render_unavailable_chart(request.symbol, request.period, request.custom_days, request.interval_hours), f"{caption} - veri gecici olarak alinamadi"
        image = self._render_chart(request.symbol, request.period, points, request.custom_days, request.interval_hours)
        return image, caption

    def _fetch_ohlc_history(
        self,
        requested_symbol: str,
        period: str,
        custom_days: int | None = None,
        interval_hours: int | None = None,
    ) -> list[OHLCPoint]:
        cache_key = _ohlc_cache_key(requested_symbol, period, custom_days, interval_hours)
        cached = self._ohlc_cache.get(cache_key)
        if cached and time.monotonic() - cached[0] <= max(0, self.settings.chart_cache_ttl_seconds):
            return list(cached[1])

        errors: list[str] = []
        providers = [
            ("Twelve Data", self.settings.twelve_data_api_key, self._fetch_twelve_data_ohlc),
            ("Finnhub", self.settings.finnhub_api_key, self._fetch_finnhub_ohlc),
            ("Alpha Vantage", self.settings.alpha_vantage_api_key, self._fetch_alpha_vantage_ohlc),
        ]
        for provider_name, api_key, fetcher in providers:
            if not api_key:
                continue
            try:
                candles = fetcher(requested_symbol, period, custom_days, interval_hours)
                candles = _post_process_ohlc(candles, period, custom_days, interval_hours)
                if candles:
                    self._ohlc_cache[cache_key] = (time.monotonic(), candles)
                    return list(candles)
                errors.append(f"{provider_name}: empty")
            except Exception as exc:
                errors.append(f"{provider_name}: {exc}")
                logger.info("%s OHLC history failed for %s: %s", provider_name, requested_symbol, exc)
        raise RuntimeError("; ".join(errors) or "No OHLC provider API key configured.")

    def _fetch_twelve_data_ohlc(
        self,
        requested_symbol: str,
        period: str,
        custom_days: int | None = None,
        interval_hours: int | None = None,
    ) -> list[OHLCPoint]:
        import requests

        symbol = _twelve_data_symbol(requested_symbol)
        if symbol is None:
            raise RuntimeError("Twelve Data fallback does not support this symbol.")
        response = requests.get(
            self.twelve_data_time_series_url,
            params={
                "symbol": symbol,
                "interval": _twelve_data_interval(period, custom_days, interval_hours),
                "outputsize": _ohlc_output_size(period, custom_days, interval_hours),
                "order": "ASC",
                "apikey": self.settings.twelve_data_api_key,
            },
            headers=self.headers,
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "error":
            raise RuntimeError(str(data.get("message") or "Twelve Data returned an error."))
        return _parse_twelve_data_ohlc(data)

    def _fetch_finnhub_ohlc(
        self,
        requested_symbol: str,
        period: str,
        custom_days: int | None = None,
        interval_hours: int | None = None,
    ) -> list[OHLCPoint]:
        import requests

        symbol = _finnhub_symbol(requested_symbol)
        if symbol is None:
            raise RuntimeError("Finnhub fallback does not support this symbol.")
        start, end = _unix_date_range(period, custom_days)
        response = requests.get(
            self.finnhub_stock_candle_url,
            params={
                "symbol": symbol,
                "resolution": _finnhub_resolution(custom_days, interval_hours),
                "from": start,
                "to": end,
                "token": self.settings.finnhub_api_key,
            },
            headers=self.headers,
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("s") != "ok":
            raise RuntimeError(str(data.get("s") or "Finnhub returned no candles."))
        return _parse_finnhub_ohlc(data)

    def _fetch_alpha_vantage_ohlc(
        self,
        requested_symbol: str,
        period: str,
        custom_days: int | None = None,
        interval_hours: int | None = None,
    ) -> list[OHLCPoint]:
        import requests

        symbol = _alpha_vantage_symbol(requested_symbol)
        if symbol is None:
            raise RuntimeError("Alpha Vantage fallback does not support this symbol.")

        params: dict[str, Any]
        if interval_hours:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": "60min",
                "outputsize": "full",
                "apikey": self.settings.alpha_vantage_api_key,
            }
        else:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self.settings.alpha_vantage_api_key,
            }
        response = requests.get(
            self.alpha_vantage_query_url,
            params=params,
            headers=self.headers,
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if "Error Message" in data or "Note" in data or "Information" in data:
            raise RuntimeError(str(data.get("Error Message") or data.get("Note") or data.get("Information")))
        return _parse_alpha_vantage_ohlc(data)

    def _extract_symbol(self, text: str) -> str | None:
        lowered = text.lower()
        candidates = [
            ("ALTIN", ["altın", "altin", "gold", "ons"]),
            ("GUMUS", ["gümüş", "gumus", "silver"]),
            ("BRENT", ["brent", "petrol"]),
            ("BTCUSD", ["bitcoin", "btc"]),
            ("ETHUSD", ["ethereum", "eth"]),
            ("USDTRY", ["dolar tl", "dolar/tl", "usdtry", "usd/try", "dolar"]),
            ("EURTRY", ["euro tl", "eurtry", "eur/tl", "euro"]),
            ("BIST100", ["bist100", "bist 100", "xu100", "bist"]),
            ("SP500", ["s&p 500", "s&p", "sp500", "sp 500"]),
            ("NASDAQ", ["nasdaq"]),
            ("DAX", ["dax"]),
            ("AMD", ["amd"]),
            ("NVDA", ["nvidia", "nvda"]),
            ("AAPL", ["apple", "aapl"]),
            ("TSLA", ["tesla", "tsla"]),
            ("MSFT", ["microsoft", "msft"]),
        ]
        for symbol, aliases in candidates:
            if any(alias in lowered for alias in aliases):
                return symbol

        ticker_match = re.search(r"\b[A-Z]{2,6}\b", text)
        if ticker_match:
            return ticker_match.group(0)
        return None

    def _extract_period(self, lowered: str) -> str:
        custom_days = self._extract_custom_days(lowered)
        if custom_days is not None:
            return _period_for_days(custom_days)
        if any(marker in lowered for marker in ["7 gün", "7 gun", "1 hafta", "haftalık", "haftalik"]):
            return "7d"
        if any(marker in lowered for marker in ["3 ay", "3 aylık", "3 aylik"]):
            return "3mo"
        if any(marker in lowered for marker in ["6 ay", "6 aylık", "6 aylik"]):
            return "6mo"
        if any(marker in lowered for marker in ["1 yıl", "1 yil", "yıllık", "yillik", "12 ay"]):
            return "1y"
        return "1mo"

    def _extract_custom_days(self, lowered: str) -> int | None:
        patterns = [
            r"\bson\s+(\d{1,3})\s*g[uü]n(?:[üu]n|deki|daki|lük|luk|l[uü]k)?\b",
            r"\b(\d{1,3})\s*g[uü]nl[uü]k\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            days = int(match.group(1))
            return min(max(days, 1), 365)
        return None

    def _extract_interval_hours(self, lowered: str) -> int | None:
        match = re.search(r"\b(\d{1,2})\s*saat(?:lik|lık|lik|lik)?\b", lowered)
        if not match:
            return None
        hours = int(match.group(1))
        if hours <= 0:
            return None
        return min(hours, 24)

    def _fetch_history(
        self,
        requested_symbol: str,
        period: str,
        custom_days: int | None = None,
        interval_hours: int | None = None,
    ) -> list[tuple[datetime, float]]:
        errors: list[str] = []
        try:
            return self._fetch_yahoo_history(requested_symbol, period, custom_days, interval_hours)
        except Exception as exc:
            errors.append(f"Yahoo: {exc}")
            logger.info("Yahoo history failed for %s, trying next provider: %s", requested_symbol, exc)

        try:
            return self._fetch_yahoo_spark_history(requested_symbol, period, custom_days, interval_hours)
        except Exception as exc:
            errors.append(f"Yahoo spark: {exc}")
            logger.info("Yahoo spark history failed for %s, trying next provider: %s", requested_symbol, exc)

        try:
            return self._fetch_nasdaq_history(requested_symbol, period, custom_days)
        except Exception as exc:
            errors.append(f"Nasdaq: {exc}")
            logger.info("Nasdaq history failed for %s, trying next provider: %s", requested_symbol, exc)

        try:
            return self._fetch_stooq_history(requested_symbol, period, custom_days)
        except Exception as exc:
            errors.append(f"Stooq: {exc}")
            logger.info("Stooq history failed for %s, trying next provider: %s", requested_symbol, exc)

        raise RuntimeError("; ".join(errors) or "Grafik verisi alinamadi.")

    def _fetch_yahoo_history(
        self,
        requested_symbol: str,
        period: str,
        custom_days: int | None = None,
        interval_hours: int | None = None,
    ) -> list[tuple[datetime, float]]:
        import requests

        symbol = normalize_symbol(requested_symbol)
        range_value, interval = _yahoo_range_interval(period, custom_days, interval_hours)
        response = None
        last_exception: Exception | None = None
        for url_template in self.yahoo_chart_urls:
            for attempt in range(2):
                try:
                    response = requests.get(
                        url_template.format(symbol=symbol),
                        params={"range": range_value, "interval": interval},
                        headers=self.headers,
                        timeout=self.settings.request_timeout_seconds,
                    )
                    if response.status_code == 429 and attempt == 0:
                        retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                        time.sleep(retry_after)
                        continue
                    response.raise_for_status()
                    data = response.json()
                    points = self._parse_yahoo_points(data)
                    points = _post_process_points(points, custom_days, interval_hours)
                    if points:
                        return points
                    raise RuntimeError("Yahoo chart response did not include close prices.")
                except Exception as exc:
                    last_exception = exc
                    continue
        raise last_exception or RuntimeError("Yahoo chart request failed.")

    def _fetch_yahoo_spark_history(
        self,
        requested_symbol: str,
        period: str,
        custom_days: int | None = None,
        interval_hours: int | None = None,
    ) -> list[tuple[datetime, float]]:
        import requests

        symbol = normalize_symbol(requested_symbol)
        range_value, interval = _yahoo_range_interval(period, custom_days, interval_hours)
        errors: list[str] = []
        for spark_symbol in _yahoo_spark_symbols(symbol):
            for url in self.yahoo_spark_urls:
                response = requests.get(
                    url,
                    params={"symbols": spark_symbol, "range": range_value, "interval": interval},
                    headers=self.headers,
                    timeout=self.settings.request_timeout_seconds,
                )
                try:
                    response.raise_for_status()
                    points = self._parse_yahoo_spark_points(response.json(), spark_symbol)
                    points = _post_process_points(points, custom_days, interval_hours)
                    if points:
                        return points
                    errors.append(f"{spark_symbol}: empty")
                except Exception as exc:
                    errors.append(f"{spark_symbol}: {exc}")
        raise RuntimeError("; ".join(errors) or "Yahoo spark response did not include close prices.")

    def _fetch_nasdaq_history(
        self,
        requested_symbol: str,
        period: str,
        custom_days: int | None = None,
    ) -> list[tuple[datetime, float]]:
        import requests

        nasdaq_symbol = _nasdaq_symbol(requested_symbol)
        if nasdaq_symbol is None:
            raise RuntimeError("Nasdaq fallback does not support this symbol.")

        symbol, assetclass = nasdaq_symbol
        start, end = _nasdaq_date_range(period, custom_days)
        response = requests.get(
            self.nasdaq_historical_url.format(symbol=symbol),
            params={
                "assetclass": assetclass,
                "fromdate": start,
                "todate": end,
                "limit": 9999,
            },
            headers={
                **self.headers,
                "Origin": "https://www.nasdaq.com",
                "Referer": "https://www.nasdaq.com/",
            },
            timeout=max(self.settings.request_timeout_seconds, 15),
        )
        response.raise_for_status()
        points = self._parse_nasdaq_points(response.json(), period, custom_days)
        if points:
            return points
        raise RuntimeError("Nasdaq response did not include close prices.")

    def _parse_yahoo_points(self, data: dict[str, Any]) -> list[tuple[datetime, float]]:
        result = (((data.get("chart") or {}).get("result") or [None])[0]) or {}
        timestamps = result.get("timestamp") or []
        close_values = ((((result.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or [])
        return _points_from_timestamps_and_closes(timestamps, close_values)

    def _parse_yahoo_spark_points(self, data: dict[str, Any], symbol: str) -> list[tuple[datetime, float]]:
        result = data.get(symbol) or {}
        if not result and data:
            first_value = next(iter(data.values()), {})
            if isinstance(first_value, dict):
                result = first_value
        timestamps = result.get("timestamp") or []
        close_values = result.get("close") or []
        return _points_from_timestamps_and_closes(timestamps, close_values)

    def _parse_nasdaq_points(self, data: dict[str, Any], period: str, custom_days: int | None = None) -> list[tuple[datetime, float]]:
        cutoff = _cutoff_datetime(period, custom_days)
        rows = (((data.get("data") or {}).get("tradesTable") or {}).get("rows") or [])
        points: list[tuple[datetime, float]] = []
        for row in rows:
            date_value = row.get("date")
            close_value = row.get("close")
            if not date_value or not close_value:
                continue
            try:
                date = datetime.strptime(str(date_value), "%m/%d/%Y")
                close = _parse_price(str(close_value))
            except ValueError:
                continue
            if date >= cutoff:
                points.append((date, close))
        return sorted(points, key=lambda item: item[0])

    def _parse_stooq_points(self, csv_text: str, period: str, custom_days: int | None = None) -> list[tuple[datetime, float]]:
        cutoff = _cutoff_datetime(period, custom_days)
        points: list[tuple[datetime, float]] = []
        for row in csv.DictReader(StringIO(csv_text)):
            date_value = row.get("Date")
            close_value = row.get("Close")
            if not date_value or not close_value or close_value.upper() == "N/D":
                continue
            try:
                date = datetime.strptime(date_value, "%Y-%m-%d")
                close = float(close_value)
            except ValueError:
                continue
            if date >= cutoff:
                points.append((date, close))
        return points

    def _fetch_stooq_history(self, requested_symbol: str, period: str, custom_days: int | None = None) -> list[tuple[datetime, float]]:
        import requests

        stooq_symbols = _stooq_symbols(
            requested_symbol,
            include_api_key_symbols=bool(self.settings.stooq_api_key),
        )
        if not stooq_symbols:
            raise RuntimeError("Stooq fallback does not support this symbol.")

        start, end = _stooq_date_range(period, custom_days)
        errors: list[str] = []
        for stooq_symbol in stooq_symbols:
            params = {"s": stooq_symbol, "i": "d", "d1": start, "d2": end}
            if self.settings.stooq_api_key:
                params["apikey"] = self.settings.stooq_api_key
            response = requests.get(
                self.stooq_daily_url,
                params=params,
                headers=self.headers,
                timeout=self.settings.request_timeout_seconds,
            )
            try:
                response.raise_for_status()
                if _stooq_requires_api_key(response.text):
                    errors.append(f"{stooq_symbol}: apikey required")
                    continue
                points = self._parse_stooq_points(response.text, period, custom_days)
                if points:
                    return points
                errors.append(f"{stooq_symbol}: empty")
            except Exception as exc:
                errors.append(f"{stooq_symbol}: {exc}")
        raise RuntimeError("; ".join(errors) or "Stooq response did not include close prices.")

    def _render_chart(
        self,
        requested_symbol: str,
        period: str,
        points: list[tuple[datetime, float]],
        custom_days: int | None = None,
        interval_hours: int | None = None,
    ) -> bytes:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        symbol = normalize_symbol(requested_symbol)
        dates = [point[0] for point in points]
        prices = [point[1] for point in points]
        name = DISPLAY_NAMES.get(symbol, requested_symbol.upper())

        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        fig.patch.set_facecolor("#f7f8fa")
        ax.set_facecolor("#ffffff")
        ax.plot(dates, prices, color="#2563eb", linewidth=2.4)
        ax.fill_between(dates, prices, min(prices), color="#dbeafe", alpha=0.55)
        ax.set_title(f"{name} - {_period_label(period, custom_days, interval_hours)}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Fiyat")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        date_format = "%d.%m %H:%M" if interval_hours else "%d.%m"
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        fig.autofmt_xdate()
        fig.tight_layout()

        output = BytesIO()
        fig.savefig(output, format="png", bbox_inches="tight")
        plt.close(fig)
        return output.getvalue()

    def _render_candlestick_chart(
        self,
        requested_symbol: str,
        period: str,
        candles: list[OHLCPoint],
        custom_days: int | None = None,
        interval_hours: int | None = None,
    ) -> bytes:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        symbol = normalize_symbol(requested_symbol)
        name = DISPLAY_NAMES.get(symbol, requested_symbol.upper())
        closes = [candle.close for candle in candles]

        fig, (ax, volume_ax) = plt.subplots(
            2,
            1,
            figsize=(9, 5.4),
            dpi=160,
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
        )
        fig.patch.set_facecolor("#f7f8fa")
        ax.set_facecolor("#ffffff")
        volume_ax.set_facecolor("#ffffff")

        x_values = mdates.date2num([candle.time for candle in candles])
        candle_width = _candle_width(x_values)
        for candle, x_value in zip(candles, x_values, strict=False):
            rising = candle.close >= candle.open
            color = "#16a34a" if rising else "#dc2626"
            lower = min(candle.open, candle.close)
            height = max(abs(candle.close - candle.open), max(closes) * 0.0008)
            ax.vlines(x_value, candle.low, candle.high, color=color, linewidth=1.1, alpha=0.95)
            ax.add_patch(
                Rectangle(
                    (x_value - candle_width / 2, lower),
                    candle_width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.85,
                )
            )
            if candle.volume is not None:
                volume_ax.bar(x_value, candle.volume, width=candle_width, color=color, alpha=0.25)

        ax.set_title(f"{name} - {_period_label(period, custom_days, interval_hours)}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Fiyat")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        volume_ax.set_ylabel("Hacim", fontsize=9)
        volume_ax.grid(True, color="#eef2f7", linewidth=0.7)
        volume_ax.spines["top"].set_visible(False)
        volume_ax.spines["right"].set_visible(False)
        volume_ax.tick_params(axis="y", labelsize=8)
        date_format = "%d.%m %H:%M" if interval_hours else "%d.%m"
        volume_ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        fig.autofmt_xdate()
        fig.tight_layout()

        output = BytesIO()
        fig.savefig(output, format="png", bbox_inches="tight")
        plt.close(fig)
        return output.getvalue()

    def _render_unavailable_chart(
        self,
        requested_symbol: str,
        period: str,
        custom_days: int | None = None,
        interval_hours: int | None = None,
    ) -> bytes:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        symbol = normalize_symbol(requested_symbol)
        name = DISPLAY_NAMES.get(symbol, requested_symbol.upper())
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        ax.axis("off")
        ax.text(
            0.5,
            0.62,
            f"{name} grafiği",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="#111827",
        )
        ax.text(
            0.5,
            0.48,
            "Geçmiş fiyat verisi şu an veri sağlayıcılardan alınamadı.",
            ha="center",
            va="center",
            fontsize=12,
            color="#334155",
            wrap=True,
        )
        ax.text(
            0.5,
            0.38,
            "Yahoo rate limit verebilir; yedek kaynak icin STOOQ_API_KEY gerekebilir. Birkaç dakika sonra tekrar deneyebilirsiniz.",
            ha="center",
            va="center",
            fontsize=10,
            color="#64748b",
            wrap=True,
        )
        ax.text(
            0.5,
            0.26,
            f"İstenen dönem: {_period_label(period, custom_days, interval_hours)}",
            ha="center",
            va="center",
            fontsize=10,
            color="#64748b",
        )

        output = BytesIO()
        fig.savefig(output, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return output.getvalue()


def _retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0.75
    try:
        return min(max(float(value), 0.25), 3.0)
    except ValueError:
        return 0.75


def _points_from_timestamps_and_closes(timestamps: list[Any], close_values: list[Any]) -> list[tuple[datetime, float]]:
    points: list[tuple[datetime, float]] = []
    for timestamp, close in zip(timestamps, close_values, strict=False):
        if timestamp is None or close is None:
            continue
        try:
            points.append((datetime.fromtimestamp(float(timestamp)), float(close)))
        except (TypeError, ValueError, OSError):
            continue
    return points


def _post_process_points(
    points: list[tuple[datetime, float]],
    custom_days: int | None,
    interval_hours: int | None,
) -> list[tuple[datetime, float]]:
    if custom_days is not None:
        cutoff = datetime.now() - timedelta(days=custom_days)
        points = [point for point in points if point[0] >= cutoff]
    if interval_hours and interval_hours > 1:
        points = _resample_points_by_hours(points, interval_hours)
    return points


def _post_process_ohlc(
    candles: list[OHLCPoint],
    period: str,
    custom_days: int | None,
    interval_hours: int | None,
) -> list[OHLCPoint]:
    cutoff = _cutoff_datetime(period, custom_days)
    filtered = [candle for candle in candles if candle.time >= cutoff]
    filtered = sorted(filtered, key=lambda candle: candle.time)
    if interval_hours and interval_hours > 1:
        filtered = _resample_ohlc_by_hours(filtered, interval_hours)
    return filtered


def _resample_points_by_hours(points: list[tuple[datetime, float]], interval_hours: int) -> list[tuple[datetime, float]]:
    if not points:
        return []
    buckets: dict[datetime, tuple[datetime, float]] = {}
    for point_time, price in sorted(points, key=lambda item: item[0]):
        bucket_time = point_time.replace(
            minute=0,
            second=0,
            microsecond=0,
            hour=(point_time.hour // interval_hours) * interval_hours,
        )
        buckets[bucket_time] = (bucket_time, price)
    return [buckets[key] for key in sorted(buckets)]


def _resample_ohlc_by_hours(candles: list[OHLCPoint], interval_hours: int) -> list[OHLCPoint]:
    if not candles:
        return []
    buckets: dict[datetime, list[OHLCPoint]] = {}
    for candle in sorted(candles, key=lambda item: item.time):
        bucket_time = candle.time.replace(
            minute=0,
            second=0,
            microsecond=0,
            hour=(candle.time.hour // interval_hours) * interval_hours,
        )
        buckets.setdefault(bucket_time, []).append(candle)

    resampled = []
    for bucket_time in sorted(buckets):
        bucket = buckets[bucket_time]
        volume_values = [candle.volume for candle in bucket if candle.volume is not None]
        resampled.append(
            OHLCPoint(
                time=bucket_time,
                open=bucket[0].open,
                high=max(candle.high for candle in bucket),
                low=min(candle.low for candle in bucket),
                close=bucket[-1].close,
                volume=sum(volume_values) if volume_values else None,
            )
        )
    return resampled


def _yahoo_spark_symbols(normalized_symbol: str) -> list[str]:
    proxy_symbols = {
        "^GSPC": ["SPY"],
        "^IXIC": ["QQQ"],
        "^DJI": ["DIA"],
        "^RUT": ["IWM"],
    }
    return [normalized_symbol, *proxy_symbols.get(normalized_symbol, [])]


def _period_label(period: str, custom_days: int | None = None, interval_hours: int | None = None) -> str:
    base = f"son {custom_days} gun" if custom_days is not None else PERIOD_ALIASES[period][2]
    if interval_hours:
        return f"{interval_hours} saatlik, {base}"
    return base


def _yahoo_range_interval(
    period: str,
    custom_days: int | None = None,
    interval_hours: int | None = None,
) -> tuple[str, str]:
    if custom_days is None and not interval_hours:
        range_value, interval, _ = PERIOD_ALIASES[period]
        return range_value, interval

    days = custom_days or _period_days(period)
    if interval_hours:
        if days <= 5:
            return "5d", "1h"
        if days <= 7:
            return "7d", "1h"
        if days <= 60:
            return "60d", "1h"
        return "1y", "1d"

    if days <= 5:
        return "5d", "1h"
    if days <= 7:
        return "7d", "1h"
    if days <= 45:
        return "1mo", "1d"
    if days <= 110:
        return "3mo", "1d"
    if days <= 210:
        return "6mo", "1d"
    return "1y", "1d"


def _ohlc_cache_key(
    requested_symbol: str,
    period: str,
    custom_days: int | None,
    interval_hours: int | None,
) -> str:
    return f"{normalize_symbol(requested_symbol)}:{period}:{custom_days or ''}:{interval_hours or ''}"


def _ohlc_output_size(period: str, custom_days: int | None = None, interval_hours: int | None = None) -> int:
    days = custom_days or _period_days(period)
    if interval_hours:
        return min(max(int((days * 24) / max(interval_hours, 1)) + 24, 50), 5000)
    if days <= 7:
        return 250
    return min(max(days + 20, 50), 5000)


def _twelve_data_symbol(requested_symbol: str) -> str | None:
    normalized = normalize_symbol(requested_symbol)
    symbols = {
        "BTC-USD": "BTC/USD",
        "ETH-USD": "ETH/USD",
        "USDTRY=X": "USD/TRY",
        "EURTRY=X": "EUR/TRY",
        "^GSPC": "SPY",
        "^IXIC": "QQQ",
        "^DJI": "DIA",
        "^RUT": "IWM",
        "XU100.IS": "XU100",
        "GC=F": "XAU/USD",
        "SI=F": "XAG/USD",
        "BZ=F": "BZ",
        "CL=F": "CL",
        "^GDAXI": "DAX",
    }
    if re.fullmatch(r"[A-Z]{1,5}", normalized):
        return normalized
    return symbols.get(normalized)


def _finnhub_symbol(requested_symbol: str) -> str | None:
    normalized = normalize_symbol(requested_symbol)
    symbols = {
        "BTC-USD": "BINANCE:BTCUSDT",
        "ETH-USD": "BINANCE:ETHUSDT",
    }
    if re.fullmatch(r"[A-Z]{1,5}", normalized):
        return normalized
    return symbols.get(normalized)


def _alpha_vantage_symbol(requested_symbol: str) -> str | None:
    normalized = normalize_symbol(requested_symbol)
    if re.fullmatch(r"[A-Z]{1,5}", normalized):
        return normalized
    return None


def _twelve_data_interval(
    period: str,
    custom_days: int | None = None,
    interval_hours: int | None = None,
) -> str:
    if interval_hours:
        return f"{interval_hours}h"
    days = custom_days or _period_days(period)
    return "1h" if days <= 7 else "1day"


def _finnhub_resolution(custom_days: int | None = None, interval_hours: int | None = None) -> str:
    if interval_hours:
        return str(interval_hours * 60)
    if custom_days is not None and custom_days <= 7:
        return "60"
    return "D"


def _unix_date_range(period: str, custom_days: int | None = None) -> tuple[int, int]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=custom_days if custom_days is not None else _period_days(period))
    return int(start.timestamp()), int(end.timestamp())


def _parse_twelve_data_ohlc(data: dict[str, Any]) -> list[OHLCPoint]:
    candles = []
    for row in data.get("values") or []:
        try:
            candles.append(
                OHLCPoint(
                    time=_parse_provider_datetime(str(row["datetime"])),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=_optional_float_value(row.get("volume")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(candles, key=lambda candle: candle.time)


def _parse_finnhub_ohlc(data: dict[str, Any]) -> list[OHLCPoint]:
    timestamps = data.get("t") or []
    opens = data.get("o") or []
    highs = data.get("h") or []
    lows = data.get("l") or []
    closes = data.get("c") or []
    volumes = data.get("v") or []
    candles = []
    for index, timestamp in enumerate(timestamps):
        try:
            candles.append(
                OHLCPoint(
                    time=datetime.fromtimestamp(float(timestamp)),
                    open=float(opens[index]),
                    high=float(highs[index]),
                    low=float(lows[index]),
                    close=float(closes[index]),
                    volume=_optional_float_value(volumes[index] if index < len(volumes) else None),
                )
            )
        except (IndexError, TypeError, ValueError, OSError):
            continue
    return candles


def _parse_alpha_vantage_ohlc(data: dict[str, Any]) -> list[OHLCPoint]:
    series_key = next((key for key in data if key.startswith("Time Series")), None)
    if not series_key:
        return []
    candles = []
    for timestamp, row in (data.get(series_key) or {}).items():
        try:
            candles.append(
                OHLCPoint(
                    time=_parse_provider_datetime(str(timestamp)),
                    open=float(row["1. open"]),
                    high=float(row["2. high"]),
                    low=float(row["3. low"]),
                    close=float(row["4. close"]),
                    volume=_optional_float_value(row.get("5. volume")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(candles, key=lambda candle: candle.time)


def _parse_provider_datetime(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def _optional_float_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candle_width(x_values: list[float]) -> float:
    if len(x_values) < 2:
        return 0.5
    gaps = [right - left for left, right in zip(x_values, x_values[1:], strict=False) if right > left]
    if not gaps:
        return 0.5
    return min(gaps) * 0.65


def _period_days(period: str) -> int:
    return {
        "7d": 14,
        "1mo": 45,
        "3mo": 110,
        "6mo": 210,
        "1y": 400,
    }.get(period, 45)


def _period_for_days(days: int | None) -> str:
    if days is None:
        return "1mo"
    if days <= 7:
        return "7d"
    if days <= 45:
        return "1mo"
    if days <= 110:
        return "3mo"
    if days <= 210:
        return "6mo"
    return "1y"


def _coerce_int(value: Any, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return min(max(coerced, minimum), maximum)


def _cutoff_datetime(period: str, custom_days: int | None = None) -> datetime:
    days = custom_days if custom_days is not None else _period_days(period)
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)


def _stooq_date_range(period: str, custom_days: int | None = None) -> tuple[str, str]:
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=custom_days if custom_days is not None else _period_days(period))
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _nasdaq_date_range(period: str, custom_days: int | None = None) -> tuple[str, str]:
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=custom_days if custom_days is not None else _period_days(period))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _parse_price(value: str) -> float:
    clean = value.replace("$", "").replace(",", "").strip()
    return float(clean)


def _nasdaq_symbol(requested_symbol: str) -> tuple[str, str] | None:
    normalized = normalize_symbol(requested_symbol)
    symbols = {
        "^GSPC": ("SPY", "etf"),
        "^IXIC": ("QQQ", "etf"),
        "^DJI": ("DIA", "etf"),
        "^RUT": ("IWM", "etf"),
        "AMD": ("AMD", "stocks"),
        "NVDA": ("NVDA", "stocks"),
        "AAPL": ("AAPL", "stocks"),
        "TSLA": ("TSLA", "stocks"),
        "MSFT": ("MSFT", "stocks"),
    }
    return symbols.get(normalized)


def _stooq_symbols(requested_symbol: str, include_api_key_symbols: bool = False) -> list[str]:
    normalized = normalize_symbol(requested_symbol)
    stooq_symbols = {
        "GC=F": ["xauusd", "gc.c"],
        "SI=F": ["xagusd", "si.c"],
        "BZ=F": ["brn.c", "bz.f"],
        "CL=F": ["cl.f"],
        "BTC-USD": ["btcusd"],
        "ETH-USD": ["ethusd"],
        "USDTRY=X": ["usdtry"],
        "EURTRY=X": ["eurtry"],
        "^GSPC": ["spy.us"],
        "^IXIC": ["qqq.us"],
        "^DJI": ["dia.us"],
        "^RUT": ["iwm.us"],
        "^GDAXI": ["dax"],
        "AMD": ["amd.us"],
        "NVDA": ["nvda.us"],
        "AAPL": ["aapl.us"],
        "TSLA": ["tsla.us"],
        "MSFT": ["msft.us"],
    }
    symbols = list(stooq_symbols.get(normalized, []))
    if include_api_key_symbols:
        api_key_symbols = {
            "^GSPC": ["^spx"],
            "^IXIC": ["^ndq"],
            "^DJI": ["^dji"],
        }
        symbols = [*api_key_symbols.get(normalized, []), *symbols]
    return symbols


def _stooq_requires_api_key(text: str) -> bool:
    lowered = text.lower()
    return "get your apikey" in lowered or ("apikey" in lowered and "captcha" in lowered)
