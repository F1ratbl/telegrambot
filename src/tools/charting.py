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
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/json,text/csv,*/*",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

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

    def create_price_chart(self, request: ChartRequest) -> tuple[bytes, str]:
        period_label = _period_label(request.period, request.custom_days, request.interval_hours)
        caption = f"{DISPLAY_NAMES.get(normalize_symbol(request.symbol), request.symbol.upper())} {period_label} fiyat grafigi"
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
            if custom_days <= 7:
                return "7d"
            if custom_days <= 45:
                return "1mo"
            if custom_days <= 110:
                return "3mo"
            if custom_days <= 210:
                return "6mo"
            return "1y"
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
            r"\bson\s+(\d{1,3})\s*g[uü]n(?:[üu]n|luk|l[uü]k)?\b",
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


def _period_days(period: str) -> int:
    return {
        "7d": 14,
        "1mo": 45,
        "3mo": 110,
        "6mo": 210,
        "1y": 400,
    }.get(period, 45)


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
