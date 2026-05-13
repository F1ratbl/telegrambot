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


class PriceChartTool:
    yahoo_chart_urls = (
        "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
    )
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
        return ChartRequest(symbol=symbol, period=self._extract_period(lowered))

    def create_price_chart(self, request: ChartRequest) -> tuple[bytes, str]:
        caption = f"{DISPLAY_NAMES.get(normalize_symbol(request.symbol), request.symbol.upper())} {PERIOD_ALIASES[request.period][2]} fiyat grafigi"
        try:
            points = self._fetch_history(request.symbol, request.period)
        except Exception as exc:
            logger.warning("Price chart data unavailable for %s: %s", request.symbol, exc)
            return self._render_unavailable_chart(request.symbol, request.period), f"{caption} - veri gecici olarak alinamadi"
        if len(points) < 2:
            return self._render_unavailable_chart(request.symbol, request.period), f"{caption} - veri gecici olarak alinamadi"
        image = self._render_chart(request.symbol, request.period, points)
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
        if any(marker in lowered for marker in ["7 gün", "7 gun", "1 hafta", "haftalık", "haftalik"]):
            return "7d"
        if any(marker in lowered for marker in ["3 ay", "3 aylık", "3 aylik"]):
            return "3mo"
        if any(marker in lowered for marker in ["6 ay", "6 aylık", "6 aylik"]):
            return "6mo"
        if any(marker in lowered for marker in ["1 yıl", "1 yil", "yıllık", "yillik", "12 ay"]):
            return "1y"
        return "1mo"

    def _fetch_history(self, requested_symbol: str, period: str) -> list[tuple[datetime, float]]:
        errors: list[str] = []
        try:
            return self._fetch_yahoo_history(requested_symbol, period)
        except Exception as exc:
            errors.append(f"Yahoo: {exc}")
            logger.warning("Yahoo history failed for %s: %s", requested_symbol, exc)

        try:
            return self._fetch_stooq_history(requested_symbol, period)
        except Exception as exc:
            errors.append(f"Stooq: {exc}")
            logger.warning("Stooq history failed for %s: %s", requested_symbol, exc)

        raise RuntimeError("; ".join(errors) or "Grafik verisi alinamadi.")

    def _fetch_yahoo_history(self, requested_symbol: str, period: str) -> list[tuple[datetime, float]]:
        import requests

        symbol = normalize_symbol(requested_symbol)
        range_value, interval, _ = PERIOD_ALIASES[period]
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
                    if points:
                        return points
                    raise RuntimeError("Yahoo chart response did not include close prices.")
                except Exception as exc:
                    last_exception = exc
                    continue
        raise last_exception or RuntimeError("Yahoo chart request failed.")

    def _parse_yahoo_points(self, data: dict[str, Any]) -> list[tuple[datetime, float]]:
        result = (((data.get("chart") or {}).get("result") or [None])[0]) or {}
        timestamps = result.get("timestamp") or []
        close_values = ((((result.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or [])
        points: list[tuple[datetime, float]] = []
        for timestamp, close in zip(timestamps, close_values, strict=False):
            if close is None:
                continue
            points.append((datetime.fromtimestamp(timestamp), float(close)))
        return points

    def _fetch_stooq_history(self, requested_symbol: str, period: str) -> list[tuple[datetime, float]]:
        import requests

        stooq_symbols = _stooq_symbols(requested_symbol)
        if not stooq_symbols:
            raise RuntimeError("Stooq fallback does not support this symbol.")

        start, end = _stooq_date_range(period)
        errors: list[str] = []
        for stooq_symbol in stooq_symbols:
            response = requests.get(
                self.stooq_daily_url,
                params={"s": stooq_symbol, "i": "d", "d1": start, "d2": end},
                headers=self.headers,
                timeout=self.settings.request_timeout_seconds,
            )
            try:
                response.raise_for_status()
                points = self._parse_stooq_points(response.text, period)
                if points:
                    return points
                errors.append(f"{stooq_symbol}: empty")
            except Exception as exc:
                errors.append(f"{stooq_symbol}: {exc}")
        raise RuntimeError("; ".join(errors) or "Stooq response did not include close prices.")

    def _parse_stooq_points(self, csv_text: str, period: str) -> list[tuple[datetime, float]]:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=_period_days(period))
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

    def _render_chart(self, requested_symbol: str, period: str, points: list[tuple[datetime, float]]) -> bytes:
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
        ax.set_title(f"{name} - {PERIOD_ALIASES[period][2]}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Fiyat")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        fig.autofmt_xdate()
        fig.tight_layout()

        output = BytesIO()
        fig.savefig(output, format="png", bbox_inches="tight")
        plt.close(fig)
        return output.getvalue()

    def _render_unavailable_chart(self, requested_symbol: str, period: str) -> bytes:
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
            "Yahoo rate limit verebilir; yedek kaynak da boş döndü. Birkaç dakika sonra tekrar deneyebilirsiniz.",
            ha="center",
            va="center",
            fontsize=10,
            color="#64748b",
            wrap=True,
        )
        ax.text(
            0.5,
            0.26,
            f"İstenen dönem: {PERIOD_ALIASES[period][2]}",
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


def _period_days(period: str) -> int:
    return {
        "7d": 14,
        "1mo": 45,
        "3mo": 110,
        "6mo": 210,
        "1y": 400,
    }.get(period, 45)


def _stooq_date_range(period: str) -> tuple[str, str]:
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=_period_days(period))
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _stooq_symbols(requested_symbol: str) -> list[str]:
    normalized = normalize_symbol(requested_symbol)
    stooq_symbols = {
        "GC=F": ["xauusd", "gc.c"],
        "SI=F": ["xagusd", "si.c"],
        "BTC-USD": ["btcusd"],
        "ETH-USD": ["ethusd"],
        "USDTRY=X": ["usdtry"],
        "EURTRY=X": ["eurtry"],
        "^GSPC": ["^spx"],
        "^IXIC": ["^ndq"],
        "^DJI": ["^dji"],
        "^GDAXI": ["dax"],
        "AMD": ["amd.us"],
        "NVDA": ["nvda.us"],
        "AAPL": ["aapl.us"],
        "TSLA": ["tsla.us"],
        "MSFT": ["msft.us"],
    }
    return stooq_symbols.get(normalized, [])
