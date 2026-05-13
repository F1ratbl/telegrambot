from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import logging
import re
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
    yahoo_chart_url = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

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
        points = self._fetch_history(request.symbol, request.period)
        if len(points) < 2:
            raise RuntimeError("Grafik cizmek icin yeterli fiyat verisi bulunamadi.")
        image = self._render_chart(request.symbol, request.period, points)
        caption = f"{DISPLAY_NAMES.get(normalize_symbol(request.symbol), request.symbol.upper())} {PERIOD_ALIASES[request.period][2]} fiyat grafigi"
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
        import requests

        symbol = normalize_symbol(requested_symbol)
        range_value, interval, _ = PERIOD_ALIASES[period]
        response = requests.get(
            self.yahoo_chart_url.format(symbol=symbol),
            params={"range": range_value, "interval": interval},
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        result = (((data.get("chart") or {}).get("result") or [None])[0]) or {}
        timestamps = result.get("timestamp") or []
        close_values = ((((result.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or [])
        points: list[tuple[datetime, float]] = []
        for timestamp, close in zip(timestamps, close_values, strict=False):
            if close is None:
                continue
            points.append((datetime.fromtimestamp(timestamp), float(close)))
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
