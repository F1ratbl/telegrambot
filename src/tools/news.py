from __future__ import annotations

import html
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

from src.config import Settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsItem:
    title: str
    link: str
    source: str | None
    published_at: str | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NewsSearchClient:
    google_news_url = "https://news.google.com/rss/search"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def search(self, query: str, limit: int | None = None) -> dict[str, Any]:
        import requests

        clean_query = query.strip()
        max_items = max(1, min(limit or self.settings.news_max_items, 10))
        rss_query = quote_plus(f"{clean_query} ekonomi finans when:7d")
        url = f"{self.google_news_url}?q={rss_query}&hl=tr&gl=TR&ceid=TR:tr"
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "telegram-economy-ai/1.0"},
                timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
            items = self._parse_items(response.text, max_items)
            return {
                "status": "ok" if items else "empty",
                "provider": "Google News RSS",
                "query": clean_query,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "items": [item.to_dict() for item in items],
            }
        except Exception as exc:
            logger.warning("Could not fetch news for %s: %s", clean_query, exc)
            return {
                "status": "error",
                "provider": "Google News RSS",
                "query": clean_query,
                "items": [],
                "error": str(exc),
            }

    def _parse_items(self, xml_text: str, limit: int) -> list[NewsItem]:
        root = ElementTree.fromstring(xml_text)
        channel = root.find("channel")
        if channel is None:
            return []

        items: list[NewsItem] = []
        for item in channel.findall("item"):
            title = _clean_text(item.findtext("title"))
            link = (item.findtext("link") or "").strip()
            published_at = _clean_text(item.findtext("pubDate")) or None
            source_element = item.find("source")
            source = _clean_text(source_element.text if source_element is not None else None) or None
            description = _clean_text(item.findtext("description"))
            summary = _build_summary(title, description, source)
            if title and link:
                items.append(
                    NewsItem(
                        title=title,
                        link=link,
                        source=source,
                        published_at=published_at,
                        summary=summary,
                    )
                )
            if len(items) >= limit:
                break
        return items


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_summary(title: str, description: str, source: str | None) -> str:
    if description and description.lower() != title.lower():
        cleaned = description
    else:
        cleaned = title
    if source and source not in cleaned:
        cleaned = f"{cleaned} Kaynak: {source}."
    if len(cleaned) > 240:
        cleaned = cleaned[:237].rstrip() + "..."
    return cleaned
