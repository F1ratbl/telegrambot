from __future__ import annotations

import html
from html.parser import HTMLParser
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus
from urllib.parse import urlparse
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
            if title and link:
                final_link, article_text = self._fetch_article_text(link)
                summary = _build_summary(title, article_text or description, source)
                items.append(
                    NewsItem(
                        title=title,
                        link=final_link or link,
                        source=source,
                        published_at=published_at,
                        summary=summary,
                    )
                )
            if len(items) >= limit:
                break
        return items

    def _fetch_article_text(self, link: str) -> tuple[str | None, str]:
        import requests

        try:
            response = requests.get(
                link,
                headers={"User-Agent": "Mozilla/5.0 telegram-economy-ai/1.0"},
                timeout=min(self.settings.request_timeout_seconds, 5.0),
                allow_redirects=True,
            )
            final_url = response.url or link
            content_type = response.headers.get("content-type", "")
            if response.status_code >= 400 or "html" not in content_type.lower():
                return final_url, ""
            parser = _ArticleTextParser()
            parser.feed(response.text)
            return _prefer_source_link(final_url, parser.canonical_url), parser.article_text()
        except Exception as exc:
            logger.info("Could not enrich news article %s: %s", link, exc)
            return link, ""


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_summary(title: str, description: str, source: str | None) -> str:
    cleaned = _remove_title_overlap(description, title)
    if not cleaned:
        cleaned = "Haber, ilgili varlikla ilgili son piyasa gelismelerine deginiyor."

    sentences = _split_sentences(cleaned)
    if sentences:
        cleaned = " ".join(sentences[:2])
    if source and source not in cleaned:
        cleaned = f"{cleaned} Kaynak: {source}."
    return _limit_text(cleaned, 280)


class _ArticleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self._in_paragraph = False
        self._paragraphs: list[str] = []
        self._current: list[str] = []
        self.title = ""
        self.meta_description = ""
        self.canonical_url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            self._current = []
        elif tag == "p":
            self._in_paragraph = True
            self._current = []
        elif tag == "meta":
            name = attrs_dict.get("name", "").lower()
            prop = attrs_dict.get("property", "").lower()
            if name == "description" or prop == "og:description":
                content = _clean_text(attrs_dict.get("content"))
                if content and not self.meta_description:
                    self.meta_description = content
        elif tag == "link":
            rel = attrs_dict.get("rel", "").lower()
            href = attrs_dict.get("href", "")
            if rel == "canonical" and href:
                self.canonical_url = href

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title" and self._in_title:
            self.title = _clean_text(" ".join(self._current))
            self._current = []
            self._in_title = False
        elif tag == "p" and self._in_paragraph:
            paragraph = _clean_text(" ".join(self._current))
            if len(paragraph) >= 80:
                self._paragraphs.append(paragraph)
            self._current = []
            self._in_paragraph = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title or self._in_paragraph:
            self._current.append(data)

    def article_text(self) -> str:
        if self.meta_description:
            return self.meta_description
        if self._paragraphs:
            return " ".join(self._paragraphs[:3])
        return self.title


def _remove_title_overlap(text: str, title: str) -> str:
    cleaned = _clean_text(text)
    clean_title = _clean_text(title)
    if not cleaned:
        return ""
    if cleaned.lower() == clean_title.lower():
        return ""
    if clean_title and cleaned.lower().startswith(clean_title.lower()):
        cleaned = cleaned[len(clean_title) :].strip(" -:|")
    return cleaned


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _limit_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _prefer_source_link(final_url: str, canonical_url: str | None) -> str:
    if canonical_url and "news.google." not in urlparse(canonical_url).netloc:
        return canonical_url
    return final_url
