from __future__ import annotations

import html
from html.parser import HTMLParser
import logging
import re
import time
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
        clean_query = query.strip()
        max_items = max(1, min(limit or self.settings.news_max_items, 10))
        errors: list[str] = []
        any_success = False

        for rss_query in _rss_query_candidates(clean_query):
            try:
                xml_text = self._fetch_rss_xml(rss_query)
                any_success = True
            except Exception as exc:
                logger.warning("Could not fetch news for %s with query %s: %s", clean_query, rss_query, exc)
                errors.append(f"{rss_query}: {exc}")
                continue

            parse_limit = min(10, max_items + 5) if _relevance_terms(clean_query) else max_items
            items = self._parse_items(xml_text, parse_limit)
            items = _filter_relevant_items(clean_query, items)[:max_items]
            if items:
                return {
                    "status": "ok",
                    "provider": "Google News RSS",
                    "query": clean_query,
                    "rss_query": rss_query,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "items": [item.to_dict() for item in items],
                }
            errors.append(f"{rss_query}: empty result")

        return {
            "status": "empty" if any_success else "error",
            "provider": "Google News RSS",
            "query": clean_query,
            "items": [],
            "error": "; ".join(errors),
        }

    def _fetch_rss_xml(self, rss_query: str) -> str:
        import requests

        encoded_query = quote_plus(rss_query)
        url = f"{self.google_news_url}?q={encoded_query}&hl=tr&gl=TR&ceid=TR:tr"
        last_exception: Exception | None = None
        for attempt in range(2):
            try:
                response = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 telegram-economy-ai/1.0"},
                    timeout=self.settings.request_timeout_seconds,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                    time.sleep(0.4)
                    continue
                response.raise_for_status()
                return response.text
            except Exception as exc:
                last_exception = exc
                if attempt == 0:
                    time.sleep(0.4)
                    continue
        if last_exception:
            raise last_exception
        raise RuntimeError("Google News RSS did not return a response.")

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
    if _is_generic_google_text(cleaned):
        cleaned = ""
    if not cleaned:
        cleaned = _title_based_summary(title)

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
        if self.meta_description and not _is_generic_google_text(self.meta_description):
            return self.meta_description
        if self._paragraphs:
            text = " ".join(self._paragraphs[:3])
            if not _is_generic_google_text(text):
                return text
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


def _build_rss_query(query: str) -> str:
    clean = query.strip()
    lowered = clean.lower()
    if _is_turkey_economy_query(lowered):
        return "Türkiye ekonomi piyasalar when:7d"
    if _is_general_economy_query(lowered):
        return "ekonomi piyasalar Türkiye when:7d"
    aliases = {
        "amd": '"AMD" stock shares earnings AI',
        "bitcoin": '"bitcoin" crypto price ETF market',
        "altin": '"altin" fiyat piyasa ons gram',
        "altın": '"altın" fiyat piyasa ons gram',
        "gumus": '"gumus" fiyat piyasa',
        "gümüş": '"gümüş" fiyat piyasa',
        "nasdaq": '"Nasdaq" stock market',
        "s&p 500": '"S&P 500" stock market',
        "sp500": '"S&P 500" stock market',
        "bist 100": '"BIST 100" borsa piyasa',
        "petrol": '"petrol" brent fiyat piyasa',
    }
    for key, value in aliases.items():
        if key in lowered:
            return f"{value} when:7d"
    return f'"{clean}" ekonomi finans piyasa when:7d'


def _rss_query_candidates(query: str) -> list[str]:
    candidates = [_build_rss_query(query)]
    for fallback in _fallback_rss_queries(query):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _fallback_rss_queries(query: str) -> list[str]:
    lowered = query.lower()
    if _is_turkey_economy_query(lowered):
        return [
            "Türkiye ekonomi when:7d",
            "ekonomi piyasalar Türkiye when:7d",
            "ekonomi piyasalar when:7d",
        ]
    if _is_general_economy_query(lowered):
        return [
            "ekonomi piyasalar when:7d",
            "Türkiye ekonomi when:7d",
        ]
    return []


def _is_turkey_economy_query(lowered_query: str) -> bool:
    return (
        any(token in lowered_query for token in ["türkiye", "turkiye"])
        and any(token in lowered_query for token in ["ekonomi", "piyasa", "finans"])
    )


def _is_general_economy_query(lowered_query: str) -> bool:
    return any(token in lowered_query for token in ["ekonomi", "piyasa", "finans"]) and not _relevance_terms(
        lowered_query
    )


def _filter_relevant_items(query: str, items: list[NewsItem]) -> list[NewsItem]:
    items = [item for item in items if not _is_blocked_news_item(item)]
    terms = _relevance_terms(query)
    if not terms:
        return items
    filtered = []
    for item in items:
        haystack = " ".join(
            part for part in [item.title, item.summary, item.source or ""] if part
        ).lower()
        if any(term in haystack for term in terms):
            filtered.append(item)
    return filtered


def _is_blocked_news_item(item: NewsItem) -> bool:
    blocked_hosts = {
        "instagram.com",
        "facebook.com",
        "x.com",
        "twitter.com",
        "tiktok.com",
        "youtube.com",
    }
    host = urlparse(item.link).netloc.lower().replace("www.", "")
    source = (item.source or "").lower()
    return any(blocked in host or blocked in source for blocked in blocked_hosts)


def _relevance_terms(query: str) -> list[str]:
    lowered = query.lower()
    term_map = {
        "amd": ["amd", "advanced micro devices"],
        "nvidia": ["nvidia", "nvda"],
        "nvda": ["nvidia", "nvda"],
        "apple": ["apple", "aapl"],
        "aapl": ["apple", "aapl"],
        "tesla": ["tesla", "tsla"],
        "tsla": ["tesla", "tsla"],
        "microsoft": ["microsoft", "msft"],
        "msft": ["microsoft", "msft"],
        "altin": ["altin", "altın", "gold", "ons", "gram"],
        "altın": ["altin", "altın", "gold", "ons", "gram"],
        "gumus": ["gumus", "gümüş", "silver"],
        "gümüş": ["gumus", "gümüş", "silver"],
        "bitcoin": ["bitcoin", "btc"],
        "ethereum": ["ethereum", "eth"],
        "nasdaq": ["nasdaq"],
        "s&p 500": ["s&p 500", "sp 500", "s&p", "sp500"],
        "sp500": ["s&p 500", "sp 500", "s&p", "sp500"],
        "bist 100": ["bist 100", "bist100", "bist"],
        "petrol": ["petrol", "brent", "oil"],
    }
    for key, terms in term_map.items():
        if key in lowered:
            return terms
    return []


def _is_generic_google_text(text: str) -> bool:
    lowered = text.lower()
    generic_markers = [
        "comprehensive up-to-date news coverage",
        "aggregated from sources all over the world by google news",
        "google news",
    ]
    return any(marker in lowered for marker in generic_markers)


def _title_based_summary(title: str) -> str:
    clean_title = _strip_source_from_title(title)
    lowered = clean_title.lower()
    if any(word in lowered for word in ["yükseldi", "yukseldi", "arttı", "artti", "ralli"]):
        return (
            f"Haberde {clean_title} ifadesi one cikiyor. Bu, olumlu haber akisi "
            "veya risk algisindaki degisimin fiyatlamayi destekledigine isaret ediyor olabilir."
        )
    if any(word in lowered for word in ["düştü", "dustu", "geriledi", "azaldı", "azaldi"]):
        return (
            f"Haberde {clean_title} ifadesi one cikiyor. Bu, zayif haber akisi veya risk "
            "istahindaki bozulmanin fiyat uzerinde baski kurduguna isaret ediyor olabilir."
        )
    if any(word in lowered for word in ["beklenti", "faiz", "enflasyon", "fed", "merkez bank"]):
        return (
            f"Haber basligi {clean_title} temasina odaklaniyor. Bu tur makro basliklar "
            "kur, emtia ve endeks fiyatlamasinda beklenti kanaliyla etkili olabilir."
        )
    return (
        f"Haber basligi {clean_title} temasina odaklaniyor. Detay metni okunamadigi icin "
        "ozet baslik uzerinden yorumlandi."
    )


def _strip_source_from_title(title: str) -> str:
    clean = _clean_text(title)
    if " - " in clean:
        return clean.rsplit(" - ", 1)[0].strip()
    return clean


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
