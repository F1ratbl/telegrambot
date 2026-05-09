from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping


DEFAULT_MARKET_SYMBOLS = [
    "BIST100",
    "SP500",
    "NASDAQ",
    "DAX",
    "BRENT",
    "GOLD",
    "USDTRY",
    "EURTRY",
    "BTCUSD",
]


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    value = _optional(env.get(name))
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    value = _optional(env.get(name))
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _optional_float(env: Mapping[str, str], name: str) -> float | None:
    value = _optional(env.get(name))
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _csv(env: Mapping[str, str], name: str, default: list[str]) -> list[str]:
    raw_value = _optional(env.get(name))
    if raw_value is None:
        return list(default)
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return values or list(default)


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str | None = None
    telegram_webhook_secret: str | None = None

    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float | None = None
    gemini_max_output_tokens: int = 1200
    gemini_max_tool_rounds: int = 4
    gemini_thinking_level: str | None = None
    gemini_retry_attempts: int = 3
    gemini_retry_base_delay_seconds: float = 1.5

    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "economy_knowledge"
    qdrant_timeout_seconds: float = 10.0
    kb_top_k: int = 5

    market_default_symbols: list[str] = field(default_factory=lambda: list(DEFAULT_MARKET_SYMBOLS))
    request_timeout_seconds: float = 8.0
    news_max_items: int = 5
    news_move_threshold_percent: float = 2.0

    memory_max_messages: int = 10
    memory_ttl_seconds: int = 60 * 60 * 6
    timezone: str = "Europe/Istanbul"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        return cls(
            telegram_bot_token=_optional(env.get("TELEGRAM_BOT_TOKEN")),
            telegram_webhook_secret=_optional(env.get("TELEGRAM_WEBHOOK_SECRET")),
            google_api_key=_optional(env.get("GOOGLE_API_KEY")) or _optional(env.get("GEMINI_API_KEY")),
            gemini_model=_optional(env.get("GEMINI_MODEL")) or "gemini-2.5-flash",
            gemini_temperature=_optional_float(env, "GEMINI_TEMPERATURE"),
            gemini_max_output_tokens=_int(env, "GEMINI_MAX_OUTPUT_TOKENS", 1200),
            gemini_max_tool_rounds=_int(env, "GEMINI_MAX_TOOL_ROUNDS", 4),
            gemini_thinking_level=_optional(env.get("GEMINI_THINKING_LEVEL")),
            gemini_retry_attempts=_int(env, "GEMINI_RETRY_ATTEMPTS", 3),
            gemini_retry_base_delay_seconds=_float(env, "GEMINI_RETRY_BASE_DELAY_SECONDS", 1.5),
            embedding_model=_optional(env.get("EMBEDDING_MODEL")) or "gemini-embedding-001",
            embedding_dimensions=_int(env, "EMBEDDING_DIMENSIONS", 768),
            qdrant_url=_optional(env.get("QDRANT_URL")),
            qdrant_api_key=_optional(env.get("QDRANT_API_KEY")),
            qdrant_collection=_optional(env.get("QDRANT_COLLECTION")) or "economy_knowledge",
            qdrant_timeout_seconds=_float(env, "QDRANT_TIMEOUT_SECONDS", 10.0),
            kb_top_k=_int(env, "KB_TOP_K", 5),
            market_default_symbols=_csv(env, "MARKET_DEFAULT_SYMBOLS", DEFAULT_MARKET_SYMBOLS),
            request_timeout_seconds=_float(env, "REQUEST_TIMEOUT_SECONDS", 8.0),
            news_max_items=_int(env, "NEWS_MAX_ITEMS", 5),
            news_move_threshold_percent=_float(env, "NEWS_MOVE_THRESHOLD_PERCENT", 2.0),
            memory_max_messages=_int(env, "MEMORY_MAX_MESSAGES", 10),
            memory_ttl_seconds=_int(env, "MEMORY_TTL_SECONDS", 60 * 60 * 6),
            timezone=_optional(env.get("APP_TIMEZONE")) or "Europe/Istanbul",
        )

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.google_api_key)

    @property
    def qdrant_enabled(self) -> bool:
        return bool(self.qdrant_url)
