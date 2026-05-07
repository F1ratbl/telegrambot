from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    text: str
    created_at: float


@dataclass(frozen=True)
class ConversationProfile:
    preferred_name: str
    created_at: float


class InMemoryConversationMemory:
    """Small warm-instance memory for conversational continuity on serverless."""

    def __init__(self, max_messages: int = 10, ttl_seconds: int = 21600) -> None:
        self.max_messages = max(2, max_messages)
        self.ttl_seconds = max(60, ttl_seconds)
        self._messages: dict[str, deque[ConversationMessage]] = defaultdict(deque)
        self._profiles: dict[str, ConversationProfile] = {}
        self._lock = Lock()

    def snapshot(self, chat_id: str | None) -> list[ConversationMessage]:
        if not chat_id:
            return []
        with self._lock:
            self._prune_locked(chat_id)
            return list(self._messages.get(chat_id, []))

    def append(self, chat_id: str | None, role: str, text: str) -> None:
        if not chat_id or not text.strip():
            return
        if role not in {"user", "model"}:
            raise ValueError(f"Unsupported conversation role: {role}")
        with self._lock:
            self._prune_locked(chat_id)
            items = self._messages[chat_id]
            items.append(ConversationMessage(role=role, text=text.strip(), created_at=time.time()))
            while len(items) > self.max_messages:
                items.popleft()

    def remember_exchange(self, chat_id: str | None, user_text: str, model_text: str) -> None:
        self.append(chat_id, "user", user_text)
        self.append(chat_id, "model", model_text)

    def get_preferred_name(self, chat_id: str | None) -> str | None:
        if not chat_id:
            return None
        with self._lock:
            self._prune_locked(chat_id)
            profile = self._profiles.get(chat_id)
            if not profile:
                return None
            return profile.preferred_name

    def set_preferred_name(self, chat_id: str | None, preferred_name: str) -> None:
        clean_name = preferred_name.strip()
        if not chat_id or not clean_name:
            return
        with self._lock:
            self._prune_locked(chat_id)
            self._profiles[chat_id] = ConversationProfile(
                preferred_name=clean_name,
                created_at=time.time(),
            )

    def _prune_locked(self, chat_id: str) -> None:
        cutoff = time.time() - self.ttl_seconds
        items = self._messages.get(chat_id)
        if not items:
            profile = self._profiles.get(chat_id)
            if profile and profile.created_at < cutoff:
                self._profiles.pop(chat_id, None)
            return
        while items and items[0].created_at < cutoff:
            items.popleft()
        if not items:
            self._messages.pop(chat_id, None)
            self._profiles.pop(chat_id, None)
            return
        profile = self._profiles.get(chat_id)
        if profile and profile.created_at < cutoff:
            self._profiles.pop(chat_id, None)
