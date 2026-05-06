from __future__ import annotations

import logging
from typing import Any

from src.config import Settings
from src.qdrant.knowledge_store import QdrantKnowledgeBase


logger = logging.getLogger(__name__)


class KnowledgeBaseTool:
    def __init__(self, settings: Settings, store: QdrantKnowledgeBase | None = None) -> None:
        self.settings = settings
        self.store = store or QdrantKnowledgeBase(settings)

    def search(self, query: str, limit: Any = None) -> dict[str, Any]:
        clean_query = query.strip()
        if not clean_query:
            return {"status": "error", "message": "Knowledge base query is empty.", "matches": []}
        if not self.settings.qdrant_enabled:
            return {
                "status": "disabled",
                "message": "QDRANT_URL is not configured, so the knowledge base is unavailable.",
                "matches": [],
            }

        safe_limit = self._safe_limit(limit)
        try:
            matches = self.store.search(clean_query, limit=safe_limit)
            return {"status": "ok", "matches": matches, "count": len(matches)}
        except Exception as exc:
            logger.exception("Knowledge base search failed.")
            return {"status": "error", "message": str(exc), "matches": []}

    def _safe_limit(self, limit: Any) -> int:
        try:
            parsed = int(limit)
        except (TypeError, ValueError):
            parsed = self.settings.kb_top_k
        return min(max(parsed, 1), 10)
