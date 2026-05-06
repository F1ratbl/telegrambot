from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.ai.embeddings import GeminiEmbeddings
from src.config import Settings


@dataclass(frozen=True)
class KnowledgeChunk:
    title: str
    content: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


class QdrantKnowledgeBase:
    def __init__(self, settings: Settings, embeddings: GeminiEmbeddings | None = None) -> None:
        self.settings = settings
        self.embeddings = embeddings or GeminiEmbeddings(settings)
        self._client = None

    def ensure_collection(self) -> None:
        from qdrant_client import models

        client = self._get_client()
        if client.collection_exists(self.settings.qdrant_collection):
            return
        client.create_collection(
            collection_name=self.settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=self.settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(self, chunks: Iterable[KnowledgeChunk], batch_size: int = 32) -> int:
        from qdrant_client import models

        self.ensure_collection()
        client = self._get_client()
        total = 0
        batch: list[KnowledgeChunk] = []

        for chunk in chunks:
            if chunk.content.strip():
                batch.append(chunk)
            if len(batch) >= batch_size:
                total += self._upsert_batch(client, models, batch)
                batch = []

        if batch:
            total += self._upsert_batch(client, models, batch)
        return total

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        client = self._get_client()
        query_vector = self.embeddings.embed_query(query)
        result = client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        points = getattr(result, "points", result)
        matches = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            matches.append(
                {
                    "score": getattr(point, "score", None),
                    "title": payload.get("title"),
                    "content": payload.get("content"),
                    "source": payload.get("source"),
                    "chunk_index": payload.get("chunk_index"),
                    "metadata": payload.get("metadata") or {},
                }
            )
        return matches

    def _upsert_batch(self, client: Any, models: Any, chunks: list[KnowledgeChunk]) -> int:
        vectors = self.embeddings.embed_documents([chunk.content for chunk in chunks])
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            points.append(
                models.PointStruct(
                    id=self._point_id(chunk),
                    vector=vector,
                    payload={
                        "title": chunk.title,
                        "content": chunk.content,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                    },
                )
            )
        client.upsert(collection_name=self.settings.qdrant_collection, points=points)
        return len(points)

    def _get_client(self) -> Any:
        if not self.settings.qdrant_url:
            raise RuntimeError("QDRANT_URL is required.")
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=self.settings.qdrant_timeout_seconds,
            )
        return self._client

    def _point_id(self, chunk: KnowledgeChunk) -> str:
        raw = f"{chunk.source}:{chunk.chunk_index}:{chunk.title}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))
