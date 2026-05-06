from __future__ import annotations

from src.config import Settings


class GeminiEmbeddings:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text], task_type="RETRIEVAL_QUERY")
        return vectors[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if not clean_texts:
            return []
        if not self.settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required for embeddings.")

        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.settings.google_api_key)

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.settings.embedding_dimensions,
        )
        response = self._client.models.embed_content(
            model=self.settings.embedding_model,
            contents=clean_texts,
            config=config,
        )
        return [list(embedding.values) for embedding in response.embeddings]
