"""Provider-agnostic embedding layer. Ingestion and retrieval talk to `Embedder`, never to a vendor SDK.

Documents and queries are embedded separately because good retrieval models are asymmetric:
Gemini uses a `task_type`, e5 uses "passage: " / "query: " prefixes.
"""

import logging
import math
from collections.abc import Sequence
from typing import Protocol

from google import genai
from google.genai import types

from app.core.config import get_settings
from app.core.llm import RETRYABLE_STATUS_CODES

logger = logging.getLogger(__name__)

Vector = list[float]

# The Gemini API accepts at most 100 texts per embed request.
GEMINI_MAX_BATCH = 100


class EmbeddingError(RuntimeError):
    """The provider returned something we can't trust (wrong count, wrong dimension, zero vector)."""


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Stable id like "gemini/gemini-embedding-2@768", stored with the vectors so that a
        model change is detected and everything gets re-embedded."""
        ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


def l2_normalize(vector: Sequence[float]) -> Vector:
    """Scale to length 1, so the dot product equals cosine similarity.

    Gemini only returns normalized vectors at 3072 dimensions, so smaller outputs need this.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise EmbeddingError("Provider returned an all-zero vector")
    return [x / norm for x in vector]


class GeminiEmbedder:
    def __init__(
        self, client: genai.Client, model: str, dim: int, batch_size: int = GEMINI_MAX_BATCH
    ) -> None:
        if not 1 <= batch_size <= GEMINI_MAX_BATCH:
            raise ValueError(f"batch_size must be between 1 and {GEMINI_MAX_BATCH}")
        self._client = client
        self._model = model
        self._dim = dim
        self._batch_size = batch_size

    @classmethod
    def from_api_key(cls, api_key: str, model: str, dim: int) -> "GeminiEmbedder":
        retry = types.HttpRetryOptions(
            attempts=4, initial_delay=1.0, max_delay=20.0, http_status_codes=RETRYABLE_STATUS_CODES
        )
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=retry))
        return cls(client, model, dim)

    @property
    def name(self) -> str:
        return f"gemini/{self._model}@{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed(batch, task_type="RETRIEVAL_DOCUMENT"))
            logger.debug("Embedded %d/%d documents with %s", len(vectors), len(texts), self.name)
        return vectors

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: Sequence[str], task_type: str) -> list[Vector]:
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty or whitespace-only text")

        # One Content per text: given a plain list of strings, gemini-embedding-2 merges
        # them all into a single vector instead of returning one per text.
        contents = [types.Content(parts=[types.Part(text=text)]) for text in texts]
        response = self._client.models.embed_content(
            model=self._model,
            contents=contents,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self._dim),
        )

        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            raise EmbeddingError(f"Sent {len(texts)} texts but got {len(embeddings)} vectors back")
        vectors: list[Vector] = []
        for embedding in embeddings:
            values = embedding.values or []
            if len(values) != self._dim:
                raise EmbeddingError(f"Expected {self._dim} dimensions, got {len(values)}")
            vectors.append(l2_normalize(values))
        return vectors


def get_embedder() -> Embedder:
    settings = get_settings()
    if settings.embedding_provider == "gemini":
        return GeminiEmbedder.from_api_key(
            settings.gemini_api_key.get_secret_value(),
            settings.embedding_model,
            settings.embedding_dim,
        )
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
