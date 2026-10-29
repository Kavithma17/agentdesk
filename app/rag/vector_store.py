"""Storage-agnostic vector store layer. Ingestion and retrieval talk to `VectorStore`, never to a
database client, so ChromaDB can later be swapped for Qdrant or pgvector."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb

from app.rag.chunking import Chunk
from app.rag.embeddings import Vector

logger = logging.getLogger(__name__)


class EmbedderMismatchError(RuntimeError):
    """The store holds vectors from a different embedder, so its vectors can't be compared."""


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    source: str
    section: str
    score: float  # cosine similarity: 1.0 means same meaning, around 0 means unrelated


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        """Add chunks, or overwrite ones whose id is already stored."""
        ...

    def update_metadata(self, chunks: Sequence[Chunk]) -> None:
        """Refresh position and document info of stored chunks without re-embedding them."""
        ...

    def delete(self, chunk_ids: Sequence[str]) -> None: ...

    def ids_for_source(self, source: str) -> set[str]:
        """Ids stored for one document, so ingestion can tell which chunks are new or gone."""
        ...

    def sources(self) -> set[str]:
        """Every document that has chunks stored, so ingestion can remove deleted files."""
        ...

    def search(self, query_vector: Vector, k: int) -> list[SearchResult]:
        """The k most similar chunks, best first."""
        ...

    def count(self) -> int: ...


class ChromaVectorStore:
    """A persistent ChromaDB collection that remembers which embedder filled it."""

    def __init__(self, client: chromadb.ClientAPI, collection: str, embedder_name: str, dim: int):
        self._embedder_name = embedder_name
        self._dim = dim
        self._collection = client.get_or_create_collection(
            collection,
            # We always pass our own vectors; None stops Chroma downloading its default model.
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedder": embedder_name, "dim": dim},
        )
        stored = (self._collection.metadata or {}).get("embedder")
        if stored != embedder_name:
            raise EmbedderMismatchError(
                f"Collection '{collection}' was built with {stored}, not {embedder_name}. "
                "Delete it and re-ingest, or switch back to the original embedder."
            )
        # Chroma rejects writes bigger than this, so large upserts are split.
        self._max_batch = client.get_max_batch_size()

    @classmethod
    def from_path(
        cls, path: Path, collection: str, embedder_name: str, dim: int
    ) -> "ChromaVectorStore":
        return cls(chromadb.PersistentClient(path=str(path)), collection, embedder_name, dim)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"Got {len(chunks)} chunks but {len(vectors)} vectors")
        if any(len(vector) != self._dim for vector in vectors):
            raise ValueError(f"Every vector must have {self._dim} dimensions")
        for start in range(0, len(chunks), self._max_batch):
            batch = chunks[start : start + self._max_batch]
            self._collection.upsert(
                ids=[chunk.id for chunk in batch],
                embeddings=[list(v) for v in vectors[start : start + self._max_batch]],
                documents=[chunk.text for chunk in batch],
                metadatas=[_metadata(chunk) for chunk in batch],
            )
        logger.debug("Upserted %d chunks", len(chunks))

    def update_metadata(self, chunks: Sequence[Chunk]) -> None:
        for start in range(0, len(chunks), self._max_batch):
            batch = chunks[start : start + self._max_batch]
            self._collection.update(
                ids=[chunk.id for chunk in batch], metadatas=[_metadata(chunk) for chunk in batch]
            )

    def delete(self, chunk_ids: Sequence[str]) -> None:
        # Chroma raises on an empty id list instead of doing nothing.
        if chunk_ids:
            self._collection.delete(ids=list(chunk_ids))
            logger.debug("Deleted %d chunks", len(chunk_ids))

    def ids_for_source(self, source: str) -> set[str]:
        return set(self._collection.get(where={"source": source}, include=[])["ids"])

    def sources(self) -> set[str]:
        metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        return {str(meta["source"]) for meta in metadatas}

    def search(self, query_vector: Vector, k: int) -> list[SearchResult]:
        if k < 1:
            raise ValueError("k must be at least 1")
        if len(query_vector) != self._dim:
            raise ValueError(f"Query vector must have {self._dim} dimensions")
        result = self._collection.query(
            query_embeddings=[list(query_vector)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma returns one list per query vector; we sent one, so take [0] of each.
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            SearchResult(
                chunk_id=chunk_id,
                text=text or "",
                source=str(meta["source"]),
                section=str(meta["section"]),
                # Chroma's cosine distance is 1 - cosine similarity.
                score=1.0 - distance,
            )
            for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances)
        ]

    def count(self) -> int:
        return self._collection.count()


def _metadata(chunk: Chunk) -> dict[str, str | int]:
    return {
        "source": chunk.source,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "doc_hash": chunk.doc_hash,
    }
