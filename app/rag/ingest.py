"""Keep the vector store in sync with the docs folder, embedding only what changed.

Run `python -m app.rag.ingest` after adding, editing or deleting documents.

Chunk ids are content hashes, so comparing the ids stored for a document with the ids the
chunker produces now tells us exactly which chunks are new (embed them) and which are gone
(delete them). Unchanged chunks are never sent to the embedding API again.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from google.genai import errors as genai_errors

from app.rag.chunking import MarkdownChunker
from app.rag.embeddings import Embedder, EmbeddingError
from app.rag.loader import ChunkedDocument, FailedFile, load_directory
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Failures that affect one document and should not stop the others. Anything else is a bug.
DOCUMENT_ERRORS = (EmbeddingError, genai_errors.APIError, ValueError)


@dataclass
class IngestReport:
    documents: int = 0
    added: int = 0  # chunks embedded and stored
    deleted: int = 0  # chunks removed because their text changed or their file is gone
    unchanged: int = 0  # chunks already stored, not re-embedded
    removed_sources: list[str] = field(default_factory=list)
    failed: list[FailedFile] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.documents} documents: {self.added} chunks added, {self.deleted} deleted, "
            f"{self.unchanged} unchanged, {len(self.removed_sources)} files removed, "
            f"{len(self.failed)} failed, in {self.seconds:.2f}s"
        )


def ingest_directory(
    directory: Path, chunker: MarkdownChunker, embedder: Embedder, store: VectorStore
) -> IngestReport:
    started = time.perf_counter()
    loaded = load_directory(directory, chunker)
    report = IngestReport(documents=len(loaded.documents), failed=list(loaded.failed))

    for document in loaded.documents:
        try:
            _sync_document(document, embedder, store, report)
        except DOCUMENT_ERRORS as error:
            # The document keeps whatever was stored before, so search still works, just stale.
            logger.error("Could not ingest %s: %s", document.source, error)
            report.failed.append(FailedFile(document.source, f"{type(error).__name__}: {error}"))

    # A file that failed to load may still be on disk, so keep its old chunks rather than
    # wiping it from search because of, say, a temporary encoding problem.
    keep = {doc.source for doc in loaded.documents} | {f.source for f in loaded.failed}
    for source in sorted(store.sources() - keep):
        stale = store.ids_for_source(source)
        store.delete(sorted(stale))
        report.deleted += len(stale)
        report.removed_sources.append(source)
        logger.info("Removed %s (%d chunks): no longer in %s", source, len(stale), directory)

    report.seconds = time.perf_counter() - started
    log = logger.warning if report.failed else logger.info
    log("Ingested %s: %s", directory, report.summary())
    return report


def _sync_document(
    document: ChunkedDocument, embedder: Embedder, store: VectorStore, report: IngestReport
) -> None:
    stored = store.ids_for_source(document.source)
    new = [chunk for chunk in document.chunks if chunk.id not in stored]
    kept = [chunk for chunk in document.chunks if chunk.id in stored]
    gone = stored - {chunk.id for chunk in document.chunks}

    # Embed first: if the API fails, nothing in the store has been touched for this document.
    if new:
        store.upsert(new, embedder.embed_documents([chunk.text for chunk in new]))
    # Same text, but its position or the document hash may have moved; this costs no API call.
    if kept:
        store.update_metadata(kept)
    if gone:
        store.delete(sorted(gone))

    report.added += len(new)
    report.unchanged += len(kept)
    report.deleted += len(gone)
    if new or gone:
        logger.info("%s: %d added, %d deleted", document.source, len(new), len(gone))


def main() -> None:
    from app.core.config import get_settings
    from app.core.logging_config import setup_logging
    from app.rag.embeddings import get_embedder
    from app.rag.vector_store import ChromaVectorStore

    settings = get_settings()
    setup_logging(settings.log_level)
    chunker = MarkdownChunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_tokens=settings.chunk_min_tokens,
    )
    embedder = get_embedder()
    store = ChromaVectorStore.from_path(
        settings.vector_store_dir, settings.vector_collection, embedder.name, embedder.dim
    )
    report = ingest_directory(settings.docs_dir, chunker, embedder, store)
    print(report.summary())
    print(f"Store now holds {store.count()} chunks ({embedder.name})")
    for failure in report.failed:
        print(f"  FAILED {failure.source}: {failure.reason}")
    raise SystemExit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
