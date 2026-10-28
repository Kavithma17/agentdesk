"""Load the knowledge-base documents from disk and chunk them, one file at a time.

Run `python -m app.rag.loader` for a dry run that chunks everything and prints a summary.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.chunking import Chunk, MarkdownChunker

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt"}
# A knowledge-base page is a few KB; anything this big is almost surely a mistake (a dump, a log).
DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class ChunkedDocument:
    source: str  # path relative to the docs folder, with "/" on every OS
    doc_hash: str
    chunks: list[Chunk]


@dataclass(frozen=True)
class FailedFile:
    source: str
    reason: str


@dataclass
class LoadReport:
    documents: list[ChunkedDocument] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)  # read fine but had no text to chunk
    failed: list[FailedFile] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def chunks(self) -> list[Chunk]:
        return [chunk for doc in self.documents for chunk in doc.chunks]

    def summary(self) -> str:
        tokens = sum(chunk.token_count for chunk in self.chunks)
        return (
            f"{len(self.documents)} documents -> {len(self.chunks)} chunks (~{tokens} tokens), "
            f"{len(self.empty)} empty, {len(self.failed)} failed, in {self.seconds:.2f}s"
        )


def find_documents(directory: Path) -> list[Path]:
    """Supported files under directory, in a stable order. Hidden files and folders are skipped."""
    if not directory.is_dir():
        # A wrong DOCS_DIR is a setup error: fail loudly instead of ingesting nothing.
        raise FileNotFoundError(f"Docs folder not found: {directory.resolve()}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )


def load_directory(
    directory: Path,
    chunker: MarkdownChunker,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> LoadReport:
    """Chunk every supported file. A file that cannot be read is logged and reported, and the
    rest still load: one broken file must not stop the whole knowledge base from updating."""
    started = time.perf_counter()
    report = LoadReport()
    for path in find_documents(directory):
        # Relative POSIX path, not just the name: two folders may both have a faq.md, and the
        # source is part of every chunk ID, so it must be the same on Windows and Linux.
        source = path.relative_to(directory).as_posix()
        try:
            size = path.stat().st_size
            if size > max_file_bytes:
                raise ValueError(f"file is {size} bytes, over the {max_file_bytes} byte limit")
            # Strict UTF-8: guessing another encoding would quietly store garbled Sinhala.
            text = path.read_text(encoding="utf-8")
            chunks = chunker.chunk(text, source=source)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            logger.error("Skipping %s: %s", source, error)
            report.failed.append(FailedFile(source, str(error)))
            continue
        except Exception as error:
            # A bug in the chunker: keep the traceback, but still load the other files.
            logger.exception("Unexpected error while chunking %s", source)
            report.failed.append(FailedFile(source, f"{type(error).__name__}: {error}"))
            continue

        if not chunks:
            logger.warning("No text to chunk in %s", source)
            report.empty.append(source)
            continue
        logger.debug("%s: %d chunks", source, len(chunks))
        report.documents.append(ChunkedDocument(source, chunks[0].doc_hash, chunks))

    report.seconds = time.perf_counter() - started
    log = logger.warning if report.failed else logger.info
    log("Loaded %s: %s", directory, report.summary())
    return report


def main() -> None:
    from app.core.config import get_settings
    from app.core.logging_config import setup_logging

    settings = get_settings()
    setup_logging(settings.log_level)
    chunker = MarkdownChunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_tokens=settings.chunk_min_tokens,
    )
    logger.info("Chunker fingerprint %s (%s)", chunker.fingerprint, chunker.tokenizer_name)
    report = load_directory(settings.docs_dir, chunker)
    for doc in report.documents:
        print(f"  {doc.source}: {len(doc.chunks)} chunks, doc_hash {doc.doc_hash[:12]}")
    for failure in report.failed:
        print(f"  FAILED {failure.source}: {failure.reason}")
    raise SystemExit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
