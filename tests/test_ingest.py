import hashlib
import uuid

import chromadb
import pytest

from app.rag.chunking import MarkdownChunker
from app.rag.embeddings import EmbeddingError
from app.rag.ingest import ingest_directory
from app.rag.vector_store import ChromaVectorStore

DIM = 8


class FakeEmbedder:
    """Deterministic vectors from a text hash; records every text it was asked to embed."""

    name = "fake/hash@8"
    dim = DIM

    def __init__(self, fail_on: str | None = None):
        self.embedded: list[str] = []
        self.fail_on = fail_on  # raise if any text contains this

    def embed_documents(self, texts):
        if self.fail_on and any(self.fail_on in text for text in texts):
            raise EmbeddingError("simulated API failure")
        self.embedded.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    def _vector(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 + 0.01 for b in digest[:DIM]]


@pytest.fixture
def docs(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "faq.md").write_text(
        "# FAQ\n\n## Warranty\n\nPhones have a one year warranty.\n\n"
        "## Hours\n\nWe are open from nine to six, Monday to Saturday.\n",
        encoding="utf-8",
    )
    (folder / "delivery.md").write_text(
        "# Delivery\n\nDelivery within Colombo costs Rs. 350 and takes two days.\n",
        encoding="utf-8",
    )
    return folder


@pytest.fixture
def store(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return ChromaVectorStore(client, f"test-{uuid.uuid4().hex}", FakeEmbedder.name, DIM)


@pytest.fixture
def chunker():
    # Tiny limits so the sample docs split into several chunks.
    return MarkdownChunker(max_tokens=30, overlap_tokens=0, min_tokens=0)


def test_first_run_embeds_every_chunk(docs, chunker, store):
    embedder = FakeEmbedder()

    report = ingest_directory(docs, chunker, embedder, store)

    assert report.added == store.count() == len(embedder.embedded) > 0
    assert report.deleted == report.unchanged == 0
    assert report.failed == []
    assert store.sources() == {"faq.md", "delivery.md"}


def test_second_run_embeds_nothing(docs, chunker, store):
    ingest_directory(docs, chunker, FakeEmbedder(), store)
    embedder = FakeEmbedder()

    report = ingest_directory(docs, chunker, embedder, store)

    assert embedder.embedded == []
    assert report.added == report.deleted == 0
    assert report.unchanged == store.count()


def test_editing_a_section_re_embeds_only_that_chunk(docs, chunker, store):
    ingest_directory(docs, chunker, FakeEmbedder(), store)
    before = store.count()
    faq = docs / "faq.md"
    faq.write_text(faq.read_text(encoding="utf-8").replace("one year", "two year"), "utf-8")
    embedder = FakeEmbedder()

    report = ingest_directory(docs, chunker, embedder, store)

    assert report.added == report.deleted == 1
    assert len(embedder.embedded) == 1 and "two year" in embedder.embedded[0]
    assert store.count() == before


def test_inserting_a_section_refreshes_positions_of_later_chunks(docs, chunker, store):
    ingest_directory(docs, chunker, FakeEmbedder(), store)
    (docs / "delivery.md").write_text(
        "# Intro\n\nWe deliver island wide, every day.\n\n"
        "# Delivery\n\nDelivery within Colombo costs Rs. 350 and takes two days.\n",
        encoding="utf-8",
    )

    ingest_directory(docs, chunker, FakeEmbedder(), store)

    # The old Delivery chunk was not re-embedded, but it is now second in the document.
    stored = store._collection.get(where={"source": "delivery.md"}, include=["metadatas"])
    indexes = sorted(meta["chunk_index"] for meta in stored["metadatas"])
    assert indexes == list(range(len(indexes)))
    assert len(indexes) > 1


def test_deleted_file_is_removed_from_the_store(docs, chunker, store):
    ingest_directory(docs, chunker, FakeEmbedder(), store)
    (docs / "delivery.md").unlink()

    report = ingest_directory(docs, chunker, FakeEmbedder(), store)

    assert report.removed_sources == ["delivery.md"]
    assert store.sources() == {"faq.md"}


def test_embedding_failure_skips_only_that_document(docs, chunker, store):
    report = ingest_directory(docs, chunker, FakeEmbedder(fail_on="Colombo"), store)

    assert [f.source for f in report.failed] == ["delivery.md"]
    assert store.sources() == {"faq.md"}


def test_unreadable_file_keeps_its_old_chunks(docs, chunker, store):
    ingest_directory(docs, chunker, FakeEmbedder(), store)
    old_ids = store.ids_for_source("delivery.md")
    (docs / "delivery.md").write_bytes(b"\xff\xfe broken encoding")

    report = ingest_directory(docs, chunker, FakeEmbedder(), store)

    assert [f.source for f in report.failed] == ["delivery.md"]
    assert report.removed_sources == []
    assert store.ids_for_source("delivery.md") == old_ids
