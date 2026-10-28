import logging
from pathlib import Path

import pytest

from app.rag.chunking import MarkdownChunker, approx_token_count
from app.rag.loader import find_documents, load_directory

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"


def write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def test_finds_supported_files_recursively_and_skips_the_rest(tmp_path):
    write(tmp_path / "a.md", "# A\n\nAlpha.")
    write(tmp_path / "notes" / "b.TXT", "Beta.")
    write(tmp_path / "image.png", b"\x89PNG")
    write(tmp_path / ".hidden.md", "secret")
    write(tmp_path / ".git" / "c.md", "not a doc")
    found = [p.relative_to(tmp_path).as_posix() for p in find_documents(tmp_path)]
    assert found == ["a.md", "notes/b.TXT"]


def test_missing_folder_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "nope", MarkdownChunker())


def test_source_is_relative_posix_path_so_same_names_do_not_clash(tmp_path):
    write(tmp_path / "en" / "faq.md", "# FAQ\n\nSame text.")
    write(tmp_path / "si" / "faq.md", "# FAQ\n\nSame text.")
    report = load_directory(tmp_path, MarkdownChunker())
    assert [d.source for d in report.documents] == ["en/faq.md", "si/faq.md"]
    assert len({c.id for c in report.chunks}) == 2


def test_bad_file_is_reported_and_the_rest_still_load(tmp_path, caplog):
    write(tmp_path / "good.md", "# Good\n\nFine.")
    write(tmp_path / "latin1.md", "Caf\xe9".encode("latin-1"))  # not valid UTF-8
    with caplog.at_level(logging.ERROR):
        report = load_directory(tmp_path, MarkdownChunker())
    assert [d.source for d in report.documents] == ["good.md"]
    assert [f.source for f in report.failed] == ["latin1.md"]
    assert "latin1.md" in caplog.text


def test_oversized_file_is_skipped(tmp_path):
    write(tmp_path / "dump.txt", "x" * 2000)
    report = load_directory(tmp_path, MarkdownChunker(), max_file_bytes=1000)
    assert report.documents == []
    assert "limit" in report.failed[0].reason


def test_chunker_bug_does_not_stop_other_files(tmp_path, caplog):
    class BrokenOnB(MarkdownChunker):
        def chunk(self, text, source):
            if source == "b.md":
                raise RuntimeError("boom")
            return super().chunk(text, source)

    write(tmp_path / "a.md", "Alpha.")
    write(tmp_path / "b.md", "Beta.")
    write(tmp_path / "c.md", "Gamma.")
    report = load_directory(tmp_path, BrokenOnB())
    assert [d.source for d in report.documents] == ["a.md", "c.md"]
    assert report.failed[0].reason == "RuntimeError: boom"
    assert "Traceback" in caplog.text


def test_empty_file_is_reported_as_empty_not_failed(tmp_path):
    write(tmp_path / "blank.md", "# Only a heading\n\n   \n")
    report = load_directory(tmp_path, MarkdownChunker())
    assert report.empty == ["blank.md"]
    assert report.failed == []


def test_summary_counts_everything(tmp_path):
    write(tmp_path / "a.md", "# A\n\nAlpha.")
    write(tmp_path / "blank.md", "")
    write(tmp_path / "bad.md", b"\xff\xfe\xfa")
    summary = load_directory(tmp_path, MarkdownChunker()).summary()
    assert summary.startswith("1 documents -> 1 chunks")
    assert "1 empty, 1 failed" in summary


def test_real_docs_load_cleanly():
    chunker = MarkdownChunker()
    report = load_directory(DOCS_DIR, chunker)
    assert report.failed == [] and report.empty == []
    assert {d.source for d in report.documents} == {
        "delivery_policy.md",
        "returns_policy.md",
        "products.md",
        "faq.md",
    }
    chunks = report.chunks
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(approx_token_count(c.text) <= chunker.max_tokens for c in chunks)
    assert any("Kandy" in c.text and "450" in c.text for c in chunks)
