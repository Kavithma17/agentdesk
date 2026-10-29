from collections import Counter
from pathlib import Path

import pytest

from app.evals.retrieval import RetrievalCase, load_cases
from app.rag.chunking import MarkdownChunker
from app.rag.loader import load_directory

ROOT = Path(__file__).parent.parent
CASES_PATH = ROOT / "evals" / "retrieval_cases.jsonl"
DOCS_DIR = ROOT / "data" / "docs"


def test_every_case_has_an_answer_in_the_current_docs():
    # If someone edits the docs and an answer disappears, the eval set is stale: fail here,
    # not later as a mysteriously bad benchmark score.
    chunks = load_directory(DOCS_DIR, MarkdownChunker()).chunks
    for case in load_cases(CASES_PATH):
        for snippet in case.relevant:
            assert any(snippet in c.text for c in chunks), f"{case.id}: {snippet!r} not in docs"


def test_eval_set_covers_every_language():
    counts = Counter(case.lang for case in load_cases(CASES_PATH))
    assert all(counts[lang] >= 8 for lang in ("en", "si", "singlish")), counts


def test_relevance_is_matched_on_normalized_text():
    case = RetrievalCase("x", "si", "q", relevant=("කො",))  # precomposed
    assert case.is_relevant("කො")  # decomposed form of the same letter


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"id": "a", "lang": "fr", "query": "q", "relevant": ["x"]}', "unknown lang"),
        ('{"id": "a", "lang": "en", "query": "q", "relevant": []}', "at least one"),
        ('{"id": "a", "lang": "en", "relevant": ["x"]}', "query"),
        ("not json", ":1:"),
    ],
)
def test_bad_lines_fail_with_line_number(tmp_path, line, message):
    path = tmp_path / "cases.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_cases(path)


def test_duplicate_ids_are_rejected(tmp_path):
    line = '{"id": "a", "lang": "en", "query": "q", "relevant": ["x"]}\n'
    path = tmp_path / "cases.jsonl"
    path.write_text(line * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)
