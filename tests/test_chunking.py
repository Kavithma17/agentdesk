import unicodedata
from itertools import pairwise
from pathlib import Path

import pytest

from app.rag.chunking import MarkdownChunker, approx_token_count, chunk_directory

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"


def make_chunker(**overrides) -> MarkdownChunker:
    """Count 1 character as 1 token so sizes in tests are easy to reason about."""
    settings = {"count_tokens": len, "max_tokens": 1500, "overlap_tokens": 0, "min_tokens": 0}
    return MarkdownChunker(**{**settings, **overrides})


def body(chunk) -> str:
    return chunk.text.split("\n\n", 1)[1]


# --- structure -------------------------------------------------------------------------------


def test_chunk_text_is_prefixed_with_heading_path():
    md = "# Policy\n\n## Delivery\n\nKandy is Rs. 450."
    [chunk] = make_chunker().chunk(md, source="policy.md")
    assert chunk.section == "Policy > Delivery"
    assert chunk.text == "Policy > Delivery\n\nKandy is Rs. 450."
    assert chunk.source == "policy.md"


def test_sibling_heading_replaces_previous_one():
    md = "# Catalog\n\n## Audio\n\n### Buds\n\nRs. 8,500\n\n## Charging\n\n### Charger\n\nRs. 3,200"
    sections = [c.section for c in make_chunker().chunk(md, source="c.md")]
    assert sections == ["Catalog > Audio > Buds", "Catalog > Charging > Charger"]


def test_headings_without_body_produce_no_chunk():
    md = "# Title\n\n## Empty\n\n## Filled\n\nSome text."
    chunks = make_chunker().chunk(md, source="x.md")
    assert [c.section for c in chunks] == ["Title > Filled"]


def test_document_without_headings_uses_file_name_as_section():
    [chunk] = make_chunker().chunk("Repairs take 7 to 10 working days.", source="warranty_notes.txt")
    assert chunk.section == "warranty notes"
    assert chunk.text == "warranty notes\n\nRepairs take 7 to 10 working days."


def test_hash_inside_code_fence_is_not_a_heading():
    md = "# Setup\n\n```bash\n# install deps\n\npip install x\n```\n\nDone."
    [chunk] = make_chunker().chunk(md, source="s.md")
    assert chunk.section == "Setup"
    # The code block stays intact, blank line inside it included.
    assert "```bash\n# install deps\n\npip install x\n```" in chunk.text


def test_unclosed_fence_runs_to_end_of_document():
    md = "# A\n\n```\n# not a heading\n## also not"
    assert [c.section for c in make_chunker().chunk(md, source="a.md")] == ["A"]


def test_closing_hashes_are_removed_from_heading():
    [chunk] = make_chunker().chunk("## Delivery ##\n\nText.", source="d.md")
    assert chunk.section == "Delivery"


def test_text_is_normalized():
    # "කො" written as ක + ෙ + ා (decomposed) must match the precomposed form; CRLF and BOM go away.
    md = "﻿# Title\r\n\r\nකො\r\nline two"
    [chunk] = make_chunker().chunk(md, source="si.md")
    assert chunk.text == "Title\n\nකො\nline two"


# --- sizing ----------------------------------------------------------------------------------


def test_long_section_is_split_on_paragraphs():
    paragraphs = ["a" * 60, "b" * 60, "c" * 60]
    md = "# T\n\n" + "\n\n".join(paragraphs)
    chunks = make_chunker(max_tokens=130).chunk(md, source="x.md")
    assert [body(c) for c in chunks] == ["a" * 60 + "\n\n" + "b" * 60, "c" * 60]
    assert all(c.section == "T" for c in chunks)


def test_heading_prefix_counts_towards_the_limit():
    md = "# A long heading here\n\n" + "word " * 100
    chunks = make_chunker(max_tokens=100).chunk(md, source="x.md")
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_block_without_paragraph_breaks_is_split_on_sentences():
    sentence = "Items must be returned with the invoice. "
    md = sentence * 50  # one ~2000-char block with no blank lines, like text copied from a PDF
    chunks = make_chunker(max_tokens=300).chunk(md, source="policy.pdf")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 300
        assert body(chunk).endswith("invoice.")  # never cut mid-sentence
        assert "\n\n" not in body(chunk)  # still one paragraph, not broken into new ones


def test_long_table_is_split_between_rows():
    rows = [f"| Item {i} | Rs. {i * 100} |" for i in range(40)]
    md = "# Prices\n\n| Item | Price |\n|---|---|\n" + "\n".join(rows)
    chunks = make_chunker(max_tokens=200).chunk(md, source="p.md")
    assert len(chunks) > 1
    for chunk in chunks:
        assert all(line.startswith("|") and line.endswith("|") for line in body(chunk).split("\n"))


def test_word_longer_than_limit_is_cut():
    chunks = make_chunker(max_tokens=100).chunk("x" * 250, source="junk.txt")
    assert all(len(c.text) <= 100 for c in chunks)
    assert "".join(body(c) for c in chunks) == "x" * 250


def test_hard_cut_never_separates_sinhala_vowel_sign():
    text = "කො" * 200  # "කො" repeated: letter + vowel sign, no spaces
    chunks = make_chunker(max_tokens=51).chunk(text, source="si.txt")
    assert len(chunks) > 1
    for chunk in chunks:
        assert not unicodedata.category(body(chunk)[0]).startswith("M")


# --- overlap ---------------------------------------------------------------------------------


def test_split_section_repeats_last_sentence_of_previous_chunk():
    sentences = [f"Sentence number {i} is here." for i in range(20)]
    md = "# T\n\n" + " ".join(sentences)
    chunks = make_chunker(max_tokens=200, overlap_tokens=40).chunk(md, source="x.md")
    assert len(chunks) > 1
    for prev, nxt in pairwise(chunks):
        last_sentence = body(prev).rsplit(". ", 1)[-1]
        assert body(nxt).startswith(last_sentence)
        assert len(nxt.text) <= 200


def test_overlap_does_not_cross_sections():
    md = "# A\n\nAlpha text here.\n\n# B\n\nBeta text here."
    chunks = make_chunker(overlap_tokens=10).chunk(md, source="x.md")
    assert [body(c) for c in chunks] == ["Alpha text here.", "Beta text here."]


# --- merging small sections ------------------------------------------------------------------


def test_tiny_sections_are_merged_keeping_each_heading():
    md = "# FAQ\n\n## Shop?\n\nOnline only.\n\n## Hours?\n\n9 to 6."
    [chunk] = make_chunker(min_tokens=40).chunk(md, source="faq.md")
    assert chunk.section == "FAQ"
    assert chunk.text == "FAQ > Shop?\n\nOnline only.\n\nFAQ > Hours?\n\n9 to 6."


def test_tiny_sections_are_not_merged_past_the_limit():
    md = "# A\n\n" + "a" * 30 + "\n\n# B\n\n" + "b" * 30
    chunks = make_chunker(max_tokens=50, min_tokens=40).chunk(md, source="x.md")
    assert [c.section for c in chunks] == ["A", "B"]


def test_parts_of_one_split_section_are_not_merged_back():
    md = "# T\n\n" + "a" * 95 + "\n\n" + "b" * 5
    chunks = make_chunker(max_tokens=100, min_tokens=20).chunk(md, source="x.md")
    assert [body(c) for c in chunks] == ["a" * 95, "b" * 5]


# --- config and real data --------------------------------------------------------------------


@pytest.mark.parametrize(
    "settings",
    [{"max_tokens": 0}, {"overlap_tokens": 50, "max_tokens": 100}, {"min_tokens": 100, "max_tokens": 100}],
)
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        MarkdownChunker(**settings)


def test_approx_count_is_higher_for_sinhala_than_english_of_same_length():
    assert approx_token_count("ක" * 40) > approx_token_count("a" * 40)


def test_real_docs_chunk_cleanly():
    chunker = MarkdownChunker()
    chunks = chunk_directory(DOCS_DIR, chunker)
    assert {c.source for c in chunks} == {
        "delivery_policy.md",
        "returns_policy.md",
        "products.md",
        "faq.md",
    }
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(approx_token_count(c.text) <= chunker.max_tokens for c in chunks)
    assert any("Kandy" in c.text and "450" in c.text for c in chunks)
