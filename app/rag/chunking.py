"""Split markdown documents into structure-aware, token-sized chunks for retrieval."""

import hashlib
import math
import re  # regular expressions like #
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")  #methn (#, ##, ###.....)hoynw
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
# Sinhala text mostly uses ".", but also accept the danda and kunddaliya used in older text.
SENTENCE_END_RE = re.compile(r"(?<=[.!?।෴])\s+")
WHITESPACE_RE = re.compile(r"\s+")
NEWLINE_RE = re.compile(r"\n")

# Embedding models such as multilingual-e5 stop at 512 tokens; 400 leaves headroom for the
# approximate counter being off and for the model's own prefix and special tokens.
DEFAULT_MAX_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 50
DEFAULT_MIN_TOKENS = 50

# Bump whenever a code change would cut the same text differently (new splitter, parser fix...),
# so that already-ingested chunks are seen as stale even though no setting changed.
CHUNKER_VERSION = 1

BlockKind = Literal["prose", "code", "table"]
Splitters = list[tuple[re.Pattern[str], str]]

# How to split a block that is too big, most natural boundary first: (splitter, joiner).
# Code and tables split by line so we never break a row or reflow code; prose by sentence, then word.
SPLITTERS: dict[BlockKind, Splitters] = {
    "prose": [(SENTENCE_END_RE, " "), (WHITESPACE_RE, " ")],
    "code": [(NEWLINE_RE, "\n")],
    "table": [(NEWLINE_RE, "\n")],
}
BLOCK_JOINER = "\n\n"

TokenCounter = Callable[[str], int]


def approx_token_count(text: str) -> int:
    """Stand-in until we plug in the embedding model's real tokenizer.

    About 4 UTF-8 bytes per token: right for English, and an over-count for Sinhala
    (3 bytes per letter), so chunks err on the side of staying under the model's limit.
    """
    return math.ceil(len(text.encode("utf-8")) / 4)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    section: str
    chunk_index: int  # position in the document, for ordering and showing neighbours
    token_count: int
    doc_hash: str  # same for every chunk of a document; changes when any part of it changes


@dataclass(frozen=True)
class _Block:
    text: str
    kind: BlockKind


@dataclass(frozen=True)
class _Draft:
    headings: tuple[str, ...]
    text: str  # heading line + body, exactly what gets embedded
    section_index: int  # parts of the same section share this, so we never merge them back


def chunk_id(source: str, text: str, occurrence: int = 0) -> str:
    """An ID that depends only on what the chunk says, not where it sits in the document.

    Editing one paragraph changes only that chunk's ID, so ingestion can re-embed just that
    chunk instead of everything after it. The source is part of the key so the same text in
    two documents stays two chunks, and deleting one document never removes the other's.
    """
    return _short_hash(f"{source}\x00{occurrence}\x00{text}")


def document_hash(text: str) -> str:
    """Fingerprint of a whole (normalized) document, so ingestion can skip files that did not change."""
    return _short_hash(text)


def _short_hash(text: str) -> str:
    # 128 bits: collisions are not a practical concern even across millions of chunks.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def normalize_text(text: str) -> str:
    """NFC so the same Sinhala letter always has the same code points; unify line endings."""
    text = text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


class MarkdownChunker:
    def __init__(
        self,
        count_tokens: TokenCounter = approx_token_count,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        tokenizer_name: str | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not 0 <= overlap_tokens < max_tokens // 2:
            raise ValueError("overlap_tokens must be at least 0 and less than half of max_tokens")
        if not 0 <= min_tokens < max_tokens:
            raise ValueError("min_tokens must be at least 0 and less than max_tokens")
        self.count = count_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        # Lambdas and partials all look alike by name, so callers can pass a clearer one.
        self.tokenizer_name = tokenizer_name or (
            f"{getattr(count_tokens, '__module__', '?')}.{getattr(count_tokens, '__qualname__', '?')}"
        )

    @property
    def fingerprint(self) -> str:
        """Changes whenever this chunker would cut the same document differently.

        Ingestion stores it next to each document's doc_hash: if either one differs from what
        is stored, that document is re-chunked; if both match, it is skipped.
        """
        settings = (
            f"v{CHUNKER_VERSION}|{self.tokenizer_name}"
            f"|max={self.max_tokens}|overlap={self.overlap_tokens}|min={self.min_tokens}"
        )
        return _short_hash(settings)

    def chunk(self, text: str, source: str) -> list[Chunk]:
        text = normalize_text(text)
        doc_hash = document_hash(text)
        # Documents without headings (plain text, PDFs) fall back to the file name as the label.
        fallback = _title_from_filename(source)
        drafts = [
            draft
            for index, (headings, blocks) in enumerate(_parse_sections(text))
            for draft in self._section_drafts(headings, blocks, index, fallback)
        ]
        # How many times each text has appeared so far, so repeated chunks still get distinct IDs.
        seen: Counter[str] = Counter()
        chunks: list[Chunk] = []
        for index, draft in enumerate(self._merge_small(drafts)):
            chunks.append(
                Chunk(
                    id=chunk_id(source, draft.text, seen[draft.text]),
                    text=draft.text,
                    source=source,
                    section=" > ".join(draft.headings) or fallback,
                    chunk_index=index,
                    token_count=self.count(draft.text),
                    doc_hash=doc_hash,
                )
            )
            seen[draft.text] += 1
        return chunks

    def _section_drafts(
        self, headings: tuple[str, ...], blocks: list[_Block], index: int, fallback: str
    ) -> list[_Draft]:
        prefix = f"{self._heading_line(headings, fallback)}\n\n"
        # The heading line is embedded too, so it comes out of the budget.
        budget = max(1, self.max_tokens - self.count(prefix))
        # Leave room for the overlap that may be put in front of each piece.
        piece_limit = max(1, budget - self.overlap_tokens)
        pieces = [
            (piece, block_no, block.kind)
            for block_no, block in enumerate(blocks)
            for piece in self._split(block.text, piece_limit, SPLITTERS[block.kind])
        ]

        bodies: list[str] = []
        current = ""
        last_piece, last_block, last_kind = "", -1, "prose"
        for piece, block_no, kind in pieces:
            # Pieces cut from one block rejoin the way that block was split, not as new paragraphs.
            joiner = SPLITTERS[kind][0][1] if block_no == last_block else BLOCK_JOINER
            candidate = f"{current}{joiner}{piece}" if current else piece
            if current and self.count(candidate) > budget:
                bodies.append(current)
                seed = self._tail(last_piece, SPLITTERS[last_kind])
                seeded = f"{seed}{joiner}{piece}" if seed else piece
                current = seeded if self.count(seeded) <= budget else piece
            else:
                current = candidate
            last_piece, last_block, last_kind = piece, block_no, kind
        if current:
            bodies.append(current)
        return [_Draft(headings, f"{prefix}{body}", index) for body in bodies]

    def _merge_small(self, drafts: list[_Draft]) -> list[_Draft]:
        """Join a section that is too small to embed well with its neighbour, when they fit."""
        merged: list[_Draft] = []
        for draft in drafts:
            if merged:
                prev = merged[-1]
                small = min(self.count(prev.text), self.count(draft.text)) < self.min_tokens
                combined = f"{prev.text}{BLOCK_JOINER}{draft.text}"
                if (
                    small
                    and prev.section_index != draft.section_index
                    and self.count(combined) <= self.max_tokens
                ):
                    headings = _common_prefix(prev.headings, draft.headings)
                    merged[-1] = _Draft(headings, combined, draft.section_index)
                    continue
            merged.append(draft)
        return merged

    def _heading_line(self, headings: tuple[str, ...], fallback: str) -> str:
        # A very deep heading path would eat the budget: drop the outermost headings first.
        labels = list(headings)
        while len(labels) > 1 and self.count(" > ".join(labels)) > self.max_tokens // 4:
            labels.pop(0)
        return " > ".join(labels) or fallback

    def _split(self, content: str, limit: int, splitters: Splitters) -> list[str]:
        """Recursively split into parts of at most limit tokens, trying each splitter in turn.
        Only when no splitter is left is the text cut at a fixed width."""
        if self.count(content) <= limit:
            return [content]
        if not splitters:
            return self._hard_cut(content, limit)

        splitter, joiner = splitters[0]
        pieces = [
            small
            for piece in splitter.split(content)
            for small in self._split(piece, limit, splitters[1:])
        ]
        return self._pack(pieces, limit, joiner)

    def _pack(self, pieces: list[str], limit: int, joiner: str) -> list[str]:
        """Greedily join consecutive pieces while they fit within limit tokens."""
        parts: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current}{joiner}{piece}" if current else piece
            if current and self.count(candidate) > limit:
                parts.append(current)
                current = piece
            else:
                current = candidate
        if current:
            parts.append(current)
        return parts

    def _hard_cut(self, text: str, limit: int) -> list[str]:
        parts: list[str] = []
        while text:
            # Binary search for the longest prefix that fits.
            lo, hi = 1, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.count(text[:mid]) <= limit:
                    lo = mid
                else:
                    hi = mid - 1
            # Never separate a Sinhala vowel sign or virama from its letter.
            cut = lo
            while 1 < cut < len(text) and _is_combining(text[cut]):
                cut -= 1
            if cut == 1 and lo > 1:
                cut = lo  # one huge cluster: cutting it is the only option
            parts.append(text[:cut])
            text = text[cut:]
        return parts

    def _tail(self, text: str, splitters: Splitters) -> str:
        """The last whole sentences (or lines, or words) of text that fit in the overlap."""
        if not self.overlap_tokens:
            return ""
        for splitter, joiner in splitters + [(WHITESPACE_RE, " ")]:
            tail = ""
            for segment in reversed([s for s in splitter.split(text) if s]):
                candidate = f"{segment}{joiner}{tail}" if tail else segment
                if self.count(candidate) > self.overlap_tokens:
                    break
                tail = candidate
            if tail:
                return tail
        return ""


def _parse_sections(text: str) -> Iterator[tuple[tuple[str, ...], list[_Block]]]:
    """Yield (heading path, blocks) for every heading section that has content.
    Expects text that has already been through normalize_text."""
    headings: list[str] = []
    blocks: list[_Block] = []
    lines: list[str] = []
    fence: str | None = None  # the opening ``` or ~~~ while inside a code block

    def end_block(kind: BlockKind = "prose") -> None:
        content = "\n".join(lines) if kind == "code" else "\n".join(lines).strip()
        lines.clear()
        if content.strip():
            if kind == "prose" and content.startswith("|"):
                kind = "table"
            blocks.append(_Block(content, kind))

    for line in text.split("\n"):
        if fence:
            lines.append(line)
            if _closes_fence(line, fence):
                end_block("code")
                fence = None
            continue
        if fence_match := FENCE_RE.match(line):
            end_block()
            fence = fence_match.group(1)
            lines.append(line)
        elif heading_match := HEADING_RE.match(line):
            end_block()
            if blocks:
                yield tuple(headings), list(blocks)
                blocks.clear()
            level = len(heading_match.group(1))
            # A new heading replaces any headings at the same or deeper level.
            headings[level - 1 :] = [heading_match.group(2).strip().rstrip("#").strip()]
        elif not line.strip():
            end_block()
        else:
            lines.append(line)
    # An unclosed fence runs to the end of the document, as in CommonMark.
    end_block("code" if fence else "prose")
    if blocks:
        yield tuple(headings), list(blocks)


def _title_from_filename(source: str) -> str:
    return Path(source).stem.replace("_", " ").replace("-", " ")


def _closes_fence(line: str, fence: str) -> bool:
    # A closing fence uses the same character, at least as many times, and nothing after it.
    stripped = line.strip()
    return stripped.startswith(fence) and set(stripped) == {fence[0]}


def _is_combining(char: str) -> bool:
    # Marks (vowel signs, virama) and the zero-width joiner used in Sinhala conjuncts.
    return unicodedata.category(char).startswith("M") or char == "‍"


def _common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    prefix: list[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        prefix.append(x)
    return tuple(prefix)
