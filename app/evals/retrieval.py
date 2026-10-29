"""The retrieval eval set: customer questions, each with the text that a correct answer chunk contains."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from app.rag.chunking import normalize_text

Language = Literal["en", "si", "singlish"]
DEFAULT_CASES_PATH = Path("evals/retrieval_cases.jsonl")


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    lang: Language
    query: str
    # Snippets of the expected answer. We match on text, not chunk IDs, because IDs change
    # whenever the docs or chunk settings change, and the eval set must survive that.
    relevant: tuple[str, ...]

    def is_relevant(self, chunk_text: str) -> bool:
        """A retrieved chunk counts as a hit if it contains any of the relevant snippets."""
        text = normalize_text(chunk_text)
        return any(snippet in text for snippet in self.relevant)


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[RetrievalCase]:
    """Read one JSON object per line. A bad line fails loudly, with its line number, because a
    silently skipped case would make the benchmark look better or worse than it really is."""
    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                case = _parse_case(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_no}: {error}") from error
            if case.id in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate id {case.id!r}")
            seen_ids.add(case.id)
            cases.append(case)
    return cases


def _parse_case(raw: dict) -> RetrievalCase:
    if raw["lang"] not in get_args(Language):
        raise ValueError(f"unknown lang {raw['lang']!r}")
    relevant = tuple(normalize_text(snippet) for snippet in raw["relevant"])
    if not relevant or not all(snippet.strip() for snippet in relevant):
        raise ValueError("relevant must list at least one non-empty snippet")
    return RetrievalCase(
        id=raw["id"], lang=raw["lang"], query=normalize_text(raw["query"]), relevant=relevant
    )
