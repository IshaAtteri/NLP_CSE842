"""Shared passage validation and JSON-serializable retrieval result schema."""
import json
from pathlib import Path
from typing import TypedDict


class RetrievalResult(TypedDict):
    passage_id: str
    doc_id: str
    url: str
    score: float
    rank: int


def validate_passages(passages):
    rows = [dict(p) for p in passages]
    if not rows:
        raise ValueError("At least one passage is required")
    ids, doc_urls, url_docs = set(), {}, {}
    for p in rows:
        for key in ("passage_id", "doc_id", "url", "text"):
            if not isinstance(p.get(key), str) or not p[key].strip():
                raise ValueError(f"Passage requires a nonempty string {key}")
        if p["passage_id"] in ids:
            raise ValueError(f"Duplicate passage ID: {p['passage_id']}")
        ids.add(p["passage_id"])
        if doc_urls.setdefault(p["doc_id"], p["url"]) != p["url"]:
            raise ValueError("A document maps to multiple URLs")
        if url_docs.setdefault(p["url"], p["doc_id"]) != p["doc_id"]:
            raise ValueError("A URL maps to multiple documents")
    return rows


def load_passages(path):
    # Blank lines or malformed JSON fail explicitly rather than skipping records.
    with Path(path).open(encoding="utf-8") as stream:
        return validate_passages([json.loads(line) for line in stream])


def make_result(passage, score, rank) -> RetrievalResult:
    return {"passage_id": passage["passage_id"], "doc_id": passage["doc_id"],
            "url": passage["url"], "score": float(score), "rank": rank}
