"""Passage construction and exact non-whitespace source-coverage validation."""
from collections import Counter
import hashlib
import json

import numpy as np

from .chunking import chunk_text_with_spans, normalize_text, tokenizer


def compact(text):
    return "".join(text.split())


def validate_document(article, passages, max_tokens=256):
    source = compact(normalize_text(article["body"]))
    if not source or not passages:
        raise ValueError("Empty document or missing passages")
    covered = 0
    lengths = []
    overlap_chars = 0
    for index, passage in enumerate(passages):
        start, end = passage["source_start"], passage["source_end"]
        if not 0 <= start <= covered < end <= len(source):
            raise ValueError("Source gap, invalid span, or overlap-only passage")
        if compact(passage["text"]) != source[start:end]:
            raise ValueError("Passage does not match its source span")
        if passage["chunk_id"] != index:
            raise ValueError("Nonsequential chunk IDs")
        if any(passage[key] != article[key] for key in ("title", "url", "source")):
            raise ValueError("Source metadata mismatch")
        length = len(tokenizer.encode(passage["text"], verbose=False))
        if not passage["text"].strip() or length > max_tokens:
            raise ValueError("Empty or oversized passage")
        lengths.append(length)
        overlap_chars += covered - start
        covered = end
    if covered != len(source):
        raise ValueError("Missing source suffix")
    return lengths, overlap_chars


def build_passages(corpus, max_tokens=256, overlap_tokens=32):
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Corpus must be a nonempty list")
    passages, failures, lengths, documents = [], [], [], []
    urls = set()
    for index, article in enumerate(corpus):
        doc_id = f"doc_{index:04d}"
        try:
            for key in ("body", "title", "url", "source"):
                if not isinstance(article.get(key), str) or not article[key].strip():
                    raise ValueError(f"Missing or empty {key}")
            if article["url"] in urls:
                raise ValueError("Duplicate article URL")
            urls.add(article["url"])
            chunks = chunk_text_with_spans(article["body"], max_tokens, overlap_tokens)
            rows = [
                {
                    **{key: value for key, value in article.items() if key != "body"},
                    **chunk,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "passage_id": f"{doc_id}_chunk_{chunk_id:04d}",
                }
                for chunk_id, chunk in enumerate(chunks)
            ]
            doc_lengths, overlap_chars = validate_document(article, rows, max_tokens)
            lengths.extend(doc_lengths)
            passages.extend(rows)
            documents.append({
                "doc_id": doc_id, "url": article["url"], "passage_count": len(rows),
                "source_nonwhitespace_chars": len(compact(article["body"])),
                "intended_overlap_nonwhitespace_chars": overlap_chars,
                "uncovered_nonwhitespace_chars": 0,
                "overlap_only_passages": 0,
            })
        except (ValueError, TypeError, KeyError) as error:
            failures.append({"doc_id": doc_id, "error": str(error)})
    if len({p["passage_id"] for p in passages}) != len(passages):
        raise ValueError("Duplicate passage IDs")
    counts = Counter(p["text"] for p in passages)
    report = {
        "input_articles": len(corpus), "processed_articles": len(documents),
        "passage_count": len(passages), "failures": failures,
        "documents": documents,
        "identical_text_extra_occurrences": sum(n - 1 for n in counts.values()),
        "duplicate_text_policy": "Retained: source spans distinguish repeated source text from overlap.",
        "coverage_coordinate": "Half-open offsets in body with all whitespace removed; case and punctuation preserved.",
        "length_tokens_including_special_tokens": {
            "min": min(lengths), "max": max(lengths),
            "mean": float(np.mean(lengths)),
            **{f"p{p}": float(np.percentile(lengths, p)) for p in (25, 50, 75, 90, 95, 99)},
        } if lengths else {},
    }
    return passages, report


def tokenizer_fingerprint():
    payload = json.dumps(tokenizer.backend_tokenizer.to_str(), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
