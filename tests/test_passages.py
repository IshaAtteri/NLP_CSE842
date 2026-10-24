import copy

import pytest

from src.hhsi.data.chunking import chunk_text, chunk_text_with_spans, normalize_text, tokenizer
from src.hhsi.data.passages import build_passages, compact, validate_document


def article(text="First sentence. Second sentence."):
    return {"body": text, "title": "Title", "url": "https://example.org/a",
            "source": "Example", "author": "Author", "category": "test", "published_at": "2026"}


def test_ids_metadata_and_determinism():
    corpus = [article(), {**article("Another article."), "url": "https://example.org/b"}]
    rows, report = build_passages(corpus)
    assert not report["failures"]
    assert report["processed_articles"] == 2
    assert rows == build_passages(corpus)[0]
    assert len({p["passage_id"] for p in rows}) == len(rows)
    assert rows[0]["passage_id"] == "doc_0000_chunk_0000"
    assert rows[1]["doc_id"] == "doc_0001"
    assert rows[0]["author"] == "Author"


@pytest.mark.parametrize("text", [
    "A sentence about science. " * 200,
    "word " * 800,
    "Hello!\r\n\r\n  Café naïve — punctuation.\tA new sentence.",
    "Repeated sentence. Repeated sentence. Repeated sentence.",
])
def test_coverage_and_token_limits(text):
    rows, report = build_passages([article(text)], max_tokens=40, overlap_tokens=8)
    assert not report["failures"]
    lengths, _ = validate_document(article(text), rows, 40)
    assert all(0 < n <= 40 for n in lengths)
    assert chunk_text(text, 40, 8) == [p["text"] for p in rows]
    assert rows[-1]["source_end"] == len(compact(normalize_text(text)))


def test_overlap_cannot_emit_only_old_content():
    text = "First short sentence. Tiny end. " + " ".join(["large"] * 28) + "."
    chunks = chunk_text_with_spans(text, max_tokens=32, overlap_tokens=10)
    assert all(b["source_end"] > a["source_end"] for a, b in zip(chunks, chunks[1:]))
    rows, report = build_passages([article(text)], 32, 10)
    assert not report["failures"]
    validate_document(article(text), rows, 32)


def test_intended_overlap_is_measured():
    rows, report = build_passages([article("One small sentence. " * 20)], 24, 8)
    assert not report["failures"]
    assert report["documents"][0]["intended_overlap_nonwhitespace_chars"] > 0


def test_tampered_content_and_mapping_rejected():
    rows, _ = build_passages([article()])
    bad = copy.deepcopy(rows)
    bad[0]["text"] = "Altered."
    with pytest.raises(ValueError):
        validate_document(article(), bad)
    bad = copy.deepcopy(rows)
    bad[0]["url"] = "wrong"
    with pytest.raises(ValueError):
        validate_document(article(), bad)


def test_invalid_inputs_are_explicit():
    with pytest.raises(ValueError):
        build_passages([])
    for value in ("", None):
        rows, report = build_passages([article(value)])
        assert rows == [] and len(report["failures"]) == 1
    _, report = build_passages([article(), article()])
    assert len(report["failures"]) == 1
    assert chunk_text("") == []
    with pytest.raises(TypeError):
        chunk_text(None)
    with pytest.raises(ValueError):
        chunk_text("text", max_tokens=2)
    with pytest.raises(ValueError):
        chunk_text("text", overlap_tokens=-1)
