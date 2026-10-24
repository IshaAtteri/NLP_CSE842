import json

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from scripts.run_bm25 import run
from src.hhsi.retrieval.bm25 import BM25Retriever, lexical_tokens
from src.hhsi.retrieval.common import load_passages


def passages():
    return [{"passage_id": f"p{i}", "doc_id": f"d{i}", "url": f"https://example.org/{i}", "text": text}
            for i, text in enumerate(["apple fruit", "blue sky", "green grass", "red flower"])]


def test_lexical_preprocessing():
    assert lexical_tokens("CAFÉ, Apple_42 — Straße!") == ["café", "apple", "42", "strasse"]


def test_scores_ranking_and_shared_result():
    rows = passages()
    retriever = BM25Retriever(rows)
    results = retriever.search("APPLE", 4)
    expected = BM25Okapi([lexical_tokens(p["text"]) for p in rows]).get_scores(["apple"])
    assert results[0]["passage_id"] == "p0"
    assert results[0]["score"] == pytest.approx(expected[0])
    assert [r["rank"] for r in results] == [1, 2, 3, 4]
    assert all(a["score"] >= b["score"] for a, b in zip(results, results[1:]))
    assert set(results[0]) == {"passage_id", "doc_id", "url", "score", "rank"}
    assert results[0]["url"] == rows[0]["url"]
    index = retriever.index
    retriever.search("sky")
    assert retriever.index is index


def test_ties_are_input_order_independent_and_oov_is_retained():
    a = BM25Retriever(passages()).search("absentword", 99)
    b = BM25Retriever(list(reversed(passages()))).search("absentword", 99)
    assert a == b
    assert [r["passage_id"] for r in a] == ["p0", "p1", "p2", "p3"]
    assert all(r["score"] == 0 for r in a)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_bad_k(top_k):
    with pytest.raises(ValueError):
        BM25Retriever(passages()).search("apple", top_k)


@pytest.mark.parametrize("query", ["", "   ", "!!!"])
def test_empty_query(query):
    with pytest.raises(ValueError):
        BM25Retriever(passages()).search(query)


def test_invalid_corpus_and_parameters():
    for rows in ([], [passages()[0]] * 2, [{**passages()[0], "text": ""}],
                 [{**passages()[0], "text": "!!!"}],
                 [passages()[0], {**passages()[1], "doc_id": "d0"}],
                 [passages()[0], {**passages()[1], "url": passages()[0]["url"]}]):
        with pytest.raises(ValueError):
            BM25Retriever(rows)
    for kwargs in ({"k1": 0}, {"b": 2}, {"epsilon": -1}, {"b": np.nan}):
        with pytest.raises(ValueError):
            BM25Retriever(passages(), **kwargs)
    with pytest.raises(TypeError):
        BM25Retriever(passages()).search(None)


def test_saved_run_preserves_all_queries_and_artifacts(tmp_path):
    p = tmp_path / "passages.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in passages()), encoding="utf-8")
    q = tmp_path / "questions.json"
    questions = [
        {"query": "apple", "question_type": "multi", "evidence_list":
         [{"url": passages()[0]["url"]}, {"url": passages()[1]["url"]}]},
        {"query": "unknown", "question_type": "null_query", "evidence_list": []},
    ]
    q.write_text(json.dumps(questions), encoding="utf-8")
    output = tmp_path / "run"
    summary = run(p, q, output, top_k=3, expected_questions=2)
    records = [json.loads(line) for line in (output / "retrieval.jsonl").read_text().splitlines()]
    assert summary["retrieved_questions"] == 2
    assert len(records) == 2 and all(len(r["results"]) == 3 for r in records)
    assert records[0]["evidence_list"] == questions[0]["evidence_list"]
    assert records[1]["question_id"] == "q_0001"
    assert load_passages(p) == passages()
    original = (output / "retrieval.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        run(p, q, output, expected_questions=2)
    assert (output / "retrieval.jsonl").read_bytes() == original
    with pytest.raises(ValueError, match="Expected"):
        run(p, q, tmp_path / "wrong_count", expected_questions=3)
    questions[0]["evidence_list"] = [{"url": "missing"}]
    q.write_text(json.dumps(questions), encoding="utf-8")
    with pytest.raises(ValueError, match="Evidence URLs"):
        run(p, q, tmp_path / "missing", expected_questions=2)


def test_blank_passage_line_is_not_silently_skipped(tmp_path):
    p = tmp_path / "invalid.jsonl"
    p.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_passages(p)
