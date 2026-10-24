"""Offline dense tests; opt-in real BGE integration via HHSI_TEST_REAL_BGE=1."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_dense import run, validate_saved_run
from src.hhsi.retrieval.dense import (
    BGEEncoder, DenseRetriever, DIMENSION, QUERY_INSTRUCTION,
    cache_metadata_path, file_hash, load_cache, validate_embeddings,
)


def passages():
    return [
        {"passage_id": name, "doc_id": "d" + name, "url": "https://example.org/" + name,
         "text": text}
        for name, text in [("p2", "Paris is in France."), ("p0", "A cat is an animal."),
                           ("p1", "Rome is in Italy.")]
    ]


def vectors():
    result = np.zeros((3, DIMENSION), dtype=np.float32)
    result[0, 0] = result[2, 0] = 1
    result[1, 1] = 1
    return result


class FakeEncoder:
    def __init__(self):
        self.signature = {"resolved_revision": "a" * 40, "dimension": DIMENSION,
                          "query_instruction": QUERY_INSTRUCTION, "pooling": "CLS"}
        self.inference_settings = {"device": "cpu"}
        self.last_encoding_stats = {}
        self.passage_calls = 0
        self.query_calls = []

    def encode_passages(self, texts, batch_size=16):
        self.passage_calls += 1
        self.last_encoding_stats = {"count": len(texts), "seconds": 0.0, "batch_size": batch_size}
        # The synthetic vectors match passage order.
        return vectors()[:len(texts)].copy()

    def encode_queries(self, texts, batch_size=32):
        self.query_calls.append((list(texts), batch_size))
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Invalid queries")
        self.last_encoding_stats = {"count": len(texts), "seconds": 0.0, "batch_size": batch_size}
        return np.repeat(vectors()[0:1], len(texts), axis=0)


def test_dimensions_normalization_ranking_ties_and_mapping():
    rows = passages()
    retriever = DenseRetriever(rows, encoder=FakeEncoder())
    assert [p["passage_id"] for p in retriever.passages] == ["p2", "p0", "p1"]
    assert retriever.embeddings.shape == (3, 768)
    np.testing.assert_allclose(np.linalg.norm(retriever.embeddings, axis=1), 1)
    results = retriever.search_embedding(vectors()[0], top_k=3)
    assert [r["passage_id"] for r in results] == ["p1", "p2", "p0"]
    assert [r["score"] for r in results] == [1.0, 1.0, 0.0]
    assert [r["rank"] for r in results] == [1, 2, 3]
    assert set(results[0]) == {"passage_id", "doc_id", "url", "score", "rank"}
    lookup = {p["passage_id"]: p for p in rows}
    assert all(r["url"] == lookup[r["passage_id"]]["url"] for r in results)
    assert len(retriever.search("Where is Paris?", 99)) == 3
    assert len(retriever.search("Where is Paris?", 1)) == 1


def test_batched_queries_reuse_one_encoder():
    encoder = FakeEncoder()
    retriever = DenseRetriever(passages(), encoder=encoder)
    results = retriever.search_many(["Question one", "Question two"], top_k=2, batch_size=2)
    assert len(results) == 2
    assert encoder.passage_calls == 1
    assert encoder.query_calls == [(["Question one", "Question two"], 2)]


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(top_k):
    retriever = DenseRetriever(passages(), encoder=FakeEncoder())
    with pytest.raises(ValueError):
        retriever.search("Query", top_k)


@pytest.mark.parametrize("kind", ["wrong_dimension", "zero", "nan", "unnormalized", "float64"])
def test_invalid_embeddings(kind):
    value = vectors().copy()
    if kind == "wrong_dimension":
        value = value[:, :2]
    elif kind == "zero":
        value[:] = 0
    elif kind == "nan":
        value[0, 0] = np.nan
    elif kind == "unnormalized":
        value *= 2
    else:
        value = value.astype(np.float64)
    with pytest.raises(ValueError):
        validate_embeddings(value, 3)


def test_invalid_passages_and_queries():
    for rows in ([], [passages()[0]] * 2, [{**passages()[0], "text": ""}]):
        with pytest.raises(ValueError):
            DenseRetriever(rows, encoder=FakeEncoder())
    retriever = DenseRetriever(passages(), encoder=FakeEncoder())
    for query in ("", None):
        with pytest.raises(ValueError):
            retriever.search(query)


def test_cache_roundtrip_avoids_passage_forward_passes(tmp_path):
    path = tmp_path / "vectors.npy"
    first = DenseRetriever(passages(), encoder=FakeEncoder(), cache_path=path,
                           dataset_sha256="input-hash")
    encoder = FakeEncoder()
    second = DenseRetriever(passages(), encoder=encoder, cache_path=path,
                            dataset_sha256="input-hash")
    assert not first.cache_hit and second.cache_hit
    assert encoder.passage_calls == 0
    assert second.passage_encoding_seconds == 0
    np.testing.assert_array_equal(first.embeddings, second.embeddings)
    assert first.search("Query") == second.search("Query")
    metadata = json.loads(cache_metadata_path(path).read_text())
    assert metadata["passage_ids"] == ["p2", "p0", "p1"]
    assert metadata["embedding_file_sha256"] == file_hash(path)


@pytest.mark.parametrize("change", ["text", "order", "file_hash", "revision", "instruction", "pooling"])
def test_cache_mismatch_rejected_without_overwriting(tmp_path, change):
    path = tmp_path / "vectors.npy"
    DenseRetriever(passages(), encoder=FakeEncoder(), cache_path=path, dataset_sha256="original")
    original = path.read_bytes()
    rows = copy.deepcopy(passages())
    encoder = FakeEncoder()
    digest = "original"
    if change == "text":
        rows[0]["text"] = "Different source content."
    elif change == "order":
        rows.reverse()
    elif change == "file_hash":
        digest = "changed"
    elif change == "revision":
        encoder.signature["resolved_revision"] = "b" * 40
    elif change == "instruction":
        encoder.signature["query_instruction"] = "Different instruction: "
    else:
        encoder.signature["pooling"] = "mean"
    with pytest.raises(ValueError, match="Incompatible"):
        DenseRetriever(rows, encoder=encoder, cache_path=path, dataset_sha256=digest)
    assert path.read_bytes() == original
    assert encoder.passage_calls == 0


def test_corrupt_and_incomplete_cache(tmp_path):
    path = tmp_path / "vectors.npy"
    retriever = DenseRetriever(passages(), encoder=FakeEncoder(), cache_path=path)
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_cache(path, retriever.cache_expected)
    incomplete = tmp_path / "incomplete.npy"
    incomplete.write_bytes(b"not a complete cache")
    with pytest.raises(ValueError, match="Incomplete"):
        load_cache(incomplete, retriever.cache_expected)


def test_unit_cache_content_validation(tmp_path):
    path = tmp_path / "vectors.npy"
    retriever = DenseRetriever(passages(), encoder=FakeEncoder(), cache_path=path)
    # Even a matching file checksum cannot make non-unit vectors valid.
    with path.open("wb") as stream:
        np.save(stream, vectors() * 2)
    meta_path = cache_metadata_path(path)
    metadata = json.loads(meta_path.read_text())
    metadata["embedding_file_sha256"] = file_hash(path)
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="normalized"):
        load_cache(path, retriever.cache_expected)


def test_encoder_cpu_pooling_instruction_batching_and_limits():
    import torch

    class Tokenizer:
        def __init__(self):
            self.seen = []

        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            self.seen = texts
            ids = [[101] + [5] * len(text.split()) + [102] for text in texts]
            return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

        def pad(self, features, **kwargs):
            length = max(len(f["input_ids"]) for f in features)
            return {k: torch.tensor([f[k] + [0] * (length - len(f[k])) for f in features])
                    for k in ("input_ids", "attention_mask")}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids, attention_mask, **kwargs):
            assert torch.is_inference_mode_enabled()
            assert not self.training
            assert input_ids.device.type == "cpu"
            self.calls += 1
            hidden = torch.zeros((*input_ids.shape, DIMENSION))
            hidden[:, 0, 0] = 3
            hidden[:, 0, 1] = 4
            hidden[:, 1:, 2] = 100  # Mean pooling would give a different answer.
            return SimpleNamespace(last_hidden_state=hidden)

    encoder = BGEEncoder.__new__(BGEEncoder)
    encoder.tokenizer = Tokenizer()
    encoder.model = Model().eval()
    encoder.device = torch.device("cpu")
    encoder.passage_max_tokens = 256
    encoder.query_max_tokens = 512
    encoder.verbose = False
    output = encoder.encode_queries(["one", "two words", "three"], batch_size=2)
    assert encoder.model.calls == 2
    assert encoder.tokenizer.seen[0] == QUERY_INSTRUCTION + "one"
    assert output.shape == (3, DIMENSION) and output.dtype == np.float32
    np.testing.assert_allclose(output[:, :2], [[0.6, 0.8]] * 3)
    assert not output[:, 2:].any()
    encoder.encode_passages(["original text"])
    assert encoder.tokenizer.seen == ["original text"]
    calls = encoder.model.calls
    with pytest.raises(ValueError, match="No text truncated"):
        encoder.encode_passages(["word " * 255])
    with pytest.raises(ValueError, match="No text truncated"):
        encoder.encode_queries(["word " * 505])
    assert encoder.model.calls == calls
    for invalid in ([], [""], [None], "one string"):
        with pytest.raises(ValueError):
            encoder.encode_passages(invalid)
    with pytest.raises(ValueError):
        encoder.encode_queries(["text"], batch_size=0)


def test_saved_runner_output_and_audit(tmp_path):
    p = tmp_path / "passages.jsonl"
    p.write_text("".join(json.dumps(row) + "\n" for row in passages()))
    questions = [{"query": "Where is Paris?", "question_type": "test",
                  "evidence_list": [{"url": passages()[0]["url"]}, {"url": passages()[2]["url"]}]},
                 {"query": "Unknown question?", "question_type": "null_query", "evidence_list": []}]
    q = tmp_path / "questions.json"
    q.write_text(json.dumps(questions))
    output = tmp_path / "run"
    encoder = FakeEncoder()
    summary = run(p, q, output, expected_questions=2, expected_passages=3, encoder=encoder,
                  top_k=2, cache_path=tmp_path / "cache.npy", query_batch_size=2)
    records = [json.loads(line) for line in (output / "retrieval.jsonl").read_text().splitlines()]
    assert summary["output_validation"]["ranked_passage_results"] == 4
    assert summary["retrieved_questions"] == 2 and summary["evaluated_questions"] == 0
    assert records[1]["question_id"] == "q_0001"
    assert records[0]["evidence_list"] == questions[0]["evidence_list"]
    assert set(records[0]) == {"question_id", "query", "question_type", "evidence_list",
                               "latency_seconds", "results"}
    assert encoder.query_calls == [([q["query"] for q in questions], 2)]
    with pytest.raises(FileExistsError):
        run(p, q, output, encoder=encoder)
    with pytest.raises(ValueError, match="Expected"):
        run(p, q, tmp_path / "count", expected_passages=4, encoder=encoder)
    questions[0]["evidence_list"] = [{"url": "missing"}]
    q.write_text(json.dumps(questions))
    with pytest.raises(ValueError, match="Evidence URLs"):
        run(p, q, tmp_path / "missing", expected_questions=2, expected_passages=3, encoder=encoder)
    records[0]["results"][0]["url"] = "wrong"
    (output / "retrieval.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    questions[0]["evidence_list"] = records[0]["evidence_list"]
    with pytest.raises(ValueError, match="mapping"):
        validate_saved_run(output / "retrieval.jsonl", passages(), questions, 2)


@pytest.mark.skipif(os.environ.get("HHSI_TEST_REAL_BGE") != "1",
                    reason="Opt-in actual BGE download/inference")
def test_real_bge_cpu_integration(tmp_path):
    encoder = BGEEncoder(device="cpu", cpu_threads=4)
    retriever = DenseRetriever(passages(), encoder=encoder, cache_path=tmp_path / "real.npy",
                               batch_size=2)
    assert retriever.embeddings.shape == (3, 768)
    np.testing.assert_allclose(np.linalg.norm(retriever.embeddings, axis=1), 1, atol=1e-5)
    results = retriever.search_many(["Which country contains Paris?", "What kind of creature is a cat?"],
                                  top_k=3, batch_size=2)
    assert results[0][0]["passage_id"] == "p2"
    assert results[1][0]["passage_id"] == "p0"
    # Padded batch and singleton encoding should agree up to float32 arithmetic.
    batch = encoder.encode_queries(["Where is Paris?", "A substantially longer query about an animal."])
    single = encoder.encode_queries(["Where is Paris?"], batch_size=1)
    np.testing.assert_allclose(batch[0], single[0], atol=2e-6, rtol=2e-5)
    cached = DenseRetriever(passages(), encoder=encoder, cache_path=tmp_path / "real.npy")
    assert cached.cache_hit
    assert cached.search("Which country contains Paris?", 3) == retriever.search(
        "Which country contains Paris?", 3)



def test_query_embedding_shape_and_unavailable_cuda(monkeypatch):
    import torch
    retriever = DenseRetriever(passages(), encoder=FakeEncoder())
    with pytest.raises(ValueError, match="one-dimensional"):
        retriever.search_embedding(vectors()[0].reshape(-1, 1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested"):
        BGEEncoder(device="cuda")
