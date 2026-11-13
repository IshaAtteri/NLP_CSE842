"""Phase 3 tests: no downloads or pretrained BGE model required."""
import copy
import importlib.util
import json

import numpy as np
import pytest
import spacy

from scripts.extract_entities import run
from src.hhsi.indexing.entities import (
    extract_entities, load_entity_records, normalize_entity, validate_entity_records,
)
from src.hhsi.indexing.similarity import (
    PassageSimilarity, entity_similarity, hybrid_distance, hybrid_similarity,
    load_passage_embeddings, semantic_similarity,
)
from src.hhsi.retrieval.dense import DIMENSION, MODEL_NAME, file_hash, json_hash, save_cache


def passages():
    return [
        {"passage_id": "p2", "doc_id": "d2", "url": "url2",
         "text": "OpenAI met OPENAI in New York on Monday."},
        {"passage_id": "p0", "doc_id": "d0", "url": "url0",
         "text": "Nothing named here."},
    ]


def rule_nlp():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([
        {"label": "ORG", "pattern": "OpenAI"}, {"label": "PRODUCT", "pattern": "OPENAI"},
        {"label": "GPE", "pattern": "New York"}, {"label": "DATE", "pattern": "Monday"},
    ])
    return nlp


@pytest.mark.parametrize("text,expected", [
    (" OpenAI ", "openai"), ("NEW YORK", "new york"),
    ("New   York", "new york"), ("“New\tYork!”", "new york"),
    ("ＯｐｅｎＡＩ", "openai"), ("Straße", "strasse"),
    ("Cafe\u0301", "café"), ("O’Reilly", "o’reilly"),
    ("AT&T", "at&t"), ("C++", "c++"), (" --- ", ""), ("", ""),
])
def test_normalization(text, expected):
    assert normalize_entity(text) == expected
    assert normalize_entity(expected) == expected


def test_extraction_deduplicates_preserves_mentions_and_order():
    rows = passages()
    before = copy.deepcopy(rows)
    records, report = extract_entities(rows, nlp=rule_nlp(), batch_size=1)
    assert rows == before
    assert [r["passage_id"] for r in records] == ["p2", "p0"]
    assert records[1]["entities"] == []
    assert [e["normalized"] for e in records[0]["entities"]] == ["openai", "new york"]
    assert records[0]["entities"][0]["mentions"] == [
        {"text": "OpenAI", "label": "ORG"}, {"text": "OPENAI", "label": "PRODUCT"}]
    assert report["total_detected_mentions_all_labels"] == 4
    assert report["total_extracted_entity_mentions"] == 3
    assert report["unique_normalized_entities"] == 2
    assert report["passages_with_no_entities"] == 1
    assert report["mean_unique_entities_per_passage"] == 1
    assert report["median_unique_entities_per_passage"] == 1
    assert report["entity_counts_by_type"] == {"GPE": 1, "ORG": 1, "PRODUCT": 1}
    selected, _ = extract_entities(rows, nlp=rule_nlp(), labels=["DATE"])
    assert selected[0]["entities"][0]["normalized"] == "monday"


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_invalid_batch(batch):
    with pytest.raises(ValueError):
        extract_entities(passages(), nlp=rule_nlp(), batch_size=batch)


@pytest.mark.parametrize("mutation", ["order", "missing", "doc_id", "duplicate", "normalized"])
def test_record_alignment(mutation):
    rows = passages()
    records, _ = extract_entities(rows, nlp=rule_nlp())
    if mutation == "order":
        records.reverse()
    elif mutation == "missing":
        records.pop()
    elif mutation == "doc_id":
        records[0]["doc_id"] = "wrong"
    elif mutation == "duplicate":
        records[0]["entities"].append(records[0]["entities"][0])
    else:
        records[0]["entities"][0]["normalized"] = "wrong"
    with pytest.raises(ValueError):
        validate_entity_records(records, rows)


def test_jsonl_report_and_no_overwrite(tmp_path):
    source, output = tmp_path / "passages.jsonl", tmp_path / "entities.jsonl"
    source.write_text("".join(json.dumps(p) + "\n" for p in passages()), encoding="utf-8")
    report = run(source, output, nlp=rule_nlp(), expected_passages=2)
    assert len(load_entity_records(output, passages())) == 2
    assert report["input_passage_dataset_sha256"] == file_hash(source)
    assert report["entity_file_sha256"] == file_hash(output)
    assert json.loads(output.with_suffix(".report.json").read_text()) == report
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        run(source, output, nlp=rule_nlp(), expected_passages=2)
    assert output.read_bytes() == before
    with pytest.raises(ValueError):
        run(source, tmp_path / "bad.jsonl", nlp=rule_nlp(), expected_passages=3)
    assert not (tmp_path / "bad.jsonl").exists()


@pytest.mark.parametrize("a,b,expected", [
    ([], [], 0), (["a"], [], 0), (["a"], ["a"], 1),
    (["openai", "microsoft"], ["openai", "google"], 1 / 3),
    (["a", "a"], ["a"], 1), (["a"], ["b"], 0),
])
def test_jaccard(a, b, expected):
    assert entity_similarity(a, b) == expected
    assert entity_similarity(b, a) == expected


@pytest.mark.parametrize("a,b,expected", [
    ([1, 0], [1, 0], 1), ([1, 0], [0, 1], 0),
    ([1, 0], [-1, 0], -1), ([0, 0], [1, 0], 0),
    ([0, 0], [0, 0], 0), ([3, 4], [6, 8], 1),
    ([1e308, 0], [1e308, 0], 1),
])
def test_cosine(a, b, expected):
    assert semantic_similarity(a, b) == pytest.approx(expected)
    assert semantic_similarity(b, a) == pytest.approx(expected)


@pytest.mark.parametrize("a,b", [([], []), ([1], [1, 2]), ([float("nan")], [1]),
                                ([float("inf")], [1]), ([[1]], [[1]])])
def test_invalid_cosine(a, b):
    with pytest.raises(ValueError):
        semantic_similarity(a, b)


@pytest.mark.parametrize("alpha", [0, 0.5, 0.7, 1])
def test_hybrid(alpha):
    args = ([1, 0], [-1, 0], {"a", "b"}, {"a", "c"})
    expected = -alpha + (1 - alpha) / 3
    assert hybrid_similarity(*args, alpha=alpha) == pytest.approx(expected)
    assert hybrid_similarity(args[1], args[0], args[3], args[2], alpha=alpha) == pytest.approx(expected)
    assert hybrid_distance(*args, alpha=alpha) == pytest.approx(1 - expected)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf"), True, "0.7", None])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        hybrid_similarity([1], [1], [], [], alpha)


def cache_fixture(tmp_path):
    rows = passages()
    vectors = np.zeros((2, DIMENSION), dtype=np.float32)
    vectors[0, 0], vectors[1, 0] = 1, -1
    cache = tmp_path / "vectors.npy"
    metadata = {
        "schema_version": 1, "passage_count": 2, "embedding_dimension": DIMENSION,
        "passage_ids": [p["passage_id"] for p in rows],
        "passage_records_sha256": json_hash(rows), "passage_file_sha256": "inputhash",
        "encoder": {"model_name": MODEL_NAME, "dimension": DIMENSION,
                    "normalization": "L2", "dtype": "float32",
                    "pooling": "last_hidden_state[:, 0, :] (CLS)", "passage_instruction": ""},
    }
    save_cache(cache, vectors, metadata)
    return rows, cache


def test_id_pairs_batch_and_cache(tmp_path):
    rows, cache = cache_fixture(tmp_path)
    records, _ = extract_entities(rows, nlp=rule_nlp())
    api = PassageSimilarity(rows, records, cache, dataset_sha256="inputhash")
    pairs = api.pairs([("p2", "p0"), ("p0", "p2")], alpha=0.5)
    assert pairs[0]["semantic"] == -1
    assert pairs[0]["hybrid"] == pairs[1]["hybrid"] == -0.5
    assert api.pairs([]) == []
    with pytest.raises(KeyError):
        api.pair("missing", "p0")
    with pytest.raises(ValueError):
        load_passage_embeddings(cache, rows[::-1], dataset_sha256="inputhash")
    with pytest.raises(ValueError):
        load_passage_embeddings(cache, rows, dataset_sha256="wrong")


@pytest.mark.parametrize("mutation", ["checksum", "encoder", "dimension", "zero", "missing_metadata"])
def test_incompatible_cache(tmp_path, mutation):
    rows, cache = cache_fixture(tmp_path)
    meta_path = cache.with_name("vectors_metadata.json")
    metadata = json.loads(meta_path.read_text())
    if mutation == "missing_metadata":
        meta_path.unlink()
    elif mutation == "checksum":
        with cache.open("ab") as stream:
            stream.write(b"corrupt")
    elif mutation == "encoder":
        metadata["encoder"]["pooling"] = "mean"
        meta_path.write_text(json.dumps(metadata))
    else:
        vectors = np.zeros((2, 2 if mutation == "dimension" else DIMENSION), dtype=np.float32)
        np.save(cache, vectors)
        metadata["embedding_file_sha256"] = file_hash(cache)
        meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_passage_embeddings(cache, rows, dataset_sha256="inputhash")


def test_missing_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_passage_embeddings(tmp_path / "absent.npy", passages(), dataset_sha256="hash")


@pytest.mark.skipif(importlib.util.find_spec("en_core_web_sm") is None, reason="NER model not installed")
def test_pretrained_ner_representative_text():
    rows = [{"passage_id": "p", "doc_id": "d", "url": "u",
             "text": "Barack Obama was born in Hawaii. Apple is based in California."}]
    records, _ = extract_entities(rows)
    keys = {e["normalized"] for e in records[0]["entities"]}
    assert {"barack obama", "hawaii"} <= keys
