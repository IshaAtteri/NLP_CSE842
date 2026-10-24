"""Run flat BGE retrieval on all MultiHop-RAG questions, without relevance evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import uuid

import numpy as np

from src.hhsi.retrieval.common import load_passages
from src.hhsi.retrieval.dense import (
    BGEEncoder, DenseRetriever, DIMENSION, cache_metadata_path, file_hash, positive_int,
)

DEFAULT_CACHE = Path("data/processed/embeddings/multihoprag_bge_base_v1_5.npy")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def load_questions(path, passages, expected_questions):
    questions = json.loads(Path(path).read_bytes())
    if not isinstance(questions, list) or not questions:
        raise ValueError("Questions must be a nonempty list")
    if len(questions) != expected_questions:
        raise ValueError(f"Expected {expected_questions} questions, got {len(questions)}")
    urls = {p["url"] for p in passages}
    failures = []
    for i, question in enumerate(questions):
        query = question.get("query")
        if not isinstance(query, str) or not any(c.isalnum() for c in query):
            raise ValueError(f"Invalid query at question {i}")
        evidence_list = question.get("evidence_list")
        if not isinstance(evidence_list, list):
            raise ValueError(f"Invalid evidence list at question {i}")
        for evidence in evidence_list:
            if not isinstance(evidence, dict) or not isinstance(evidence.get("url"), str) or not evidence["url"]:
                raise ValueError(f"Missing evidence URL at question {i}")
            if evidence["url"] not in urls:
                failures.append({"question_id": f"q_{i:04d}", "url": evidence["url"]})
    if failures:
        raise ValueError(f"Evidence URLs missing from passages: {failures}")
    return questions


def validate_saved_run(path, passages, questions, top_k):
    """Audit all serialized results without computing relevance metrics."""
    lookup = {p["passage_id"]: p for p in passages}
    expected_depth = min(top_k, len(passages))
    count = entries = 0
    with Path(path).open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            row = json.loads(line)
            if i >= len(questions):
                raise ValueError("Too many saved question records")
            question = questions[i]
            if (row["question_id"] != f"q_{i:04d}" or row["query"] != question["query"]
                    or row["question_type"] != question.get("question_type")
                    or row["evidence_list"] != question["evidence_list"]):
                raise ValueError(f"Saved question order or annotations differ at {i}")
            results = row["results"]
            if len(results) != expected_depth or len({r["passage_id"] for r in results}) != expected_depth:
                raise ValueError(f"Wrong result depth or duplicate passages at {i}")
            if results != sorted(results, key=lambda r: (-r["score"], r["passage_id"])):
                raise ValueError(f"Incorrect score/tie ordering at {i}")
            if not np.isfinite(row["latency_seconds"]) or row["latency_seconds"] < 0:
                raise ValueError("Invalid saved latency")
            for rank, result in enumerate(results, 1):
                passage = lookup.get(result["passage_id"])
                if passage is None or any(result[k] != passage[k] for k in ("doc_id", "url")):
                    raise ValueError(f"Unknown passage or invalid mapping at {i}")
                if (result["rank"] != rank or not np.isfinite(result["score"])
                        or abs(result["score"]) > 1.00001):
                    raise ValueError(f"Invalid rank or cosine score at {i}")
            count += 1
            entries += len(results)
    if count != len(questions):
        raise ValueError(f"Only {count}/{len(questions)} question records saved")
    return {"question_records": count, "ranked_passage_results": entries, "status": "passed"}


def run(passages_path, questions_path, output_dir, *, top_k=50, cache_path=DEFAULT_CACHE,
        batch_size=16, query_batch_size=32, device="auto", cpu_threads=4, revision="main",
        query_max_tokens=512, expected_questions=2556, expected_passages=6638, encoder=None):
    start = time.perf_counter()
    for value, name in ((top_k, "top_k"), (batch_size, "batch_size"),
                        (query_batch_size, "query_batch_size")):
        positive_int(value, name)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    passages = load_passages(passages_path)
    if len(passages) != expected_passages:
        raise ValueError(f"Expected {expected_passages} passages, got {len(passages)}")
    questions = load_questions(questions_path, passages, expected_questions)
    input_seconds = time.perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=False)
    load_start = time.perf_counter()
    encoder = encoder if encoder is not None else BGEEncoder(
        revision=revision, device=device, cpu_threads=cpu_threads,
        query_max_tokens=query_max_tokens, verbose=True)
    model_load_seconds = time.perf_counter() - load_start
    config = {
        "method": "Flat BGE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": encoder.signature, "inference": encoder.inference_settings,
        "requested_revision": revision,
        "passages_path": str(passages_path), "passages_sha256": file_hash(passages_path),
        "questions_path": str(questions_path), "questions_sha256": file_hash(questions_path),
        "cache_path": str(cache_path) if cache_path is not None else None,
        "cache_metadata_path": str(cache_metadata_path(cache_path)) if cache_path is not None else None,
        "passage_batch_size": batch_size, "query_batch_size": query_batch_size,
        "top_k_requested": top_k, "top_k_available": min(top_k, len(passages)),
        "total_passages": len(passages), "total_documents": len({p["url"] for p in passages}),
        "total_questions": len(questions),
        "questions_without_evidence": sum(not q["evidence_list"] for q in questions),
        "evidence_matching_failures": [], "ranking_unit": "passage",
        "ranking": "Exact float32 CPU dot products over all L2-normalized passage vectors",
        "tie_break": "passage_id ascending; original passage embedding row order preserved",
        "question_id_policy": "Zero-based original question order",
        "python": platform.python_version(), "platform": platform.platform(),
        "code_sha256": {p: file_hash(p) for p in (
            "src/hhsi/retrieval/common.py", "src/hhsi/retrieval/dense.py", "scripts/run_dense.py")},
        "latency_definition": (
            "Per-question latency_seconds measures search_embedding only: query vector validation, "
            "exact similarity over every passage, sorting and result construction. "
            "Excludes model loading, passage/query encoding and JSON writing; "
            "do not compare directly to BM25's text-to-results latency."),
        "total_duration_definition": (
            "run() entry through saved-output audit, including loading, cache I/O, inference, "
            "search and JSONL writing; excludes Python imports and final summary serialization."),
    }
    write_json(output_dir / "config.json", config)
    retriever = DenseRetriever(passages, encoder=encoder, cache_path=cache_path,
                               dataset_sha256=config["passages_sha256"], batch_size=batch_size)
    print(f"Passage cache {'loaded' if retriever.cache_hit else 'generated'}; "
          f"{len(passages)} x {DIMENSION} embeddings", flush=True)
    query_start = time.perf_counter()
    queries = encoder.encode_queries([q["query"] for q in questions], batch_size=query_batch_size)
    query_seconds = time.perf_counter() - query_start
    query_stats = dict(encoder.last_encoding_stats)
    latencies = []
    retrieval_path = output_dir / "retrieval.jsonl"
    with retrieval_path.open("x", encoding="utf-8") as stream:
        for i, (question, vector) in enumerate(zip(questions, queries, strict=True)):
            search_start = time.perf_counter()
            results = retriever.search_embedding(vector, top_k)
            latency = time.perf_counter() - search_start
            latencies.append(latency)
            row = {"question_id": f"q_{i:04d}", "query": question["query"],
                   "question_type": question.get("question_type"),
                   "evidence_list": question["evidence_list"], "latency_seconds": latency,
                   "results": results}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"Retrieved {i + 1}/{len(questions)} questions", flush=True)
    audit = validate_saved_run(retrieval_path, passages, questions, top_k)
    summary = {
        "status": "complete", "total_passages": len(passages), "embedding_dimension": DIMENSION,
        "retrieved_questions": len(questions), "evaluated_questions": 0,
        "evaluation_status": "Not implemented; no relevance metrics computed",
        "device": encoder.inference_settings["device"],
        "resolved_revision": encoder.signature["resolved_revision"],
        "cache_hit": retriever.cache_hit,
        "timings_seconds": {
            "input_loading_and_validation": input_seconds,
            "model_loading": model_load_seconds,
            "passage_embedding_generation": retriever.passage_encoding_seconds,
            "cache_loading_and_validation": retriever.cache_load_seconds,
            "cache_writing": retriever.cache_write_seconds,
            "question_embedding_generation": query_seconds,
            "similarity_search_total": sum(latencies),
            "similarity_search_mean": float(np.mean(latencies)),
            "similarity_search_median": float(np.median(latencies)),
            "similarity_search_p95": float(np.percentile(latencies, 95)),
            "amortized_query_encoding_plus_search": (query_seconds + sum(latencies)) / len(questions),
            "total_experiment": time.perf_counter() - start,
        },
        "passage_encoding_stats": retriever.passage_encoding_stats,
        "passage_stats_origin": "cache generation run" if retriever.cache_hit else "current run",
        "query_encoding_stats": query_stats, "output_validation": audit,
        "evidence_matching_failures": [],
        "retrieval_sha256": file_hash(retrieval_path),
        "embedding_file_sha256": retriever.cache_metadata.get("embedding_file_sha256"),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, default=Path("data/processed/multihoprag_passages.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/raw/MultiHopRAG.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--query-max-tokens", type=int, default=512)
    parser.add_argument("--expected-questions", type=int, default=2556)
    parser.add_argument("--expected-passages", type=int, default=6638)
    args = parser.parse_args()
    output = args.output_dir or Path("results/baselines/multihoprag") / (
        "dense_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + "_" + uuid.uuid4().hex[:8])
    print(f"Output directory: {output}", flush=True)
    summary = run(
        args.passages, args.questions, output, top_k=args.top_k, cache_path=args.cache,
        batch_size=args.batch_size, query_batch_size=args.query_batch_size,
        device=args.device, cpu_threads=args.cpu_threads, revision=args.revision,
        query_max_tokens=args.query_max_tokens, expected_questions=args.expected_questions,
        expected_passages=args.expected_passages)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("query_encoding_stats", "passage_encoding_stats")}, indent=2))


if __name__ == "__main__":
    main()
