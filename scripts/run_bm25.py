"""Retrieve all MultiHop-RAG questions; save results for later evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import uuid

import numpy as np

from src.hhsi.retrieval.bm25 import BM25Retriever, lexical_tokens
from src.hhsi.retrieval.common import load_passages


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def run(passages_path, questions_path, output_dir, *, top_k=50,
        expected_questions=2556, k1=1.5, b=0.75, epsilon=0.25):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    passages = load_passages(passages_path)
    questions = json.loads(Path(questions_path).read_bytes())
    if not isinstance(questions, list) or not questions:
        raise ValueError("Questions must be a nonempty list")
    if len(questions) != expected_questions:
        raise ValueError(f"Expected {expected_questions} questions, got {len(questions)}")
    available_urls = {p["url"] for p in passages}
    matching_failures = []
    for i, question in enumerate(questions):
        if not lexical_tokens(question["query"]):
            raise ValueError(f"Empty query at question {i}")
        if not isinstance(question.get("evidence_list"), list):
            raise ValueError(f"Invalid evidence list at question {i}")
        for evidence in question["evidence_list"]:
            if not isinstance(evidence.get("url"), str) or not evidence["url"]:
                raise ValueError(f"Missing evidence URL at question {i}")
            if evidence["url"] not in available_urls:
                matching_failures.append({"question_id": f"q_{i:04d}", "url": evidence["url"]})
    if matching_failures:
        raise ValueError(f"Evidence URLs missing from passages: {matching_failures}")
    start = time.perf_counter()
    retriever = BM25Retriever(passages, k1=k1, b=b, epsilon=epsilon)
    index_seconds = time.perf_counter() - start
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "method": "BM25", **retriever.configuration,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passages_path": str(passages_path), "passages_sha256": sha256(passages_path),
        "questions_path": str(questions_path), "questions_sha256": sha256(questions_path),
        "top_k_requested": top_k, "top_k_available": min(top_k, len(passages)),
        "ranking_unit": "passage", "question_id_policy": "Zero-based input question order",
        "total_passages": len(passages), "total_documents": len(available_urls),
        "total_questions": len(questions),
        "questions_without_evidence": sum(not q["evidence_list"] for q in questions),
        "evidence_matching_failures": matching_failures,
        "index_preparation_seconds": index_seconds,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: version(p) for p in ("rank-bm25", "numpy")},
        "randomness": "None: exhaustive scoring and deterministic tie breaking",
        "code_sha256": {p: sha256(p) for p in (
            "src/hhsi/retrieval/bm25.py", "src/hhsi/retrieval/common.py", "scripts/run_bm25.py")},
        "latency_definition": "Query preprocessing, exhaustive scoring, sorting and result construction; excludes file I/O.",
    }
    write_json(output_dir / "config.json", config)
    latencies = []
    with (output_dir / "retrieval.jsonl").open("x", encoding="utf-8") as stream:
        for i, question in enumerate(questions):
            start = time.perf_counter()
            results = retriever.search(question["query"], top_k)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            record = {
                "question_id": f"q_{i:04d}", "query": question["query"],
                "question_type": question.get("question_type"),
                "evidence_list": question["evidence_list"],
                "latency_seconds": elapsed, "results": results,
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"Retrieved {i + 1}/{len(questions)} questions", flush=True)
    summary = {
        "status": "complete", "retrieved_questions": len(latencies),
        "evaluated_questions": 0, "evaluation_status": "Deferred to Phase 4",
        "total_passages": len(passages), "index_preparation_seconds": index_seconds,
        "retrieval_seconds": sum(latencies),
        "latency_seconds": {"mean": float(np.mean(latencies)),
                            "median": float(np.median(latencies)),
                            "p95": float(np.percentile(latencies, 95)),
                            "min": min(latencies), "max": max(latencies)},
        "evidence_matching_failures": matching_failures,
        "retrieval_sha256": sha256(output_dir / "retrieval.jsonl"),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, default=Path("data/processed/multihoprag_passages.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/raw/MultiHopRAG.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--expected-questions", type=int, default=2556)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--epsilon", type=float, default=0.25)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("results/baselines/multihoprag") / (
        "bm25_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8])
    summary = run(args.passages, args.questions, output_dir, top_k=args.top_k,
                  expected_questions=args.expected_questions, k1=args.k1, b=args.b, epsilon=args.epsilon)
    print(f"Saved to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
