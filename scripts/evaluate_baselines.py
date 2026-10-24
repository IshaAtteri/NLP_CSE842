"""Compare saved BM25 and dense rankings without running retrieval again."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

from src.hhsi.evaluation import evaluate
from src.hhsi.retrieval.common import load_passages

BASE = Path("results/baselines/multihoprag")
DEFAULT_BM25 = BASE / "bm25_20261021T162111_053749Z_3e0e3601"
DEFAULT_DENSE = BASE / "dense_20261022T152341_294173Z_cddb4578"


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def run(bm25_dir, dense_dir, output_dir, *, ks=(5, 10, 20, 50),
        passages_path="data/processed/multihoprag_passages.jsonl",
        questions_path="data/raw/MultiHopRAG.json"):
    start = time.perf_counter()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory exists: {output_dir}")
    passages = load_passages(passages_path)
    questions = json.loads(Path(questions_path).read_bytes())
    hashes = {"passages_sha256": digest(passages_path), "questions_sha256": digest(questions_path)}
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoffs": list(ks), **hashes,
        "definitions": {
            "relevance": "Exact evidence URL match; article-level proxy, not passage fact verification",
            "recall": "Unique evidence articles found in top-k passages / unique gold evidence articles",
            "mrr": "Mean reciprocal rank of first evidence-article passage within k; zero if absent",
            "all_hop_coverage": "Fraction of questions with every unique evidence article in top-k passages",
            "aggregation": "Macro average over questions with nonempty evidence; others excluded",
            "ranking": "Original passage ranks, no article deduplication before cutoff",
            "mrr_limit": "MRR is truncated at k; saved top-50 cannot establish unrestricted MRR",
        },
        "inputs": {},
        "code_sha256": {p: digest(p) for p in ("src/hhsi/evaluation.py", "scripts/evaluate_baselines.py")},
    }
    summaries, rows = {}, {}
    for name, directory in (("bm25", bm25_dir), ("dense", dense_dir)):
        directory = Path(directory)
        source_config = json.loads((directory / "config.json").read_bytes())
        source_summary = json.loads((directory / "summary.json").read_bytes())
        if any(source_config.get(key) != value for key, value in hashes.items()):
            raise ValueError(f"{name}: dataset hashes do not match")
        retrieval = directory / "retrieval.jsonl"
        retrieval_hash = digest(retrieval)
        if source_summary.get("retrieval_sha256") != retrieval_hash:
            raise ValueError(f"{name}: retrieval checksum mismatch")
        config["inputs"][name] = {"directory": str(directory), "retrieval_sha256": retrieval_hash}
        summaries[name], rows[name] = evaluate(retrieval, questions, passages, ks)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, records in rows.items():
        with (output_dir / f"{name}_per_question.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    summary = {"baselines": summaries, "evaluation_seconds": time.perf_counter() - start}
    for name, value in (("config.json", config), ("summary.json", summary)):
        with (output_dir / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-dir", type=Path, default=DEFAULT_BM25)
    parser.add_argument("--dense-dir", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--passages", type=Path, default=Path("data/processed/multihoprag_passages.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/raw/MultiHopRAG.json"))
    args = parser.parse_args()
    output = args.output_dir or Path("results/evaluation/multihoprag") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    result = run(args.bm25_dir, args.dense_dir, output, ks=args.ks,
                 passages_path=args.passages, questions_path=args.questions)
    print(json.dumps(result, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
