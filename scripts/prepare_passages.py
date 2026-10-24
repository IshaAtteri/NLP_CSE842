"""Run from the repository root: python -m scripts.prepare_passages."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

from src.hhsi.data.chunking import MODEL_NAME
from src.hhsi.data.passages import build_passages, tokenizer_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/raw/corpus.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/multihoprag_passages.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--overlap-tokens", type=int, default=32)
    parser.add_argument("--expected-articles", type=int, default=609)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".report.json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("Output or report already exists; choose a new --output to preserve artifacts.")
    raw = args.corpus.read_bytes()
    corpus = json.loads(raw)
    if len(corpus) != args.expected_articles:
        raise ValueError(f"Expected {args.expected_articles} articles, got {len(corpus)}")
    start = time.perf_counter()
    passages, report = build_passages(corpus, args.max_tokens, args.overlap_tokens)
    report["configuration"] = {
        "corpus": str(args.corpus), "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "model": MODEL_NAME, "tokenizer_sha256": tokenizer_fingerprint(),
        "max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
        "document_id_policy": "Zero-based input corpus order, matching scripts/corpus.py",
        "indexed_text": "body only; metadata preserved separately",
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("transformers", "tokenizers", "spacy", "numpy")},
        "chunking_code_sha256": hashlib.sha256(Path("src/hhsi/data/chunking.py").read_bytes()).hexdigest(),
        "validation_code_sha256": hashlib.sha256(Path("src/hhsi/data/passages.py").read_bytes()).hexdigest(),
    }
    report["processing_seconds"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if report["failures"]:
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        raise RuntimeError(f"Processing failed; no partial passage dataset saved. See {report_path}")
    payload = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in passages).encode("utf-8")
    report["passages_sha256"] = hashlib.sha256(payload).hexdigest()
    with args.output.open("xb") as stream:
        stream.write(payload)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "documents"}, indent=2))


if __name__ == "__main__":
    main()
