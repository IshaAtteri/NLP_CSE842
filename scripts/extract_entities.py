"""Extract reproducible entity records without modifying passages or embeddings."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

from src.hhsi.indexing.entities import DEFAULT_LABELS, extract_entities, load_entity_records, load_ner
from src.hhsi.retrieval.common import load_passages
from src.hhsi.retrieval.dense import file_hash

DEFAULT_PASSAGES = Path("data/processed/multihoprag_passages.jsonl")
DEFAULT_OUTPUT = Path("data/processed/entities/multihoprag_entities.jsonl")


def run(passages_path=DEFAULT_PASSAGES, output=DEFAULT_OUTPUT, *, model="en_core_web_sm",
        labels=DEFAULT_LABELS, batch_size=64, expected_passages=6638, nlp=None):
    output = Path(output)
    report_path = output.with_suffix(".report.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("Entity output or report exists; choose a new --output")
    passages = load_passages(passages_path)
    if len(passages) != expected_passages:
        raise ValueError(f"Expected {expected_passages} passages, got {len(passages)}")
    dataset_hash = file_hash(passages_path)
    start = time.perf_counter()
    nlp = load_ner(model) if nlp is None else nlp
    records, report = extract_entities(passages, nlp=nlp, labels=labels, batch_size=batch_size)
    report.update({
        "schema_version": 1, "processing_seconds": time.perf_counter() - start,
        "model_name": model, "model_version": nlp.meta.get("version"),
        "spacy_version": version("spacy"), "pipeline": nlp.pipe_names,
        "batch_size": batch_size, "input_passage_dataset_sha256": dataset_hash,
        "input_path": str(passages_path),
        "code_sha256": {str(p): file_hash(p) for p in (
            Path(__file__), Path("src/hhsi/indexing/entities.py"))},
    })
    if file_hash(passages_path) != dataset_hash:
        raise ValueError("Passage dataset changed during extraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    load_entity_records(output, passages)
    report["entity_file_sha256"] = file_hash(output)
    # Written last: report signals a completed, validated extraction.
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expected-passages", type=int, default=6638)
    args = parser.parse_args()
    print(json.dumps(run(args.passages, args.output, model=args.model, labels=args.labels,
                         batch_size=args.batch_size, expected_passages=args.expected_passages), indent=2))


if __name__ == "__main__":
    main()
