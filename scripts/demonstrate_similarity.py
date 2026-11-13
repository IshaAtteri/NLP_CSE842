"""Save three illustrative selected-pair comparisons, never retrieval metrics."""
import argparse
import json
from pathlib import Path
import numpy as np

from src.hhsi.indexing.entities import load_entity_records
from src.hhsi.indexing.similarity import PassageSimilarity
from src.hhsi.retrieval.common import load_passages
from src.hhsi.retrieval.dense import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, default=Path("data/processed/multihoprag_passages.jsonl"))
    parser.add_argument("--entities", type=Path, default=Path("data/processed/entities/multihoprag_entities.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/embeddings/multihoprag_bge_base_v1_5.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/phase3/multihoprag_similarity_examples.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new --output to preserve existing examples")
    passages = load_passages(args.passages)
    records = load_entity_records(args.entities, passages)
    api = PassageSimilarity(passages, records, args.cache, dataset_sha256=file_hash(args.passages))
    # Fixed small anchor sample; only one length-N score vector exists at a time.
    # Exclude same-document pairs to avoid selecting overlapping passage chunks.
    anchors = list(range(min(12, len(passages))))
    best_shared, best_related, least_related = None, None, None
    for i in anchors:
        scores = api.embeddings @ api.embeddings[i]
        for j, score in enumerate(scores):
            if passages[i]["doc_id"] == passages[j]["doc_id"]:
                continue
            shared = api.entities[i] & api.entities[j]
            candidate = (float(score), i, j)
            if shared and (best_shared is None or candidate[0] > best_shared[0]):
                best_shared = candidate
            if not shared and (best_related is None or candidate[0] > best_related[0]):
                best_related = candidate
            if not shared and (least_related is None or candidate[0] < least_related[0]):
                least_related = candidate
    examples = []
    for name, candidate in (("shared_named_entities", best_shared),
                            ("related_without_shared_entities", best_related),
                            ("relatively_unrelated", least_related)):
        if candidate is None:
            raise ValueError(f"No eligible demonstration pair for {name}")
        _, i, j = candidate
        a, b = passages[i]["passage_id"], passages[j]["passage_id"]
        result = api.pair(a, b, alpha=0.7)
        result["category"] = name
        result["hybrid_alpha_0_7"] = result.pop("hybrid")
        result["hybrid_alpha_0_5"] = api.pair(a, b, alpha=0.5)["hybrid"]
        result["passage_a"] = {"title": passages[i].get("title"), "text": passages[i]["text"]}
        result["passage_b"] = {"title": passages[j].get("title"), "text": passages[j]["text"]}
        examples.append(result)
    report = {
        "purpose": "Illustrative similarity examples, not retrieval performance measurements",
        "selection": "Across documents, first 12 passages as anchors; maximum cosine with shared entities, maximum cosine without shared entities, minimum cosine without shared entities; first encountered tie wins",
        "alpha_note": "0.5 and 0.7 are illustrative settings, not test-set tuned",
        "input_hashes": {str(p): file_hash(p) for p in (args.passages, args.entities, args.cache)},
        "validated_passages": len(passages), "embedding_shape": list(api.embeddings.shape),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
