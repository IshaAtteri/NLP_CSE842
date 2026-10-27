import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

import spacy


ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = ROOT / "outputs" / "embeddings"

MODEL_NAME = "en_core_web_sm"

# Entity types retained for HHSI.
# Dates and numerical quantities are excluded from entity matching.
ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "NORP",
    "EVENT",
    "PRODUCT",
    "WORK_OF_ART",
}


def normalize_entity(text):
    """Standardize Unicode, capitalization, and whitespace."""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.casefold().split())


def read_passages(path):
    """Read passages one at a time, keeping the same order"""
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def extract_entity_data(doc):
    """Keep entity mentions and build a deduplicated entity set."""
    mentions = []
    entity_set = set()

    for entity in doc.ents:
        if entity.label_ not in ENTITY_TYPES:
            continue

        normalized = normalize_entity(entity.text)

        if not normalized:
            continue

        mentions.append({
            "text": entity.text,
            "label": entity.label_,
            "normalized": normalized,
            "start_char": entity.start_char,
            "end_char": entity.end_char,
        })

        entity_set.add(normalized)

    return mentions, sorted(entity_set)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "multihoprag",
            "hotpotqa_validation_first20",
            "hotpotqa_validation",
            "hotpotqa_train",
        ],
        help="Select an existing embedding output folder.",
    )

    args = parser.parse_args()

    folder = EMBEDDINGS_DIR / args.dataset
    passages_path = folder / "passages.jsonl"
    ids_path = folder / "passage_ids.json"

    if not passages_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Required passage files were not found in:\n{folder}\n"
            "Run generate_embeddings.py for this dataset first."
        )

    expected_ids = json.loads(
        ids_path.read_text(encoding="utf-8")
    )

    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Duplicate IDs in passage_ids.json.")

    print("Loading spaCy model:", MODEL_NAME)

    nlp = spacy.load(
        MODEL_NAME,
        exclude=["tokenizer"],
        disable=[
            "tagger",
            "parser",
            "attribute_ruler",
            "lemmatizer",
        ],
    )

    # Keeping the original passage record alongside its processed text.
    inputs = (
        (passage["text"], passage)
        for passage in read_passages(passages_path)
    )

    output_path = folder / "passage_entities.jsonl"
    temporary_path = folder / "passage_entities.jsonl.tmp"

    processed = 0
    empty_count = 0
    mention_count = 0
    label_counts = Counter()
    examples = []

    print(f"Extracting entities from {len(expected_ids)} passages...")

    # To process texts in batches while preserving input order.
    with temporary_path.open("w", encoding="utf-8") as output:
        for doc, passage in nlp.pipe(
            inputs,
            as_tuples=True,
            batch_size=32,
        ):
            passage_id = passage["passage_id"]

            # To check alignment with the saved embedding IDs.
            if (
                processed >= len(expected_ids)
                or passage_id != expected_ids[processed]
            ):
                raise ValueError(
                    f"Passage ID mismatch at row {processed}."
                )

            mentions, entity_set = extract_entity_data(doc)

            record = {
                "passage_id": passage_id,
                "article_id": passage["article_id"],
                "entities": mentions,
                "entity_set": entity_set,
            }

            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            processed += 1
            mention_count += len(mentions)
            empty_count += int(not entity_set)

            label_counts.update(
                mention["label"] for mention in mentions
            )

            if entity_set and len(examples) < 3:
                examples.append({
                    "passage_id": passage_id,
                    "entity_set": entity_set,
                })

            if processed % 500 == 0:
                print(
                    f"Processed {processed}/{len(expected_ids)} passages"
                )

    if processed != len(expected_ids):
        raise ValueError(
            f"Expected {len(expected_ids)} passages, "
            f"but processed {processed}."
        )

    # To publish the completed file after alignment checks pass.
    temporary_path.replace(output_path)

    summary = {
        "dataset": args.dataset,
        "spacy_version": spacy.__version__,
        "model": MODEL_NAME,
        "model_version": nlp.meta.get("version"),
        "tokenizer_mode": "default English tokenizer; saved tokenizer excluded",
        "retained_entity_types": sorted(ENTITY_TYPES),
        "normalization": "NFKC + casefold + whitespace normalization",
        "passage_count": processed,
        "passages_without_retained_entities": empty_count,
        "entity_mention_count": mention_count,
        "mentions_by_type": dict(label_counts),
        "examples": examples,
    }

    summary_path = folder / "entity_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nCompleted successfully!")
    print("Passages processed:", processed)
    print("Entity mentions extracted:", mention_count)
    print("Passages without retained entities:", empty_count)
    print("Entities saved to:", output_path)
    print("Summary saved to:", summary_path)

    print("\nSample entity sets:")
    for example in examples:
        print(example["passage_id"], "->", example["entity_set"])


if __name__ == "__main__":
    main()