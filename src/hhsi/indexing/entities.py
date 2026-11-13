"""Deterministic surface-form entities; no linking or alias resolution."""
from collections import Counter
import json
import statistics
import unicodedata

from ..retrieval.common import validate_passages
from ..retrieval.dense import positive_int

DEFAULT_LABELS = ("PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "PRODUCT",
                  "WORK_OF_ART", "NORP")
NORMALIZATION = {
    "version": 1, "unicode": "NFKC", "case": "casefold",
    "whitespace": "collapse and strip",
    "punctuation": "strip Unicode punctuation at boundaries; preserve internal punctuation and symbols",
    "deduplication": "normalized string; first mention retained, variants preserved",
}


def normalize_entity(text):
    if not isinstance(text, str):
        raise TypeError("Entity text must be a string")
    value = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    while value and (value[0].isspace() or unicodedata.category(value[0]).startswith("P")):
        value = value[1:]
    while value and (value[-1].isspace() or unicodedata.category(value[-1]).startswith("P")):
        value = value[:-1]
    return value


def load_ner(model="en_core_web_sm"):
    import spacy
    try:
        nlp = spacy.load(model)
    except OSError as error:
        raise RuntimeError(f"Install the NER model: python -m spacy download {model}") from error
    if "ner" not in nlp.pipe_names:
        raise ValueError("The pipeline must contain a pretrained NER component")
    # Keep model defaults to avoid changing component dependencies.
    return nlp


def extract_entities(passages, *, nlp=None, model="en_core_web_sm",
                     labels=DEFAULT_LABELS, batch_size=64):
    """Return ordered records and counts. Counts by type count retained raw mentions."""
    rows = validate_passages(passages)
    positive_int(batch_size, "batch_size")
    if isinstance(labels, str):
        raise ValueError("labels must be a collection, not a string")
    labels = frozenset(labels)
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Entity labels must be nonempty strings")
    nlp = load_ner(model) if nlp is None else nlp
    records, counts = [], Counter()
    total_detected = 0
    for passage, doc in zip(rows, nlp.pipe((p["text"] for p in rows),
                                          batch_size=batch_size), strict=True):
        unique = {}
        for ent in doc.ents:
            total_detected += 1
            if ent.label_ not in labels:
                continue
            counts[ent.label_] += 1
            normalized = normalize_entity(ent.text)
            if not normalized:
                continue
            variant = {"text": ent.text, "label": ent.label_}
            if normalized not in unique:
                unique[normalized] = {**variant, "normalized": normalized,
                                      "mentions": []}
            # Preserve every original surface form and type, including duplicates.
            unique[normalized]["mentions"].append(variant)
        records.append({"passage_id": passage["passage_id"], "doc_id": passage["doc_id"],
                        "entities": list(unique.values())})
    validate_entity_records(records, rows)
    sizes = [len(row["entities"]) for row in records]
    report = {
        "passages_processed": len(rows),
        "total_detected_mentions_all_labels": total_detected,
        "total_extracted_entity_mentions": sum(counts.values()),
        "entity_counts_by_type": dict(sorted(counts.items())),
        "unique_normalized_entities": len({e["normalized"] for r in records for e in r["entities"]}),
        "passages_with_no_entities": sizes.count(0),
        "mean_unique_entities_per_passage": statistics.mean(sizes),
        "median_unique_entities_per_passage": statistics.median(sizes),
        "retained_labels": sorted(labels), "normalization": NORMALIZATION,
    }
    return records, report


def validate_entity_records(records, passages):
    rows = validate_passages(passages)
    if len(records) != len(rows):
        raise ValueError("Expected exactly one entity record per passage")
    for record, passage in zip(records, rows, strict=True):
        if any(record.get(k) != passage[k] for k in ("passage_id", "doc_id")):
            raise ValueError("Entity passage-ID/document-ID alignment mismatch")
        if not isinstance(record.get("entities"), list):
            raise ValueError("entities must be a list")
        seen = set()
        for entity in record["entities"]:
            if not isinstance(entity, dict):
                raise ValueError("Entity must be an object")
            text, label, normalized = (entity.get(k) for k in ("text", "label", "normalized"))
            if (not isinstance(text, str) or not isinstance(label, str) or not label
                    or not isinstance(normalized, str) or not normalized
                    or normalized != normalize_entity(text) or normalized in seen):
                raise ValueError("Invalid or duplicate normalized entity")
            seen.add(normalized)
    return records


def load_entity_records(path, passages):
    with open(path, encoding="utf-8") as stream:
        return validate_entity_records([json.loads(line) for line in stream], passages)
