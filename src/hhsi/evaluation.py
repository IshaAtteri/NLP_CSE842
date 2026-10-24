"""Evidence-article retrieval metrics at original passage ranks (no model inference)."""
import json
import math
from pathlib import Path


def evaluate(path, questions, passages, ks=(5, 10, 20, 50)):
    ks = tuple(sorted(set(ks)))
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("Cutoffs must be positive integers")
    lookup = {p["passage_id"]: p for p in passages}
    urls = {p["url"] for p in passages}
    totals = {str(k): {"recall": 0.0, "mrr": 0.0, "all_hop_coverage": 0.0} for k in ks}
    per_question = []
    evaluated = 0
    with Path(path).open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i >= len(questions):
                raise ValueError("Too many question records")
            row = json.loads(line)
            question = questions[i]
            if (row["question_id"] != f"q_{i:04d}" or
                    any(row[key] != question.get(key) for key in
                        ("query", "question_type", "evidence_list"))):
                raise ValueError(f"Question identity/annotations mismatch at {i}")
            gold = {e["url"] for e in question["evidence_list"]}
            if not gold <= urls:
                raise ValueError(f"Unknown evidence URL at {i}")
            results = row["results"]
            if len(results) < max(ks):
                raise ValueError(f"Saved retrieval depth is less than requested cutoff at {i}")
            seen = set()
            previous = None
            for rank, result in enumerate(results, 1):
                passage = lookup.get(result["passage_id"])
                if passage is None or any(result[key] != passage[key] for key in ("doc_id", "url")):
                    raise ValueError(f"Invalid passage mapping at {i}")
                if result["passage_id"] in seen or result["rank"] != rank:
                    raise ValueError(f"Duplicate passage or incorrect rank at {i}")
                seen.add(result["passage_id"])
                score = result["score"]
                if not math.isfinite(score):
                    raise ValueError(f"Nonfinite score at {i}")
                order = (-score, result["passage_id"])
                if previous is not None and order < previous:
                    raise ValueError(f"Incorrect score/tie ordering at {i}")
                previous = order
            metrics = None
            if gold:
                evaluated += 1
                metrics = {}
                for k in ks:
                    hits = gold & {r["url"] for r in results[:k]}
                    first = next((r["rank"] for r in results[:k] if r["url"] in gold), None)
                    metrics[str(k)] = {
                        "recall": len(hits) / len(gold),
                        "mrr": 1.0 / first if first is not None else 0.0,
                        "all_hop_coverage": float(hits == gold),
                    }
                    for name, value in metrics[str(k)].items():
                        totals[str(k)][name] += value
            per_question.append({"question_id": row["question_id"],
                                 "question_type": row["question_type"],
                                 "evidence_articles": len(gold), "metrics": metrics})
    if len(per_question) != len(questions):
        raise ValueError("Missing question records")
    if not evaluated:
        raise ValueError("No questions with evidence to evaluate")
    for values in totals.values():
        for name in values:
            values[name] /= evaluated
    return {"total_questions": len(questions), "evaluated_questions": evaluated,
            "excluded_no_evidence": len(questions) - evaluated,
            "metrics_at_k": totals}, per_question
