import copy
import json

import pytest

from scripts.evaluate_baselines import digest, run
from src.hhsi.evaluation import evaluate


def fixture_data(tmp_path):
    passages = [{"passage_id": f"p{i}", "doc_id": url, "url": url, "text": "text"}
                for i, url in enumerate(["a", "a", "b", "c"])]
    questions = [{"query": f"query {i}", "question_type": "test",
                  "evidence_list": [{"url": u} for u in gold]}
                 for i, gold in enumerate([["a", "a", "b"], ["c"], []])]
    results = [{k: p[k] for k in ("passage_id", "doc_id", "url")} |
               {"score": float(4-i), "rank": i+1} for i, p in enumerate(passages)]
    rows = [{**q, "question_id": f"q_{i:04d}", "results": copy.deepcopy(results)}
            for i, q in enumerate(questions)]
    path = tmp_path / "retrieval.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return passages, questions, rows, path


def test_hand_computed_metrics_duplicates_misses_and_empty(tmp_path):
    passages, questions, _, path = fixture_data(tmp_path)
    summary, rows = evaluate(path, questions, passages, (1, 2, 3, 4))
    assert summary["evaluated_questions"] == 2
    assert summary["excluded_no_evidence"] == 1
    assert rows[2]["metrics"] is None
    assert summary["metrics_at_k"]["2"] == {"recall": .25, "mrr": .5, "all_hop_coverage": 0}
    assert summary["metrics_at_k"]["3"] == {"recall": .5, "mrr": .5, "all_hop_coverage": .5}
    assert summary["metrics_at_k"]["4"] == {"recall": 1, "mrr": .625, "all_hop_coverage": 1}


@pytest.mark.parametrize("fault", ["depth", "order", "mapping", "duplicate", "nan", "missing", "gold"])
def test_rejects_invalid_saved_inputs(tmp_path, fault):
    passages, questions, rows, path = fixture_data(tmp_path)
    if fault == "depth": rows[0]["results"].pop()
    if fault == "order": rows.reverse()
    if fault == "mapping": rows[0]["results"][0]["url"] = "wrong"
    if fault == "duplicate": rows[0]["results"][1]["passage_id"] = "p0"
    if fault == "nan": rows[0]["results"][0]["score"] = float("nan")
    if fault == "missing": rows.pop()
    if fault == "gold": questions[0]["evidence_list"].append({"url": "unknown"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        evaluate(path, questions, passages, (4,))


@pytest.mark.parametrize("ks", [(), (0,), (-1,), (1.5,), (True,)])
def test_invalid_cutoffs(tmp_path, ks):
    passages, questions, _, path = fixture_data(tmp_path)
    with pytest.raises(ValueError): evaluate(path, questions, passages, ks)


def test_saved_comparison_provenance_and_no_overwrite(tmp_path):
    passages, questions, _, path = fixture_data(tmp_path)
    pp, qp = tmp_path / "passages.jsonl", tmp_path / "questions.json"
    pp.write_text("".join(json.dumps(p) + "\n" for p in passages))
    qp.write_text(json.dumps(questions))
    config = {"passages_sha256": digest(pp), "questions_sha256": digest(qp)}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "summary.json").write_text(json.dumps({"retrieval_sha256": digest(path)}))
    output = tmp_path / "evaluation"
    result = run(tmp_path, tmp_path, output, ks=(2, 4), passages_path=pp, questions_path=qp)
    assert result["baselines"]["bm25"] == result["baselines"]["dense"]
    assert json.loads((output / "summary.json").read_text()) == result
    assert len((output / "dense_per_question.jsonl").read_text().splitlines()) == 3
    with pytest.raises(FileExistsError):
        run(tmp_path, tmp_path, output, ks=(2,), passages_path=pp, questions_path=qp)
    config["questions_sha256"] = "mismatch"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hashes"):
        run(tmp_path, tmp_path, tmp_path / "bad", ks=(2,), passages_path=pp, questions_path=qp)
