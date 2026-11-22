import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASET = "yixuantt/MultiHopRAG"
REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
RAW = ROOT / "data/multihoprag"
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results"


def run(*args, dry_run=False):
    command = [sys.executable, *map(str, args)]
    print("Running:", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def ensure_inputs(dry_run):
    from huggingface_hub import hf_hub_download

    for filename in ("corpus.json", "MultiHopRAG.json"):
        destination = RAW / filename
        if destination.is_file():
            continue
        print(f"Downloading {DATASET}/{filename} to {destination}", flush=True)
        if not dry_run:
            source = hf_hub_download(DATASET, filename, repo_type="dataset", revision=REVISION)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    import spacy

    if not spacy.util.is_package("en_core_web_sm"):
        run("-m", "spacy", "download", "en_core_web_sm", dry_run=dry_run)

    config = json.loads((ROOT / "config/answer_models.json").read_text(encoding="utf-8"))
    model = Path(config["profiles"][config["default_profile"]]["model_path"])
    server = Path(config["server_exe"])
    if not model.is_file() or not server.is_file():
        raise FileNotFoundError(f"Llama model or server missing: {model}, {server}")


def clear_generated(dry_run):
    for path in (PROCESSED, RESULTS):
        resolved = path.resolve()
        if resolved.parent != ROOT.resolve():
            raise ValueError(f"Refusing to delete outside project root: {resolved}")
        if path.exists():
            print(f"Removing generated artifacts: {resolved}", flush=True)
            if not dry_run:
                shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show actions without downloading, deleting, or running stages")
    args = parser.parse_args()
    ensure_inputs(args.dry_run)
    clear_generated(args.dry_run)

    questions = RAW / "MultiHopRAG.json"
    bm25 = RESULTS / "baselines/multihoprag/bm25_fresh"
    dense = RESULTS / "baselines/multihoprag/dense_fresh"
    evaluation = RESULTS / "evaluation/multihoprag/fresh"
    stage = lambda *command: run("-m", *command, dry_run=args.dry_run)

    stage("scripts.prepare_passages", "--corpus", RAW / "corpus.json")
    stage("scripts.extract_entities")
    stage("scripts.run_bm25", "--questions", questions, "--output-dir", bm25)
    stage("scripts.run_dense", "--questions", questions, "--output-dir", dense)
    stage("scripts.build_hierarchy")
    stage("scripts.demonstrate_similarity")
    stage("scripts.evaluate_baselines", "--questions", questions,
          "--bm25-dir", bm25, "--dense-dir", dense, "--output-dir", evaluation)
    stage("scripts.plot_baselines", evaluation / "summary.json")
    stage("scripts.answer_with_evidence", dense / "retrieval.jsonl", "--question-id", "q_0000")


if __name__ == "__main__":
    main()
