import json
from pathlib import Path
import shutil
import subprocess
import sys

from huggingface_hub import hf_hub_download
import spacy


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/multihoprag"
RESULTS = ROOT / "results"


def run(module, *args):
    command = [sys.executable, "-m", module, *map(str, args)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    for name in ("corpus.json", "MultiHopRAG.json"):
        path = RAW / name
        if not path.is_file():
            source = hf_hub_download("yixuantt/MultiHopRAG", name, repo_type="dataset",
                                     revision="71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82")
            RAW.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, path)

    if not spacy.util.is_package("en_core_web_sm"):
        run("spacy", "download", "en_core_web_sm")

    config = json.loads((ROOT / "config/answer_models.json").read_text(encoding="utf-8"))
    model = Path(config["profiles"][config["default_profile"]]["model_path"])
    server = Path(config["server_exe"])
    if not model.is_file() or not server.is_file():
        raise FileNotFoundError(f"Llama model or server missing: {model}, {server}")

    for path in (ROOT / "data/processed", RESULTS):
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Refusing to delete outside project: {path}")
        if path.exists():
            print(f"Clearing {path}", flush=True)
            shutil.rmtree(path)

    questions = RAW / "MultiHopRAG.json"
    bm25 = RESULTS / "baselines/multihoprag/bm25_fresh"
    dense = RESULTS / "baselines/multihoprag/dense_fresh"
    evaluation = RESULTS / "evaluation/multihoprag/fresh"

    run("scripts.prepare_passages", "--corpus", RAW / "corpus.json")
    run("scripts.extract_entities")
    run("scripts.run_bm25", "--questions", questions, "--output-dir", bm25)
    run("scripts.run_dense", "--questions", questions, "--output-dir", dense)
    run("scripts.build_hierarchy")
    run("scripts.demonstrate_similarity")
    run("scripts.evaluate_baselines", "--questions", questions, "--bm25-dir", bm25,
        "--dense-dir", dense, "--output-dir", evaluation)
    run("scripts.plot_baselines", evaluation / "summary.json")
    run("scripts.answer_with_evidence", dense / "retrieval.jsonl")


if __name__ == "__main__":
    main()
