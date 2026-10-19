from pathlib import Path

from huggingface_hub import snapshot_download

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "multihoprag"

if not DATA_DIR.exists():
    print("Downloading MultiHop-RAG...")
    snapshot_download(
        repo_id="yixuantt/MultiHopRAG",
        repo_type="dataset",
        local_dir=DATA_DIR,
    )
    print(f"Downloaded to {DATA_DIR}")
else:
    print(f"Dataset already exists at {DATA_DIR}")

HOTPOT_DIR = DATA_DIR.parent / "hotpotqa"
print("Downloading/checking HotpotQA (distractor)...") # 10 Wikipedia paragraphs per question, rather than entire articles
snapshot_download(
    repo_id="hotpotqa/hotpot_qa",
    repo_type="dataset",
    local_dir=HOTPOT_DIR,
    allow_patterns=["distractor/*.parquet", "README.md"],
)
print(f"HotpotQA saved to {HOTPOT_DIR}")
