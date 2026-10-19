import json
from pathlib import Path

import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "multihoprag"
corpus = json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))
questions = json.loads((DATA_DIR / "MultiHopRAG.json").read_text(encoding="utf-8"))

print(f"Dataset has {len(corpus)} news articles and {len(questions)} questions.")
# print(json.dumps(questions[0], indent=2))

hotpot_dir = DATA_DIR.parent / "hotpotqa" / "distractor"
for split in ("train", "validation"):
    files = sorted(hotpot_dir.glob(f"{split}-*.parquet"))
    count = sum(pq.ParquetFile(file).metadata.num_rows for file in files)
    print(f"HotpotQA {split} has {count} questions.")

file = next(hotpot_dir.glob("validation-*.parquet"))
row = next(pq.ParquetFile(file).iter_batches(batch_size=1)).to_pylist()[0]
print("HotpotQA first question:", row["question"])
print("Source paragraphs:", len(row["context"]["title"]))
# print(json.dumps(row, indent=2))
