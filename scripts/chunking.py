import json
from pathlib import Path

from src.hhsi.data.chunking import chunk_text, tokenizer

CORPUS_PATH = Path("data/raw/corpus.json")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

for doc_id, article in enumerate(corpus[:5]):
    chunks = chunk_text(article["body"])

    lengths = [
        len(tokenizer.encode(chunk, verbose=False))
        for chunk in chunks
    ]

    print(f"\nArticle {doc_id}: {article['title']}")
    print(f"Chunks: {len(chunks)}")
    print(f"Min tokens: {min(lengths)}")
    print(f"Max tokens: {max(lengths)}")

    assert chunks, "Article produced no chunks"
    assert all(0 < length <= 256 for length in lengths)

print("\nAll 5 articles passed.")