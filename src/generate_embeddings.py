import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer



ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "embeddings"

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Starting settings: up to 384 tokens per passage, with 64 tokens repeated between neighboring passages.
CHUNK_TOKENS = 384
OVERLAP_TOKENS = 64


def save_json(path, value):
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def make_passages(model, article_id, text, metadata):
    """Split source text into passages and preserve source information."""
    tokenized = model.tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )

    offsets = tokenized["offset_mapping"]
    passages = []
    start = 0

    while start < len(offsets):
        end = min(start + CHUNK_TOKENS, len(offsets))

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        passage_text = text[char_start:char_end]

        
        token_count = len(
            model.tokenizer(
                passage_text,
                truncation=False,
            )["input_ids"]
        )

        if token_count > model.max_seq_length:
            raise ValueError(
                f"Passage from {article_id} exceeds the model's "
                "input limit. Reduce CHUNK_TOKENS."
            )

        passages.append({
            "passage_id": f"{article_id}_p{len(passages):04d}",
            "article_id": article_id,
            "text": passage_text,
            "char_start": char_start,
            "char_end": char_end,
            **metadata,
        })

        if end == len(offsets):
            break

        start = end - OVERLAP_TOKENS

    return passages


def prepare_multihoprag(model):
    """Prepare passages from the MultiHopRAG article corpus."""
    corpus_path = DATA_DIR / "multihoprag" / "corpus.json"

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {corpus_path}\n"
            "Run download_data.py first."
        )

    with corpus_path.open("r", encoding="utf-8") as file:
        corpus = json.load(file)

    passages = []

    for index, article in enumerate(corpus):
        article_id = f"mhr_{index:04d}"

        # Only article content and metadata are indexed.
        # Gold answers and evidence annotations are not included.
        text = (
            f"Title: {article['title']}\n"
            f"Source: {article.get('source') or ''}\n"
            f"Published: {article.get('published_at') or ''}\n\n"
            f"{article['body']}"
        )

        metadata = {
            "title": article["title"],
            "source": article.get("source"),
            "published_at": article.get("published_at"),
            "url": article.get("url"),
        }

        passages.extend(
            make_passages(model, article_id, text, metadata)
        )

    print(f"Loaded {len(corpus)} MultiHopRAG articles.")
    print(f"Created {len(passages)} passages.")

    return passages


def prepare_hotpotqa(model, split, limit):
    """Prepare HotpotQA passages and preserve per-question candidates."""
    folder = DATA_DIR / "hotpotqa" / "distractor"
    files = sorted(folder.glob(f"{split}-*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No {split} Parquet files found in {folder}\n"
            "Run download_data.py first."
        )

    passages = []
    paragraph_to_passages = {}
    question_candidates = {}
    question_count = 0

    for file in files:
        parquet_file = pq.ParquetFile(file)

        for batch in parquet_file.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                if limit and question_count >= limit:
                    print(f"Prepared {question_count} HotpotQA questions.")
                    return passages, question_candidates

                candidate_ids = []

                titles = row["context"]["title"]
                sentence_lists = row["context"]["sentences"]

                for title, sentences in zip(titles, sentence_lists):
                    # To reuse embeddings for identical paragraphs.
                    identity = json.dumps(
                        [title, sentences],
                        ensure_ascii=False,
                    )

                    digest = hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest()

                    article_id = f"hp_{digest}"

                    if article_id not in paragraph_to_passages:
                        body = " ".join(
                            sentence.strip() for sentence in sentences
                        )
                        text = f"Title: {title}\n\n{body}"

                        chunks = make_passages(
                            model,
                            article_id,
                            text,
                            {
                                "title": title,
                                "sentences": sentences,
                            },
                        )

                        passages.extend(chunks)

                        paragraph_to_passages[article_id] = [
                            passage["passage_id"]
                            for passage in chunks
                        ]

                    candidate_ids.extend(
                        paragraph_to_passages[article_id]
                    )

                # Preserve the candidate set for each question.
                question_candidates[row["id"]] = list(
                    dict.fromkeys(candidate_ids)
                )

                question_count += 1

                if question_count % 1000 == 0:
                    print(f"Prepared {question_count} HotpotQA questions...")

    print(f"Prepared {question_count} HotpotQA questions.")
    return passages, question_candidates


def generate_and_save(model, name, passages, candidates=None):
    """Generate embeddings and save aligned output files."""
    if not passages:
        raise ValueError(f"No passages were prepared for {name}.")

    folder = OUTPUT_DIR / name
    folder.mkdir(parents=True, exist_ok=True)

    passage_ids = [
        passage["passage_id"] for passage in passages
    ]

    if len(passage_ids) != len(set(passage_ids)):
        raise ValueError("Duplicate passage IDs found.")

    # To save text and IDs before the embedding computation.
    with (folder / "passages.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for passage in passages:
            file.write(
                json.dumps(passage, ensure_ascii=False) + "\n"
            )

    save_json(folder / "passage_ids.json", passage_ids)

    if candidates is not None:
        save_json(
            folder / "question_candidates.json",
            candidates,
        )

    print(f"\nGenerating embeddings for {len(passages)} passages...")

    embeddings = model.encode(
        [passage["text"] for passage in passages],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    #To validate the generated vectors.
    if embeddings.shape != (len(passages), 768):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numbers.")

    if not np.allclose(
        np.linalg.norm(embeddings, axis=1),
        1.0,
        atol=1e-4,
    ):
        raise ValueError("Embeddings are not correctly normalized.")

    np.save(
        folder / "passage_embeddings.npy",
        embeddings,
    )

    save_json(folder / "config.json", {
        "model": MODEL_NAME,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "normalized_embeddings": True,
        "passage_count": len(passages),
        "embedding_dimension": embeddings.shape[1],
    })

    print("\nCompleted successfully!")
    print("Embedding shape:", embeddings.shape)
    print("Output folder:", folder)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["multihoprag", "hotpotqa"],
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="validation",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="HotpotQA question limit; 0 processes the entire split.",
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer.")

    print("Loading BGE-base:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    print("Model device:", model.device)

    if args.dataset == "multihoprag":
        passages = prepare_multihoprag(model)

        generate_and_save(
            model,
            "multihoprag",
            passages,
        )

    else:
        passages, candidates = prepare_hotpotqa(
            model,
            args.split,
            args.limit,
        )

        suffix = f"_first{args.limit}" if args.limit else ""
        output_name = f"hotpotqa_{args.split}{suffix}"

        generate_and_save(
            model,
            output_name,
            passages,
            candidates,
        )


if __name__ == "__main__":
    main()