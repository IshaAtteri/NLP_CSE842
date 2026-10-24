"""Flat BGE retrieval: CLS embeddings, L2 normalization, exhaustive cosine search."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from .common import make_result, validate_passages

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DIMENSION = 768
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_embeddings(vectors, count):
    if not isinstance(vectors, np.ndarray) or vectors.shape != (count, DIMENSION):
        raise ValueError(f"Expected embedding shape ({count}, {DIMENSION})")
    if vectors.dtype != np.float32:
        raise ValueError("Embeddings must have float32 dtype")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain nonfinite values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("Embeddings must be L2-normalized nonzero vectors")
    return vectors


class BGEEncoder:
    """One pretrained model instance, used for both passage and batched query encoding."""

    def __init__(self, *, revision="main", device="auto", cpu_threads=4,
                 passage_max_tokens=256, query_max_tokens=512, verbose=False):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        start = time.perf_counter()
        for value, name in ((cpu_threads, "cpu_threads"),
                            (passage_max_tokens, "passage_max_tokens"),
                            (query_max_tokens, "query_max_tokens")):
            positive_int(value, name)
        selected = "cuda" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device)
        self.device = torch.device(selected)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are cpu, cuda, cuda:<index>, or auto")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable in this PyTorch environment")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(0)
        torch.set_num_threads(cpu_threads)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False

        # Hub snapshot paths contain the resolved immutable commit, unlike a branch.
        config_path = hf_hub_download(MODEL_NAME, "config.json", revision=revision)
        resolved_revision = Path(config_path).parent.name
        if not re.fullmatch(r"[0-9a-f]{40}", resolved_revision):
            raise ValueError(f"Cannot resolve immutable model revision from {config_path}")
        config = AutoConfig.from_pretrained(MODEL_NAME, revision=resolved_revision)
        if config.hidden_size != DIMENSION:
            raise ValueError(f"Expected {DIMENSION}-dimensional BGE encoder")
        if not 2 < passage_max_tokens <= min(256, config.max_position_embeddings):
            raise ValueError("Passage limit must be between 3 and 256 tokens")
        if not 2 < query_max_tokens <= config.max_position_embeddings:
            raise ValueError("Query limit exceeds the encoder context or is too small")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=resolved_revision)
        self.tokenizer.padding_side = "right"  # CLS must stay at position zero.
        self.model = AutoModel.from_pretrained(
            MODEL_NAME, revision=resolved_revision, config=config,
            use_safetensors=True, dtype=torch.float32, attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        self.passage_max_tokens = passage_max_tokens
        self.query_max_tokens = query_max_tokens
        self.verbose = verbose
        self.last_encoding_stats = {}
        self.signature = {
            "model_name": MODEL_NAME, "resolved_revision": resolved_revision,
            "model_config": json.loads(json.dumps(config.to_dict())),
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_sha256": hashlib.sha256(
                self.tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest(),
            "padding_side": "right", "truncation": False,
            "pooling": "last_hidden_state[:, 0, :] (CLS)",
            "normalization": "L2", "dimension": DIMENSION, "dtype": "float32",
            "query_instruction": QUERY_INSTRUCTION, "passage_instruction": "",
            "passage_max_tokens": passage_max_tokens, "query_max_tokens": query_max_tokens,
            "attention_implementation": "sdpa",
            "packages": {p: version(p) for p in ("torch", "transformers", "tokenizers", "numpy")},
        }
        self.inference_settings = {
            "device": str(self.device), "cpu_threads": torch.get_num_threads(),
            "seed": 0, "deterministic_algorithms": True, "eval": True,
            "inference_mode": True, "autocast": False,
            "cuda_available": torch.cuda.is_available(),
            "hardware": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
        }
        self.load_seconds = time.perf_counter() - start

    def encode(self, texts, *, is_query=False, batch_size=16):
        import torch

        positive_int(batch_size, "batch_size")
        if isinstance(texts, str):
            raise ValueError("Expected a sequence of texts, not one string")
        texts = list(texts)
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Texts must be a nonempty sequence of nonempty strings")
        start = time.perf_counter()
        inputs = [QUERY_INSTRUCTION + text for text in texts] if is_query else texts
        tokens = self.tokenizer(inputs, padding=False, truncation=False,
                                add_special_tokens=True, return_attention_mask=True)
        lengths = [len(ids) for ids in tokens["input_ids"]]
        limit = self.query_max_tokens if is_query else self.passage_max_tokens
        oversized = [(i, n) for i, n in enumerate(lengths) if n > limit]
        if oversized:
            raise ValueError(
                f"{'Query including instruction' if is_query else 'Passage'} token limit "
                f"{limit} exceeded: (index, tokens)={oversized[:10]}; "
                f"{len(oversized)} total. No text truncated.")
        tokenization_seconds = time.perf_counter() - start
        embeddings = np.empty((len(texts), DIMENSION), dtype=np.float32)
        batch_times = []
        last_progress = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                batch_start = time.perf_counter()
                end = min(offset + batch_size, len(texts))
                features = [{key: value[i] for key, value in tokens.items()}
                            for i in range(offset, end)]
                batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                output = self.model(**batch, output_hidden_states=False,
                                    output_attentions=False, return_dict=True)
                cls = output.last_hidden_state[:, 0, :]
                normalized = torch.nn.functional.normalize(cls.float(), p=2, dim=1)
                # CPU copy synchronizes CUDA, so recorded durations include inference.
                embeddings[offset:end] = normalized.cpu().numpy()
                del output, cls, normalized, batch
                batch_times.append(time.perf_counter() - batch_start)
                now = time.perf_counter()
                if self.verbose and (now - last_progress >= 20 or end == len(texts)):
                    elapsed = now - start
                    eta = elapsed / end * (len(texts) - end)
                    print(f"Encoded {'queries' if is_query else 'passages'} "
                          f"{end}/{len(texts)}; {elapsed:.1f}s elapsed, {eta:.1f}s remaining",
                          flush=True)
                    last_progress = now
        validate_embeddings(embeddings, len(texts))
        self.last_encoding_stats = {
            "count": len(texts), "batch_size": batch_size, "batches": len(batch_times),
            "seconds": time.perf_counter() - start,
            "tokenization_seconds": tokenization_seconds, "batch_seconds": batch_times,
            "token_lengths": {"min": min(lengths), "max": max(lengths),
                              "mean": float(np.mean(lengths))},
        }
        return embeddings

    def encode_passages(self, texts, batch_size=16):
        return self.encode(texts, batch_size=batch_size)

    def encode_queries(self, texts, batch_size=32):
        return self.encode(texts, is_query=True, batch_size=batch_size)


def cache_metadata_path(path):
    path = Path(path)
    return path.with_name(path.stem + "_metadata.json")


def load_cache(path, expected):
    path = Path(path)
    metadata_path = cache_metadata_path(path)
    if not path.exists() and not metadata_path.exists():
        return None
    if not path.exists() or not metadata_path.exists():
        raise ValueError("Incomplete embedding cache; choose a new --cache path to regenerate.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Incompatible embedding cache ({key}); choose a new --cache path.")
    if metadata.get("embedding_file_sha256") != file_hash(path):
        raise ValueError("Embedding cache checksum mismatch; choose a new --cache path.")
    vectors = np.load(path, allow_pickle=False)
    validate_embeddings(vectors, expected["passage_count"])
    return vectors, metadata


def save_cache(path, vectors, metadata):
    path = Path(path)
    metadata_path = cache_metadata_path(path)
    if path.exists() or metadata_path.exists():
        raise FileExistsError("Embedding cache exists; choose a new --cache path.")
    validate_embeddings(vectors, metadata["passage_count"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    # Metadata is written last; interrupted/incomplete caches cannot be reused.
    complete = {**metadata, "embedding_file_sha256": file_hash(path)}
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(complete, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return complete


class DenseRetriever:
    def __init__(self, passages, *, encoder=None, cache_path=None, dataset_sha256=None,
                 batch_size=16, revision="main", device="auto", cpu_threads=4,
                 query_max_tokens=512, verbose=False):
        positive_int(batch_size, "batch_size")
        self.passages = validate_passages(passages)  # Keep original input ordering.
        self.encoder = encoder if encoder is not None else BGEEncoder(
            revision=revision, device=device, cpu_threads=cpu_threads,
            query_max_tokens=query_max_tokens, verbose=verbose)
        self._tie_ids = np.array([p["passage_id"] for p in self.passages])
        self.cache_expected = {
            "schema_version": 1, "passage_count": len(self.passages),
            "embedding_dimension": DIMENSION, "passage_ids": self._tie_ids.tolist(),
            "passage_records_sha256": json_hash(self.passages),
            "passage_file_sha256": dataset_sha256, "encoder": self.encoder.signature,
        }
        start = time.perf_counter()
        cached = load_cache(cache_path, self.cache_expected) if cache_path is not None else None
        self.cache_load_seconds = time.perf_counter() - start if cache_path is not None else 0.0
        self.cache_hit = cached is not None
        self.passage_encoding_seconds = 0.0
        self.cache_write_seconds = 0.0
        if cached is not None:
            self.embeddings, self.cache_metadata = cached
            self.passage_encoding_stats = self.cache_metadata["generation"]["encoding"]
        else:
            start = time.perf_counter()
            self.embeddings = self.encoder.encode_passages(
                [p["text"] for p in self.passages], batch_size=batch_size)
            validate_embeddings(self.embeddings, len(self.passages))
            self.passage_encoding_seconds = time.perf_counter() - start
            self.passage_encoding_stats = dict(self.encoder.last_encoding_stats)
            self.cache_metadata = {
                **self.cache_expected,
                "generation": {"inference": self.encoder.inference_settings,
                               "batch_size": batch_size,
                               "encoding": self.passage_encoding_stats},
            }
            if cache_path is not None:
                start = time.perf_counter()
                self.cache_metadata = save_cache(cache_path, self.embeddings, self.cache_metadata)
                self.cache_write_seconds = time.perf_counter() - start

    def search_embedding(self, embedding, top_k=50):
        """Exact CPU dot products over all cached vectors; query encoding is separate."""
        positive_int(top_k, "top_k")
        query = np.asarray(embedding)
        if query.shape != (DIMENSION,):
            raise ValueError(f"Expected a one-dimensional query vector of length {DIMENSION}")
        validate_embeddings(query.reshape(1, -1), 1)
        scores = self.embeddings @ query
        if not np.isfinite(scores).all():
            raise ValueError("Similarity scores are nonfinite")
        order = np.lexsort((self._tie_ids, -scores))[:top_k]
        return [make_result(self.passages[int(i)], scores[i], rank)
                for rank, i in enumerate(order, start=1)]

    def search(self, query, top_k=50):
        positive_int(top_k, "top_k")
        vector = self.encoder.encode_queries([query], batch_size=1)[0]
        return self.search_embedding(vector, top_k)

    def search_many(self, queries, top_k=50, batch_size=32):
        positive_int(top_k, "top_k")
        vectors = self.encoder.encode_queries(queries, batch_size=batch_size)
        return [self.search_embedding(vector, top_k) for vector in vectors]
