"""Selected-pair similarity for later clustering; never builds an all-pairs matrix.

Raw cosine lies in [-1, 1]; hybrid lies in [-alpha, 1]. The dissimilarity
1 - hybrid lies in [0, 1 + alpha] in exact arithmetic, but is NOT a metric:
triangle inequality and identity of indiscernibles are not guaranteed.
No cosine rescaling or clipping is performed.
"""
import numbers
import numpy as np

from ..retrieval.common import validate_passages
from ..retrieval.dense import DIMENSION, MODEL_NAME, file_hash, json_hash, load_cache
from .entities import validate_entity_records


def _alpha(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite number between 0 and 1")
    return float(alpha)


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("Expected a finite nonempty one-dimensional embedding")
    # Scale before taking a norm to avoid overflow for large finite inputs.
    scale = np.max(np.abs(vector))
    if scale == 0:
        return np.zeros_like(vector)
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def semantic_similarity(embedding_a, embedding_b):
    """Raw cosine; a zero-norm input returns 0. Cache validation rejects zeros."""
    a, b = _unit(embedding_a), _unit(embedding_b)
    if a.shape != b.shape:
        raise ValueError("Embedding dimensions must match")
    return float(a @ b)


def entity_similarity(entities_a, entities_b):
    """Jaccard of normalized strings; labels are metadata, not matching keys."""
    if isinstance(entities_a, str) or isinstance(entities_b, str):
        raise ValueError("Expected collections of normalized strings")
    a, b = set(entities_a), set(entities_b)
    if any(not isinstance(e, str) or not e for e in a | b):
        raise ValueError("Entity keys must be nonempty normalized strings")
    return len(a & b) / len(a | b) if a | b else 0.0


def hybrid_similarity(embedding_a, embedding_b, entities_a, entities_b, alpha=0.7):
    alpha = _alpha(alpha)
    return alpha * semantic_similarity(embedding_a, embedding_b) + (1 - alpha) * entity_similarity(entities_a, entities_b)


def hybrid_distance(embedding_a, embedding_b, entities_a, entities_b, alpha=0.7):
    """D_H = 1 - S_H; a dissimilarity, not a guaranteed metric distance."""
    return 1.0 - hybrid_similarity(embedding_a, embedding_b, entities_a, entities_b, alpha)


def load_passage_embeddings(cache_path, passages, *, dataset_sha256):
    """Read and validate existing BGE cache without constructing/loading an encoder."""
    rows = validate_passages(passages)
    expected = {
        "schema_version": 1, "passage_count": len(rows), "embedding_dimension": DIMENSION,
        "passage_ids": [p["passage_id"] for p in rows],
        "passage_records_sha256": json_hash(rows), "passage_file_sha256": dataset_sha256,
    }
    cached = load_cache(cache_path, expected)
    if cached is None:
        raise FileNotFoundError("BGE cache missing; use the existing dense cache workflow first")
    vectors, metadata = cached
    encoder = metadata.get("encoder", {})
    required = {"model_name": MODEL_NAME, "dimension": DIMENSION,
                "normalization": "L2", "dtype": "float32",
                "pooling": "last_hidden_state[:, 0, :] (CLS)", "passage_instruction": ""}
    if any(encoder.get(k) != v for k, v in required.items()):
        raise ValueError("Incompatible BGE encoder conventions")
    return vectors, metadata


class PassageSimilarity:
    """ID-based pair/batch API. Output order matches input pairs; memory O(batch).

    Constructor validates cache alignment and entity records. No BGE model loads.
    Use pairs([(id_a, id_b), ...], alpha=0.7), or pair(id_a, id_b).
    """
    def __init__(self, passages, entity_records, cache_path, *, dataset_sha256):
        self.passages = validate_passages(passages)
        records = validate_entity_records(entity_records, self.passages)
        self.embeddings, self.cache_metadata = load_passage_embeddings(
            cache_path, self.passages, dataset_sha256=dataset_sha256)
        self.indices = {p["passage_id"]: i for i, p in enumerate(self.passages)}
        self.entities = [frozenset(e["normalized"] for e in r["entities"]) for r in records]

    def pair(self, passage_a, passage_b, alpha=0.7):
        alpha = _alpha(alpha)
        i, j = self.indices[passage_a], self.indices[passage_b]
        semantic = semantic_similarity(self.embeddings[i], self.embeddings[j])
        entity = entity_similarity(self.entities[i], self.entities[j])
        hybrid = alpha * semantic + (1 - alpha) * entity
        return {"passage_id_a": passage_a, "passage_id_b": passage_b,
                "shared_entities": sorted(self.entities[i] & self.entities[j]),
                "semantic": semantic, "entity": entity, "alpha": alpha,
                "hybrid": hybrid, "dissimilarity": 1 - hybrid}

    def pairs(self, passage_pairs, alpha=0.7):
        _alpha(alpha)
        return [self.pair(a, b, alpha) for a, b in passage_pairs]
