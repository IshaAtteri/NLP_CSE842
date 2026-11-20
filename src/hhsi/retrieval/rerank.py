"""Calculate hybrid scores for the beam search candidates and return the top-k passages in the format needed for evaluation.

This reranker uses a similarity formula rather than a cross-encoder model. If beam search uses the same formula and alpha, the passage order should usually stay the same.
"""
from dataclasses import dataclass

import numpy as np

from ..indexing.entities import validate_entity_records
from ..indexing.similarity import _alpha, _unit, hybrid_similarity
from .common import make_result, validate_passages
from .dense import DIMENSION, positive_int, validate_embeddings
from .hierarchical import BeamSearchOutput, HierarchicalBeamSearcher


class HybridPassageReranker:
    """Calculate scores only for the given candidate passages using their original embeddings.

    Use the same checked passages, embeddings, and entity records that were used to build the index.
    Remove repeated passage IDs. Keep different passages from the same article because the evaluator counts passage positions when applying its cutoffs.
    """

    def __init__(self, passages, embeddings, entity_records):
        self.passages = validate_passages(passages)
        self.embeddings = validate_embeddings(embeddings, len(self.passages))
        records = validate_entity_records(entity_records, self.passages)
        self.positions = {p["passage_id"]: i for i, p in enumerate(self.passages)}
        self.entities = [frozenset(e["normalized"] for e in record["entities"])
                         for record in records]

    def rerank(self, query_embedding, query_entities, candidates, *, top_k=50,
               alpha=0.7):
        """Return result dictionaries sorted from highest to lowest score. Use passage IDs to break ties.

        Accept beam search output, a collection of candidate objects, or a collection of passage IDs. Calculate fresh scores instead of using the saved candidate scores.
        Raise an error if there are too few candidates. Do not add passages from outside the candidate list.
        """
        positive_int(top_k, "top_k")
        weight = _alpha(alpha)
        query = np.asarray(query_embedding, dtype=np.float64)
        if (query.shape != (DIMENSION,) or not np.isfinite(query).all()
                or not np.any(query)):
            raise ValueError(f"Expected a finite nonzero query vector of length {DIMENSION}")
        query = _unit(query)
        if isinstance(query_entities, (str, bytes)):
            raise ValueError("query_entities must be a collection of normalized names")
        entities = frozenset(query_entities)
        if any(not isinstance(e, str) or not e for e in entities):
            raise ValueError("Query entity keys must be nonempty normalized strings")
        if isinstance(candidates, BeamSearchOutput):
            candidates = candidates.candidates
        if isinstance(candidates, (str, bytes)):
            raise ValueError("Pass candidates as a collection, not a single string")

        unique = set()
        for candidate in candidates:
            key = candidate if isinstance(candidate, str) else getattr(candidate, "passage_id", None)
            if not isinstance(key, str) or key not in self.positions:
                raise ValueError(f"Unknown or invalid candidate passage ID: {key!r}")
            unique.add(key)
        needed = min(top_k, len(self.passages))
        if len(unique) < needed:
            raise ValueError(
                f"Need {needed} unique candidates, got {len(unique)}. "
                "Increase beam_width or request a smaller top_k.")

        scored = []
        for key in sorted(unique):
            i = self.positions[key]
            score = hybrid_similarity(query, self.embeddings[i], entities,
                                      self.entities[i], alpha=weight)
            scored.append((key, score))
        scored.sort(key=lambda row: (-row[1], row[0]))
        return [make_result(self.passages[self.positions[key]], score, rank)
                for rank, (key, score) in enumerate(scored[:needed], start=1)]


@dataclass(frozen=True)
class RetrievalOutput:
    results: list[dict]
    beam: BeamSearchOutput
    rerank_alpha: float


class HierarchicalRetriever:
    """Run beam search, then rerank its candidates to select the final top-k passages. Query embeddings and entities must already be available.

    Before creating this retriever, load the saved index using load_hierarchy(directory, passages, embeddings, entity_records).Use the same retriever for different questions. Do not change its input index or arrays after creating it.
    """ 

    def __init__(self, index, passages, embeddings, entity_records):
        self.reranker = HybridPassageReranker(passages, embeddings, entity_records)
        self.searcher = HierarchicalBeamSearcher(index)
        if index.passage_ids != [p["passage_id"] for p in self.reranker.passages]:
            raise ValueError("Index and passage order differ")
        # Check that leaf embeddings and entities match the inputs, even when passage IDs match.
        leaves = {n.passage_id: n for n in index.nodes if n.passage_id is not None}
        for i, passage in enumerate(self.reranker.passages):
            leaf = leaves[passage["passage_id"]]
            centroid = index.centroids[self.searcher.positions[leaf.node_id]]
            if (leaf.entities != self.reranker.entities[i]
                    or not np.allclose(centroid, _unit(embeddings[i]), rtol=1e-10, atol=1e-12)):
                raise ValueError("Index leaf vectors/entities differ from reranking inputs")

    def search_embedding(self, query_embedding, query_entities=(), *, top_k=50,
                         beam_width=128, alpha=None, rerank_alpha=None,
                         include_trace=False):
        """Find and select passages without using the correct answers or gold evidence.

        By default, use the index's alpha for beam search and the same value for reranking. Different alpha values change the scores but do not rebuild the index.
        """
        positive_int(top_k, "top_k")
        positive_int(beam_width, "beam_width")
        needed = min(top_k, self.searcher.passage_count)
        if beam_width < needed:
            raise ValueError(f"beam_width must be at least {needed} for this top_k")
        weight = self.searcher.default_alpha if alpha is None else _alpha(alpha)
        final_weight = weight if rerank_alpha is None else _alpha(rerank_alpha)
        if isinstance(query_entities, (str, bytes)):
            raise ValueError("query_entities must be a collection of normalized names")
        entities = frozenset(query_entities)  # Store query entities once so beam search and reranking can both use them.
        beam = self.searcher.search(query_embedding, entities, beam_width=beam_width,
                                    alpha=weight, include_trace=include_trace)
        results = self.reranker.rerank(query_embedding, entities, beam,
                                      top_k=top_k, alpha=final_weight)
        return RetrievalOutput(results, beam, final_weight)
