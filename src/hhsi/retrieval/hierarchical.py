"""	Use beam search to find passages in the HHSI hierarchy.

This code takes a loaded hierarchy, a query embedding, and normalized query entities. Model loading and index building happen separately. It does not use gold evidence. Final passage reranking is handled in another file.
"""
from dataclasses import dataclass

import numpy as np

from ..indexing.similarity import _alpha, _unit, entity_similarity


@dataclass(frozen=True)
class BeamCandidate:
    passage_id: str
    node_id: str
    score: float
    semantic: float
    entity: float


@dataclass(frozen=True)
class BeamSearchOutput:
    candidates: tuple[BeamCandidate, ...]
    beam_width: int
    alpha: float
    rounds: int
    nodes_scored: int
    internal_nodes_expanded: int
    nodes_pruned: int
    trace: tuple[dict, ...]


class HierarchicalBeamSearcher:
    """Use the same checked index to search for different queries.
    In each round, expand the internal nodes in the beam into their children. Keep any leaf nodes as candidates too, since some branches end earlier than others. Keep the best beam_width nodes and repeat until only leaf nodes remain.   

    Score each node by comparing it with the query. Scores are not added along the path. Cluster embeddings and combined entity sets help guide the search, but a removed branch could contain a better passage. A beam wide enough to keep all passages searches the whole tree.

    First load the saved index using indexing.hierarchy.load_hierarchy() so its file hashes, cluster embeddings, and entities are checked. This constructor also checks the tree structure. Do not change the index after creating the searcher.
    """

    def __init__(self, index):
        self.index = index
        self.nodes = index.node_map()
        self.positions = {n.node_id: i for i, n in enumerate(index.nodes)}
        if not index.nodes or len(self.nodes) != len(index.nodes):
            raise ValueError("Hierarchy must have nonempty, unique nodes")
        if index.root_id not in self.nodes:
            raise ValueError("Hierarchy root is missing")
        vectors = np.asarray(index.centroids)
        if (vectors.ndim != 2 or vectors.shape[0] != len(index.nodes)
                or vectors.shape[1] == 0 or not np.isfinite(vectors).all()):
            raise ValueError("Invalid centroid array")
        self.dimension = vectors.shape[1]
        root = self.nodes[index.root_id]
        if root.parent_id is not None:
            raise ValueError("Root must have no parent")

        # Use a stack to check the tree, including very deep branches.
        seen, leaves = set(), []
        stack = [index.root_id]
        while stack:
            key = stack.pop()
            if key in seen:
                raise ValueError("Hierarchy contains a cycle or repeated child")
            seen.add(key)
            node = self.nodes[key]
            if any(not isinstance(e, str) or not e for e in node.entities):
                raise ValueError("Node entities must be nonempty normalized strings")
            if node.child_ids:
                if node.passage_id is not None or len(node.child_ids) != 2:
                    raise ValueError("Internal nodes need two children and no passage")
                children = []
                for child_id in node.child_ids:
                    child = self.nodes.get(child_id)
                    if child is None or child.parent_id != key:
                        raise ValueError("Invalid child or parent reference")
                    children.append(child)
                if node.passage_count != sum(c.passage_count for c in children):
                    raise ValueError("Incorrect descendant passage count")
                stack.extend(node.child_ids)
            else:
                if (not isinstance(node.passage_id, str) or not node.passage_id
                        or node.passage_count != 1):
                    raise ValueError("Each leaf must identify exactly one passage")
                leaves.append(node.passage_id)
        if seen != set(self.nodes):
            raise ValueError("Hierarchy contains unreachable nodes")
        if (len(leaves) != len(set(leaves))
                or len(index.passage_ids) != len(leaves)
                or set(leaves) != set(index.passage_ids)):
            raise ValueError("Hierarchy passage coverage is inconsistent")
        self.passage_count = len(leaves)
        self.default_alpha = _alpha(index.config["alpha"])

    def search(self, query_embedding, query_entities=(), *, beam_width=64,
               alpha=None, include_trace=False):
        """Return up to beam_width different passages, sorted from highest to lowest score.

        Normalize the query entities with indexing.entities.normalize_entity() before calling this method.
        If the query has no entities, its entity similarity is zero. Keep the same alpha value.
        To select the final top 50 passages later, use a beam width of at least 50, such as 64.   
        By default, use the alpha value used to build the index. Passing a different alpha changes the search scores but does not rebuild the hierarchy..
        """
        if type(beam_width) is not int or beam_width <= 0:
            raise ValueError("beam_width must be a positive integer")
        weight = self.default_alpha if alpha is None else _alpha(alpha)
        vector = np.asarray(query_embedding, dtype=np.float64)
        if (vector.shape != (self.dimension,) or not np.isfinite(vector).all()
                or not np.any(vector)):
            raise ValueError(f"Query must be a finite nonzero vector of length {self.dimension}")
        query = _unit(vector)
        if isinstance(query_entities, (str, bytes)):
            raise ValueError("query_entities must be a collection, not a string")
        entities = frozenset(query_entities)
        if any(not isinstance(e, str) or not e for e in entities):
            raise ValueError("Query entity keys must be nonempty normalized strings")

        width = min(beam_width, self.passage_count)
        scores = {}

        def score(key):
           # Calculate each visited node's score only once for this query.
            if key not in scores:
                node = self.nodes[key]
                centroid = self.index.centroids[self.positions[key]]
                semantic = float(query @ _unit(centroid))
                entity = entity_similarity(entities, node.entities)
                scores[key] = (weight * semantic + (1 - weight) * entity,
                               semantic, entity)
            return scores[key]

        def order(key):
            node = self.nodes[key]
            # Use passage IDs to break ties between leaves, as the evaluator expects.
            tie_id = node.passage_id if node.passage_id is not None else node.node_id
            return (-score(key)[0], tie_id, node.node_id)

        beam = [self.index.root_id]
        score(beam[0])
        rounds = expanded = pruned = 0
        trace = []
        while any(self.nodes[key].child_ids for key in beam):
            options = []
            for key in beam:
                children = self.nodes[key].child_ids
                if children:
                    expanded += 1
                    options.extend(children)
                else:
                    options.append(key)
            options.sort(key=order)
            beam = options[:width]
            removed = len(options) - len(beam)
            pruned += removed
            rounds += 1
            if include_trace:
                trace.append({
                    "round": rounds, "options": len(options), "pruned": removed,
                    "retained": [{"node_id": key, "score": score(key)[0],
                                  "is_leaf": not self.nodes[key].child_ids}
                                 for key in beam],
                })

        candidates = tuple(
            BeamCandidate(self.nodes[key].passage_id, key, *score(key))
            for key in sorted(beam, key=order)
        )
        if len(candidates) != width:
            raise RuntimeError("Traversal did not produce the expected candidate count")
        return BeamSearchOutput(candidates, beam_width, weight, rounds, len(scores),
                                expanded, pruned, tuple(trace))
