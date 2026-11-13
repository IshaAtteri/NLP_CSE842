"""Exact average-linkage indexes; construction and structural inspection only."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import psutil
from scipy.cluster.hierarchy import linkage
from scipy.sparse import csr_matrix

from ..retrieval.common import validate_passages
from ..retrieval.dense import DIMENSION, file_hash, json_hash, positive_int, validate_embeddings
from .entities import validate_entity_records
from .similarity import _alpha, _unit, entity_similarity

SCHEMA_VERSION = 1
ROUNDOFF_TOLERANCE = 1e-12


def node_id(i):
    return f"node_{i:06d}"


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def input_signature(passages, embeddings, records):
    return {
        "passage_records_sha256": json_hash(passages),
        "entity_records_sha256": json_hash(records),
        "embedding_array_sha256": array_hash(embeddings),
        "embedding_shape": list(embeddings.shape), "embedding_dtype": str(embeddings.dtype),
    }


def configuration(alpha=0.7, block_size=128):
    alpha = _alpha(alpha)
    positive_int(block_size, "block_size")
    return {
        "algorithm": "exact_average_linkage", "alpha": alpha, "block_size": block_size,
        "stopping": "single root; one passage per leaf", "random_seed": None,
        "distance": "1 - (alpha * raw_cosine + (1-alpha) * entity_jaccard)",
        "roundoff_tolerance": ROUNDOFF_TOLERANCE,
        "tie_policy": "SciPy nearest-neighbor-chain with fixed input order; children by minimum input row",
        "centroid": "L2-normalized sum of original descendant passage embeddings; exact zero sum stays zero",
        "packages": {p: version(p) for p in ("numpy", "scipy")},
    }


def _inputs(passages, embeddings, records):
    passages = validate_passages(passages)
    validate_embeddings(embeddings, len(passages))
    validate_entity_records(records, passages)
    entities = [frozenset(e["normalized"] for e in r["entities"]) for r in records]
    return passages, entities


def memory_estimate(count, block_size=128):
    """Conservative incremental numeric working-set estimate, not measured process RSS."""
    pairs = count * (count - 1) // 2
    return {
        "pair_count": pairs, "condensed_bytes": 8 * pairs,
        "estimated_working_bytes": (
            16 * pairs + 4 * count * DIMENSION * 8
            + 5 * min(count, block_size) * count * 8 + 128 * 1024**2),
    }


def _incidence(entities):
    vocabulary = {e: i for i, e in enumerate(sorted(set().union(*entities)))}
    columns, offsets = [], [0]
    for entity_set in entities:
        columns.extend(sorted(vocabulary[e] for e in entity_set))
        offsets.append(len(columns))
    # int32 avoids the overflow possible with uint8 mention intersections.
    return csr_matrix((np.ones(len(columns), dtype=np.int32), columns, offsets),
                      shape=(len(entities), len(vocabulary)), dtype=np.int32)


def condensed_distances(embeddings, entities, *, alpha=0.7, block_size=128,
                        max_working_mib=1024, progress=None):
    """Exact vectorized Phase 3 formula, in bounded blocks and condensed storage.

    Only negative dissimilarities within 1e-12 of zero are set to zero for SciPy.
    This is reported roundoff repair, not a cosine transformation.
    """
    alpha = _alpha(alpha)
    positive_int(block_size, "block_size")
    positive_int(max_working_mib, "max_working_mib")
    count = len(entities)
    validate_embeddings(embeddings, count)
    estimate = memory_estimate(count, block_size)
    available = psutil.virtual_memory().available
    if estimate["estimated_working_bytes"] > min(max_working_mib * 1024**2, int(available * 0.75)):
        raise MemoryError(f"Estimated working set {estimate['estimated_working_bytes']} bytes "
                          f"exceeds budget or available-memory safety limit ({available} bytes available)")
    unit = embeddings.astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    incidence = _incidence(entities) if alpha < 1 else None
    sizes = np.array([len(e) for e in entities])
    result = np.empty(estimate["pair_count"], dtype=np.float64)
    repairs = 0
    for start in range(0, count, block_size):
        end = min(count, start + block_size)
        scores = alpha * (unit[start:end] @ unit.T)
        if incidence is not None:
            intersection = (incidence[start:end] @ incidence.T).toarray()
            union = sizes[start:end, None] + sizes[None, :] - intersection
            jaccard = np.divide(intersection, union, out=np.zeros_like(scores), where=union != 0)
            scores += (1 - alpha) * jaccard
        for local, i in enumerate(range(start, end)):
            distance = 1 - scores[local, i + 1:]
            if not np.isfinite(distance).all() or np.any(distance < -ROUNDOFF_TOLERANCE):
                raise ValueError("Invalid hybrid dissimilarity")
            negative = distance < 0
            repairs += int(negative.sum())
            distance[negative] = 0
            offset = count * i - i * (i + 1) // 2
            result[offset:offset + len(distance)] = distance
        if progress:
            progress(end, count)
    return result, {**estimate, "available_memory_before_bytes": available,
                    "roundoff_repairs": repairs}


@dataclass
class Node:
    node_id: str
    depth: int
    parent_id: str | None
    child_ids: list[str]
    passage_count: int
    entities: frozenset[str]
    passage_id: str | None
    merge_distance: float


@dataclass
class Hierarchy:
    nodes: list[Node]
    centroids: np.ndarray
    root_id: str
    passage_ids: list[str]
    config: dict
    signature: dict
    provenance: dict

    def node_map(self):
        return {n.node_id: n for n in self.nodes}

    def iter_subtree(self, root_id=None):
        """Structural DFS only; no query scoring or retrieval policy."""
        lookup = self.node_map()
        stack = [self.root_id if root_id is None else root_id]
        while stack:
            node = lookup[stack.pop()]
            yield node
            stack.extend(reversed(node.child_ids))

    def path_to_passage(self, passage_id):
        """Parent-chain inspection, not query retrieval."""
        lookup = self.node_map()
        leaf = next((n for n in self.nodes if n.passage_id == passage_id), None)
        if leaf is None:
            raise KeyError(passage_id)
        path = [leaf.node_id]
        while lookup[path[-1]].parent_id is not None:
            path.append(lookup[path[-1]].parent_id)
        return path[::-1]


def build_hierarchy(passages, embeddings, records, *, alpha=0.7, block_size=128,
                    max_working_mib=1024, provenance=None, progress=None):
    start = time.perf_counter()
    passages, entities = _inputs(passages, embeddings, records)
    config = configuration(alpha, block_size)
    distances, details = condensed_distances(
        embeddings, entities, alpha=alpha, block_size=block_size,
        max_working_mib=max_working_mib, progress=progress)
    distance_seconds = time.perf_counter() - start
    tick = time.perf_counter()
    count = len(passages)
    merges = linkage(distances, method="average", optimal_ordering=False) if count > 1 else np.empty((0, 4))
    del distances
    linkage_seconds = time.perf_counter() - tick
    total = 2 * count - 1
    sums = np.zeros((total, DIMENSION), dtype=np.float64)
    sums[:count] = embeddings
    min_row = list(range(count))
    nodes = [Node(node_id(i), -1, None, [], 1, entities[i], p["passage_id"], 0.0)
             for i, p in enumerate(passages)]
    for offset, merge in enumerate(merges):
        i = count + offset
        a, b = sorted((int(merge[0]), int(merge[1])), key=lambda j: min_row[j])
        sums[i] = sums[a] + sums[b]
        nodes[a].parent_id = nodes[b].parent_id = node_id(i)
        children = [node_id(a), node_id(b)]
        nodes.append(Node(node_id(i), -1, None, children,
                          nodes[a].passage_count + nodes[b].passage_count,
                          nodes[a].entities | nodes[b].entities, None, float(merge[2])))
        min_row.append(min(min_row[a], min_row[b]))
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    centroids = np.divide(sums, norms, out=np.zeros_like(sums), where=norms != 0)
    del sums
    index = Hierarchy(nodes, centroids, node_id(total - 1),
                      [p["passage_id"] for p in passages], config,
                      input_signature(passages, embeddings, records), provenance or {})
    lookup = index.node_map()
    stack = [(index.root_id, 0)]
    while stack:
        current, depth = stack.pop()
        lookup[current].depth = depth
        stack.extend((child, depth + 1) for child in lookup[current].child_ids)
    construction_seconds = time.perf_counter() - start
    tick = time.perf_counter()
    validate_hierarchy(index, passages, embeddings, records)
    details.update({
        "construction_seconds": construction_seconds, "distance_seconds": distance_seconds,
        "linkage_seconds": linkage_seconds, "validation_seconds": time.perf_counter() - tick,
        "zero_centroid_nodes": int(np.count_nonzero(np.linalg.norm(centroids, axis=1) == 0)),
    })
    return index, details


def validate_hierarchy(index, passages, embeddings, records, *, expected_config=None):
    """Check every edge, leaf, count, centroid and entity union against source inputs."""
    passages, entities = _inputs(passages, embeddings, records)
    count, total = len(passages), len(index.nodes)
    if index.config != configuration(index.config.get("alpha"), index.config.get("block_size")):
        raise ValueError("Unsupported or incompatible construction configuration/package versions")
    if expected_config is not None and index.config != expected_config:
        raise ValueError("Unexpected construction configuration")
    if index.signature != input_signature(passages, embeddings, records):
        raise ValueError("Incompatible source signatures")
    if index.passage_ids != [p["passage_id"] for p in passages]:
        raise ValueError("Passage order mismatch")
    ids = [n.node_id for n in index.nodes]
    if total != 2 * count - 1 or ids != [node_id(i) for i in range(total)]:
        raise ValueError("Invalid, duplicate or noncanonical node IDs/count")
    if index.root_id != ids[-1]:
        raise ValueError("Invalid root ID")
    if (index.centroids.shape != (total, DIMENSION) or index.centroids.dtype != np.float64
            or not np.isfinite(index.centroids).all()):
        raise ValueError("Invalid centroid dimensions/dtype/values")
    positions = {key: i for i, key in enumerate(ids)}
    parents = Counter()
    sums = np.zeros_like(index.centroids)
    min_row = []
    for i, node in enumerate(index.nodes):
        if not isinstance(node.depth, int) or isinstance(node.depth, bool) or node.depth < 0:
            raise ValueError("Invalid node depth")
        if not isinstance(node.passage_count, int) or isinstance(node.passage_count, bool):
            raise ValueError("Invalid descendant count")
        if not np.isfinite(node.merge_distance) or node.merge_distance < 0:
            raise ValueError("Invalid merge dissimilarity")
        if i < count:
            if (node.child_ids or node.passage_id != passages[i]["passage_id"]
                    or node.passage_count != 1 or node.entities != entities[i]
                    or node.merge_distance != 0):
                raise ValueError("Invalid leaf coverage/entity/count")
            sums[i] = embeddings[i]
            min_row.append(i)
        else:
            if len(node.child_ids) != 2 or len(set(node.child_ids)) != 2 or node.passage_id is not None:
                raise ValueError("Internal node must have exactly two distinct children")
            if any(child not in positions or positions[child] >= i for child in node.child_ids):
                raise ValueError("Invalid child reference or cycle")
            a, b = (positions[c] for c in node.child_ids)
            if min_row[a] >= min_row[b]:
                raise ValueError("Noncanonical child ordering")
            left, right = index.nodes[a], index.nodes[b]
            if (left.parent_id != node.node_id or right.parent_id != node.node_id
                    or left.depth != node.depth + 1 or right.depth != node.depth + 1):
                raise ValueError("Parent-child/depth mismatch")
            if node.passage_count != left.passage_count + right.passage_count:
                raise ValueError("Incorrect descendant count")
            if node.entities != left.entities | right.entities:
                raise ValueError("Incorrect entity union")
            if node.merge_distance + ROUNDOFF_TOLERANCE < max(left.merge_distance, right.merge_distance):
                raise ValueError("Nonmonotonic average-linkage merge distances")
            sums[i] = sums[a] + sums[b]
            min_row.append(min_row[a])
            parents.update(node.child_ids)
        if not np.allclose(index.centroids[i], _unit(sums[i]), rtol=1e-10, atol=1e-12):
            raise ValueError("Incorrect descendant-weighted centroid")
    root = index.nodes[-1]
    if root.parent_id is not None or root.depth != 0 or root.passage_count != count:
        raise ValueError("Invalid root parent/depth/count")
    if any(parents[k] != 1 for k in ids[:-1]) or parents[index.root_id] != 0:
        raise ValueError("Each non-root node needs exactly one parent")
    seen = set()
    for node in index.iter_subtree():
        if node.node_id in seen:
            raise ValueError("Cycle or multiple parents")
        seen.add(node.node_id)
    if seen != set(ids):
        raise ValueError("Unreachable nodes")
    return True


def tree_payload(index):
    vocabulary = sorted(index.nodes[-1].entities)
    keys = {e: i for i, e in enumerate(vocabulary)}
    return {
        "root_id": index.root_id, "passage_ids": index.passage_ids, "entity_vocabulary": vocabulary,
        "nodes": [{
            "node_id": n.node_id, "depth": n.depth, "parent_id": n.parent_id,
            "child_ids": n.child_ids, "passage_count": n.passage_count,
            "entity_ids": sorted(keys[e] for e in n.entities),
            "passage_id": n.passage_id, "merge_distance": n.merge_distance,
        } for n in index.nodes],
    }


def fingerprint(index):
    return {"tree_sha256": json_hash(tree_payload(index)),
            "centroids_sha256": array_hash(index.centroids)}


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def save_hierarchy(index, directory, passages, embeddings, records):
    validate_hierarchy(index, passages, embeddings, records)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tree_path, vectors_path = directory / "tree.json", directory / "centroids.npy"
    _write_json(tree_path, tree_payload(index))
    with vectors_path.open("xb") as stream:
        np.save(stream, index.centroids, allow_pickle=False)
    metadata = {
        "schema_version": SCHEMA_VERSION, "config": index.config,
        "input_signature": index.signature, "provenance": index.provenance,
        "fingerprint": fingerprint(index),
        "files": {p.name: file_hash(p) for p in (tree_path, vectors_path)},
    }
    # Last file is the completion marker. No pickle or executable serialized objects.
    _write_json(directory / "metadata.json", metadata)
    return sum(p.stat().st_size for p in directory.iterdir())


def load_hierarchy(directory, passages, embeddings, records, *, expected_config=None,
                   expected_provenance=None):
    directory = Path(directory)
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported index schema")
        if set(metadata["files"]) != {"tree.json", "centroids.npy"}:
            raise ValueError("Invalid index file manifest")
        for name, digest in metadata["files"].items():
            if file_hash(directory / name) != digest:
                raise ValueError(f"Index checksum mismatch: {name}")
        if expected_provenance is not None and metadata["provenance"] != expected_provenance:
            raise ValueError("Incompatible input-file provenance")
        tree = json.loads((directory / "tree.json").read_text(encoding="utf-8"))
        vocabulary = tree["entity_vocabulary"]
        if vocabulary != sorted(set(vocabulary)) or any(not isinstance(e, str) or not e for e in vocabulary):
            raise ValueError("Invalid entity vocabulary")
        nodes = []
        for row in tree["nodes"]:
            keys = row["entity_ids"]
            if (any(type(k) is not int or not 0 <= k < len(vocabulary) for k in keys)
                    or keys != sorted(set(keys))):
                raise ValueError("Invalid entity references")
            nodes.append(Node(row["node_id"], row["depth"], row["parent_id"],
                              row["child_ids"], row["passage_count"],
                              frozenset(vocabulary[k] for k in keys), row["passage_id"],
                              row["merge_distance"]))
        index = Hierarchy(nodes, np.load(directory / "centroids.npy", allow_pickle=False),
                          tree["root_id"], tree["passage_ids"], metadata["config"],
                          metadata["input_signature"], metadata["provenance"])
        validate_hierarchy(index, passages, embeddings, records, expected_config=expected_config)
        if fingerprint(index) != metadata["fingerprint"]:
            raise ValueError("Index fingerprint mismatch")
        return index
    except (KeyError, TypeError, IndexError, OSError) as error:
        raise ValueError(f"Incomplete or malformed index: {error}") from error


def _summary(values):
    return {"min": int(min(values)), "max": int(max(values)), "mean": float(np.mean(values)),
            "median": float(np.median(values))} if values else None


def hierarchy_statistics(index):
    leaves = [n for n in index.nodes if n.passage_id is not None]
    internal = [n for n in index.nodes if n.child_ids]
    levels = defaultdict(list)
    for node in index.nodes:
        levels[node.depth].append(node.passage_count)
    return {
        "total_nodes": len(index.nodes), "leaf_nodes": len(leaves), "internal_nodes": len(internal),
        "leaf_depth": _summary([n.depth for n in leaves]),
        "branching_factor_internal_nodes": _summary([len(n.child_ids) for n in internal]),
        "cluster_sizes_by_depth": {str(k): {"nodes": len(v), **_summary(v)}
                                   for k, v in sorted(levels.items())},
    }


def _membership_masks(index):
    """Temporary exact membership bitsets; no descendant lists persisted in the index."""
    masks = {}
    for i, node in enumerate(index.nodes):
        masks[node.node_id] = (1 << i) if node.passage_id is not None else (
            masks[node.child_ids[0]] | masks[node.child_ids[1]])
    return masks


def _cut(index, k):
    positive_int(k, "cut_clusters")
    k = min(k, len(index.passage_ids))
    positions = {n.node_id: i for i, n in enumerate(index.nodes)}
    frontier = {-positions[index.root_id]}
    heap = list(frontier)
    # Undo the latest merges to obtain exactly k disjoint clusters.
    while len(frontier) < k:
        key = heapq.heappop(heap)
        frontier.remove(key)
        for child in index.nodes[-key].child_ids:
            value = -positions[child]
            frontier.add(value)
            heapq.heappush(heap, value)
    return [index.nodes[-key].node_id for key in sorted(frontier)]


def cohesion_at_cut(index, embeddings, records, k):
    groups = _cut(index, k)
    lookup = {pid: i for i, pid in enumerate(index.passage_ids)}
    sets = [set(e["normalized"] for e in r["entities"]) for r in records]
    unit = embeddings.astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    pair_count, cosine_sum, entity_sum = 0, 0.0, 0.0
    sizes = []
    for group in groups:
        members = [lookup[n.passage_id] for n in index.iter_subtree(group) if n.passage_id is not None]
        sizes.append(len(members))
        pairs = len(members) * (len(members) - 1) // 2
        if pairs:
            summed = unit[members].sum(axis=0)
            cosine_sum += float((summed @ summed - len(members)) / 2)
            entity_sum += sum(entity_similarity(sets[a], sets[b])
                              for offset, a in enumerate(members) for b in members[offset + 1:])
            pair_count += pairs
    return {
        "requested_clusters": k, "actual_clusters": len(groups), "cluster_sizes": _summary(sizes),
        "within_cluster_unordered_pairs": pair_count,
        "pair_weighted_mean_cosine": cosine_sum / pair_count if pair_count else None,
        "pair_weighted_mean_entity_jaccard": entity_sum / pair_count if pair_count else None,
    }


def compare_hierarchies(semantic, hybrid, embeddings, records, cuts=(64, 256, 1024)):
    if semantic.signature != hybrid.signature or semantic.passage_ids != hybrid.passage_ids:
        raise ValueError("Cannot compare indexes with different inputs")
    a_config, b_config = dict(semantic.config), dict(hybrid.config)
    a_config.pop("alpha")
    b_config.pop("alpha")
    if a_config != b_config:
        raise ValueError("Construction settings differ beyond alpha")
    a, b = _membership_masks(semantic), _membership_masks(hybrid)
    count = len(semantic.passage_ids)
    changed = sum(a.get(semantic.nodes[i].parent_id, 1 << i) !=
                  b.get(hybrid.nodes[i].parent_id, 1 << i) for i in range(count))
    a_clusters = {a[n.node_id] for n in semantic.nodes if n.child_ids}
    b_clusters = {b[n.node_id] for n in hybrid.nodes if n.child_ids}
    return {
        "cluster_memberships_differ": a_clusters != b_clusters,
        "immediate_parent_membership_changed_passages": changed,
        "immediate_parent_membership_changed_fraction": changed / count,
        "shared_internal_memberships": len(a_clusters & b_clusters),
        "semantic_only_internal_memberships": len(a_clusters - b_clusters),
        "hybrid_only_internal_memberships": len(b_clusters - a_clusters),
        "cohesion_definition": "Exact mean over unordered distinct passage pairs within disjoint k-cluster cuts; weighted by pair count, singleton clusters contribute no pairs",
        "semantic": {"structure": hierarchy_statistics(semantic),
                     "cuts": [cohesion_at_cut(semantic, embeddings, records, k) for k in cuts]},
        "hybrid": {"structure": hierarchy_statistics(hybrid),
                   "cuts": [cohesion_at_cut(hybrid, embeddings, records, k) for k in cuts]},
    }
