"""Synthetic 768-D fixtures; no model downloads or corpus required."""
import copy
import json

import numpy as np
import pytest

from src.hhsi.indexing.hierarchy import (
    build_hierarchy, compare_hierarchies, condensed_distances, configuration,
    fingerprint, hierarchy_statistics, load_hierarchy, save_hierarchy, validate_hierarchy,
)
from src.hhsi.indexing.similarity import hybrid_distance, semantic_similarity
from src.hhsi.retrieval.dense import DIMENSION, file_hash


def fixture():
    rows = [{"passage_id": f"p{i}", "doc_id": f"d{i}", "url": f"u{i}", "text": "text"}
            for i in range(4)]
    vectors = np.zeros((4, DIMENSION), dtype=np.float32)
    vectors[:, :2] = [[1, 0], [.99, .1], [.8, .6], [-1, 0]]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    records = [{"passage_id": p["passage_id"], "doc_id": p["doc_id"],
                "entities": [{"text": e, "label": "ORG", "normalized": e}]}
               for p, e in zip(rows, ["a", "b", "a", "c"])]
    return rows, vectors, records


def build(alpha=0.7):
    rows, vectors, records = fixture()
    index, _ = build_hierarchy(rows, vectors, records, alpha=alpha, block_size=2)
    return index, rows, vectors, records


def test_known_merges_and_hybrid_influence():
    semantic, rows, vectors, records = build(1)
    hybrid, *_ = build(.7)
    assert semantic.nodes[4].child_ids == ["node_000000", "node_000001"]
    assert hybrid.nodes[4].child_ids == ["node_000000", "node_000002"]
    assert semantic.nodes[4].merge_distance == pytest.approx(
        1 - semantic_similarity(vectors[0], vectors[1]))
    assert semantic.nodes[5].merge_distance == pytest.approx(
        (2 - semantic_similarity(vectors[0], vectors[2])
         - semantic_similarity(vectors[1], vectors[2])) / 2)
    comparison = compare_hierarchies(semantic, hybrid, vectors, records, cuts=(2, 4))
    assert comparison["cluster_memberships_differ"]
    assert comparison["immediate_parent_membership_changed_passages"] == 3
    assert comparison["semantic"]["cuts"][-1]["pair_weighted_mean_cosine"] is None


@pytest.mark.parametrize("alpha", [0, .5, .7, 1])
def test_blocked_distances_equal_phase3(alpha):
    _, vectors, records = fixture()
    entities = [set(e["normalized"] for e in r["entities"]) for r in records]
    distances, report = condensed_distances(vectors, entities, alpha=alpha, block_size=2)
    expected = [hybrid_distance(vectors[i], vectors[j], entities[i], entities[j], alpha)
                for i in range(4) for j in range(i + 1, 4)]
    np.testing.assert_allclose(distances, expected, atol=1e-12, rtol=1e-12)
    assert report["pair_count"] == 6


def test_all_centroids_entities_paths_counts_and_depths():
    index, rows, vectors, records = build()
    for i, node in enumerate(index.nodes):
        leaves = [n for n in index.iter_subtree(node.node_id) if n.passage_id is not None]
        positions = [index.passage_ids.index(n.passage_id) for n in leaves]
        summed = vectors[positions].astype(np.float64).sum(axis=0)
        expected = summed / np.linalg.norm(summed) if np.linalg.norm(summed) else summed
        np.testing.assert_allclose(index.centroids[i], expected, atol=1e-12)
        assert node.passage_count == len(leaves)
        assert node.entities == set().union(*(n.entities for n in leaves))
        if node.passage_id:
            path = index.path_to_passage(node.passage_id)
            assert path[0] == index.root_id and path[-1] == node.node_id
            assert len(path) == node.depth + 1
    stats = hierarchy_statistics(index)
    assert stats["total_nodes"] == 7 and stats["leaf_nodes"] == 4
    assert stats["branching_factor_internal_nodes"]["mean"] == 2
    assert validate_hierarchy(index, rows, vectors, records)
    with pytest.raises(KeyError):
        index.path_to_passage("absent")


def test_deterministic_ties_and_empty_entities():
    rows, vectors, records = fixture()
    vectors[:] = vectors[0]
    for record in records:
        record["entities"] = []
    a, _ = build_hierarchy(rows, vectors, records, alpha=0)
    b, _ = build_hierarchy(rows, vectors, records, alpha=0)
    assert fingerprint(a) == fingerprint(b)
    assert all(not node.entities for node in a.nodes)
    assert all(node.merge_distance == 1 for node in a.nodes[4:])
    assert compare_hierarchies(a, b, vectors, records, cuts=(2,))[
        "immediate_parent_membership_changed_passages"] == 0


def test_single_passage_and_zero_sum_centroid():
    rows, vectors, records = fixture()
    single, _ = build_hierarchy(rows[:1], vectors[:1], records[:1])
    assert len(single.nodes) == 1 and single.nodes[0].depth == 0
    assert single.path_to_passage("p0") == [single.root_id]
    vectors[1] = -vectors[0]
    paired, report = build_hierarchy(rows[:2], vectors[:2], records[:2])
    assert np.count_nonzero(paired.centroids[-1]) == 0
    assert report["zero_centroid_nodes"] == 1


@pytest.mark.parametrize("alpha", [-1, 1.01, float("nan"), float("inf"), True, None, "0.7"])
def test_invalid_alpha(alpha):
    rows, vectors, records = fixture()
    with pytest.raises(ValueError):
        build_hierarchy(rows, vectors, records, alpha=alpha)


@pytest.mark.parametrize("kind", ["dimension", "zero", "nan", "dtype", "entity_order", "passage_order", "empty"])
def test_invalid_inputs(kind):
    rows, vectors, records = fixture()
    if kind == "dimension":
        vectors = vectors[:, :2]
    elif kind == "zero":
        vectors[0] = 0
    elif kind == "nan":
        vectors[0, 0] = np.nan
    elif kind == "dtype":
        vectors = vectors.astype(np.float64)
    elif kind == "entity_order":
        records.reverse()
    elif kind == "passage_order":
        rows.reverse()
    else:
        rows, records, vectors = [], [], vectors[:0]
    with pytest.raises(ValueError):
        build_hierarchy(rows, vectors, records)


@pytest.mark.parametrize("kind", [
    "duplicate_id", "missing_leaf", "duplicate_passage", "parent", "child", "cycle",
    "depth", "count", "centroid", "entity_union", "root", "centroid_dimension",
])
def test_tree_corruption(kind):
    index, rows, vectors, records = build()
    if kind == "duplicate_id":
        index.nodes[0].node_id = index.nodes[1].node_id
    elif kind == "missing_leaf":
        index.nodes.pop(0)
    elif kind == "duplicate_passage":
        index.nodes[0].passage_id = index.nodes[1].passage_id
    elif kind == "parent":
        index.nodes[0].parent_id = None
    elif kind == "child":
        index.nodes[-1].child_ids[0] = "absent"
    elif kind == "cycle":
        index.nodes[-1].child_ids[0] = index.root_id
    elif kind == "depth":
        index.nodes[0].depth += 1
    elif kind == "count":
        index.nodes[-1].passage_count += 1
    elif kind == "centroid":
        index.centroids[-1] = 0
    elif kind == "entity_union":
        index.nodes[-1].entities = frozenset()
    elif kind == "root":
        index.root_id = index.nodes[0].node_id
    else:
        index.centroids = index.centroids[:, :2]
    with pytest.raises(ValueError):
        validate_hierarchy(index, rows, vectors, records)


def test_serialization_roundtrip_no_overwrite(tmp_path):
    index, rows, vectors, records = build()
    target = tmp_path / "index"
    size = save_hierarchy(index, target, rows, vectors, records)
    restored = load_hierarchy(target, rows, vectors, records, expected_config=index.config)
    assert fingerprint(restored) == fingerprint(index)
    assert size == sum(p.stat().st_size for p in target.iterdir())
    with pytest.raises(FileExistsError):
        save_hierarchy(index, target, rows, vectors, records)
    with pytest.raises(ValueError):
        load_hierarchy(target, rows, vectors, records, expected_config=configuration(1, 2))
    with pytest.raises(ValueError):
        load_hierarchy(target, rows, vectors, records, expected_provenance={"wrong": 1})
    altered = copy.deepcopy(rows)
    altered[0]["text"] = "changed"
    with pytest.raises(ValueError):
        load_hierarchy(target, altered, vectors, records)


@pytest.mark.parametrize("kind", ["checksum", "schema", "signature", "config", "missing", "tree_with_checksum"])
def test_corrupt_serialized_metadata(tmp_path, kind):
    index, rows, vectors, records = build()
    target = tmp_path / "index"
    save_hierarchy(index, target, rows, vectors, records)
    path = target / "metadata.json"
    meta = json.loads(path.read_text())
    if kind == "checksum":
        (target / "centroids.npy").write_bytes(b"broken")
    elif kind == "missing":
        path.unlink()
    elif kind == "tree_with_checksum":
        tree = json.loads((target / "tree.json").read_text())
        tree["nodes"][0]["passage_id"] = "wrong"
        (target / "tree.json").write_text(json.dumps(tree))
        meta["files"]["tree.json"] = file_hash(target / "tree.json")
        path.write_text(json.dumps(meta))
    else:
        if kind == "schema":
            meta["schema_version"] = -1
        elif kind == "signature":
            meta["input_signature"]["embedding_array_sha256"] = "wrong"
        else:
            meta["config"]["alpha"] = 2
        path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_hierarchy(target, rows, vectors, records)


def test_memory_budget():
    _, vectors, records = fixture()
    entities = [set() for _ in records]
    with pytest.raises(MemoryError):
        condensed_distances(vectors, entities, max_working_mib=1)


def test_cohesion_matches_explicit_pair_means():
    from src.hhsi.indexing.hierarchy import cohesion_at_cut
    from src.hhsi.indexing.similarity import entity_similarity
    index, _, vectors, records = build(1)
    # The known two-cluster cut is {p0,p1,p2}, {p3}.
    stats = cohesion_at_cut(index, vectors, records, 2)
    pairs = [(0, 1), (0, 2), (1, 2)]
    entities = [set(e["normalized"] for e in r["entities"]) for r in records]
    assert stats["within_cluster_unordered_pairs"] == 3
    assert stats["cluster_sizes"]["min"] == 1
    assert stats["cluster_sizes"]["max"] == 3
    assert stats["pair_weighted_mean_cosine"] == pytest.approx(
        np.mean([semantic_similarity(vectors[i], vectors[j]) for i, j in pairs]))
    assert stats["pair_weighted_mean_entity_jaccard"] == pytest.approx(
        np.mean([entity_similarity(entities[i], entities[j]) for i, j in pairs]))
