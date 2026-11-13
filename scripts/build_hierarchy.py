"""Build and validate complete semantic and hybrid hierarchies; no retrieval."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import psutil

from src.hhsi.indexing.entities import NORMALIZATION, load_entity_records
from src.hhsi.indexing.hierarchy import (
    _write_json, build_hierarchy, compare_hierarchies, configuration, fingerprint,
    hierarchy_statistics, load_hierarchy, memory_estimate, save_hierarchy,
)
from src.hhsi.indexing.similarity import _alpha, hybrid_distance, load_passage_embeddings
from src.hhsi.retrieval.common import load_passages
from src.hhsi.retrieval.dense import cache_metadata_path, file_hash

DEFAULT_ROOT = Path("data/processed/indexes")


def load_inputs(passages_path, cache_path, entity_path):
    passages = load_passages(passages_path)
    records = load_entity_records(entity_path, passages)
    report_path = Path(entity_path).with_suffix(".report.json")
    extraction = json.loads(report_path.read_text(encoding="utf-8"))
    if (extraction["input_passage_dataset_sha256"] != file_hash(passages_path)
            or extraction["entity_file_sha256"] != file_hash(entity_path)
            or extraction["passages_processed"] != len(passages)
            or extraction["normalization"] != NORMALIZATION):
        raise ValueError("Entity report is incompatible with passage/entity inputs")
    embeddings, metadata = load_passage_embeddings(
        cache_path, passages, dataset_sha256=file_hash(passages_path))
    source_paths = [Path(passages_path), Path(cache_path), cache_metadata_path(cache_path),
                    Path(entity_path), report_path]
    code_paths = [Path(__file__), Path("src/hhsi/indexing/hierarchy.py"),
                  Path("src/hhsi/indexing/similarity.py"), Path("src/hhsi/indexing/entities.py"),
                  Path("src/hhsi/retrieval/dense.py"), Path("src/hhsi/retrieval/common.py")]
    provenance = {
        "files": {str(p.resolve()): file_hash(p) for p in source_paths},
        "code": {str(p.resolve()): file_hash(p) for p in code_paths},
        "encoder": metadata["encoder"], "normalization": NORMALIZATION,
        "entity_model": {"name": extraction["model_name"], "version": extraction["model_version"]},
    }
    return passages, embeddings, records, provenance


def run(args):
    alpha = _alpha(args.alpha)
    variants = [("semantic", 1.0), ("hybrid_alpha_" + str(alpha).replace(".", "_"), alpha)]
    if not args.validate_only and args.output_root.exists():
        raise FileExistsError("Output root exists; use --validate-only or a new --output-root")
    passages, embeddings, records, provenance = load_inputs(args.passages, args.cache, args.entities)
    if len(passages) != args.expected_passages:
        raise ValueError(f"Expected {args.expected_passages} passages, got {len(passages)}")
    if args.validate_only:
        for name, weight in variants:
            loaded = load_hierarchy(
                args.output_root / name, passages, embeddings, records,
                expected_config=configuration(weight, args.block_size),
                expected_provenance=provenance)
            print(f"Reload validated: {name}, {len(loaded.nodes)} nodes, {len(passages)} passages", flush=True)
            del loaded
        return
    print(json.dumps({"memory_plan": memory_estimate(len(passages), args.block_size),
                      "available_memory_bytes": psutil.virtual_memory().available}), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=False)
    indexes, reports = [], {}
    for name, weight in variants:
        last_progress = [time.perf_counter()]
        def progress(done, total):
            now = time.perf_counter()
            if done == total or now - last_progress[0] >= 20:
                print(f"{name}: pairwise rows {done}/{total}", flush=True)
                last_progress[0] = now
        print(f"Building {name}: alpha={weight}", flush=True)
        index, details = build_hierarchy(
            passages, embeddings, records, alpha=weight, block_size=args.block_size,
            max_working_mib=args.max_working_mib, provenance=provenance, progress=progress)
        if args.verify_determinism:
            start = time.perf_counter()
            repeated, _ = build_hierarchy(
                passages, embeddings, records, alpha=weight, block_size=args.block_size,
                max_working_mib=args.max_working_mib, provenance=provenance, progress=progress)
            if fingerprint(index) != fingerprint(repeated):
                raise ValueError("Repeated construction was not deterministic")
            details["determinism_rebuild_seconds"] = time.perf_counter() - start
            details["determinism_verified"] = True
            del repeated
            gc.collect()
        else:
            details["determinism_verified"] = False
        directory = args.output_root / name
        start = time.perf_counter()
        size = save_hierarchy(index, directory, passages, embeddings, records)
        save_seconds = time.perf_counter() - start
        start = time.perf_counter()
        reloaded = load_hierarchy(
            directory, passages, embeddings, records,
            expected_config=configuration(weight, args.block_size),
            expected_provenance=provenance)
        if fingerprint(reloaded) != fingerprint(index):
            raise ValueError("Reload differs from the constructed index")
        details["reload_validation_seconds"] = time.perf_counter() - start
        del reloaded
        report = {
            **details, **hierarchy_statistics(index), "config": index.config,
            "fingerprint": fingerprint(index), "index_storage_bytes": size,
            "storage_definition": "tree.json + centroids.npy + metadata.json, excluding construction report",
            "save_and_validation_seconds": save_seconds,
        }
        _write_json(directory / "construction.report.json", report)
        reports[name] = {**report, "directory_storage_bytes": sum(p.stat().st_size for p in directory.iterdir())}
        indexes.append(index)
        print(f"Saved {name}: {details['construction_seconds']:.2f}s, {size} bytes; reload valid", flush=True)
        gc.collect()
    print("Comparing memberships and exact within-cluster cohesion", flush=True)
    start = time.perf_counter()
    comparison = compare_hierarchies(*indexes, embeddings, records)
    comparison["comparison_seconds"] = time.perf_counter() - start
    if not comparison["cluster_memberships_differ"]:
        entity_sets = [{e["normalized"] for e in r["entities"]} for r in records]
        gaps = [abs(hybrid_distance(embeddings[i], embeddings[i+1], entity_sets[i], entity_sets[i+1], alpha)
                    - hybrid_distance(embeddings[i], embeddings[i+1], entity_sets[i], entity_sets[i+1], 1.0))
                for i in range(len(passages) - 1)]
        comparison["identical_tree_diagnostic"] = {
            "adjacent_pair_distances_changed": sum(g > 1e-12 for g in gaps),
            "maximum_distance_change": max(gaps, default=0),
            "note": "Different distances can retain the same merge order; alpha has not been adjusted",
        }
    # Confirm source artifacts stayed unchanged for the entire run.
    if any(file_hash(path) != digest for path, digest in provenance["files"].items()):
        raise ValueError("Source artifacts changed during construction")
    _write_json(args.output_root / "comparison.json", comparison)
    _write_json(args.output_root / "construction_summary.json", {
        "variants": reports, "comparison_file_sha256": file_hash(args.output_root / "comparison.json"),
        "passages": len(passages), "provenance": provenance,
        "scope": "Index construction and structural comparison only; no retrieval evaluation",
    })
    print(json.dumps({
        "changed_parent_passages": comparison["immediate_parent_membership_changed_passages"],
        "changed_parent_fraction": comparison["immediate_parent_membership_changed_fraction"],
        "cluster_memberships_differ": comparison["cluster_memberships_differ"],
        "comparison_seconds": comparison["comparison_seconds"],
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, default=Path("data/processed/multihoprag_passages.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/embeddings/multihoprag_bge_base_v1_5.npy"))
    parser.add_argument("--entities", type=Path, default=Path("data/processed/entities/multihoprag_entities.jsonl"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-working-mib", type=int, default=1024)
    parser.add_argument("--expected-passages", type=int, default=6638)
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
