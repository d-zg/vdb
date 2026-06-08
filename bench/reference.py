#!/usr/bin/env python3
"""Reference benchmark: FAISS and hnswlib on an ann-benchmarks dataset.

Reproduces the exact protocol of vdb_bench (src/bench_main.cpp) so the
numbers are directly comparable to the C++ scoreboard:

  - same dataset file, same first-N queries, same k
  - one query at a time (no batching) — matches ann-benchmarks' default mode
  - single-threaded search, enforced
  - recall@k measured against the file's precomputed `neighbors` ground truth
  - QPS = num_queries / wall-clock over the whole sweep (build time excluded)

Usage:
  .venv/bin/python bench/reference.py [--dataset PATH] [--queries N] [--k K]
                                      [--ours QPS [--ours-recall R]]

Pass --ours with the QPS from ./build/release/vdb_bench to see vdb as a
percentage of each reference line.
"""

import argparse
import time

import faiss
import h5py
import hnswlib
import numpy as np


def load_dataset(path: str, k: int, max_queries: int):
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int64)
    if max_queries:
        test = test[:max_queries]
        neighbors = neighbors[:max_queries]
    assert k <= neighbors.shape[1], "k exceeds ground-truth depth"
    return train, test, neighbors[:, :k]


def recall_at_k(got: np.ndarray, truth: np.ndarray) -> float:
    """Mean |got ∩ truth| / k over all queries. Order-insensitive."""
    assert got.shape == truth.shape
    hits = sum(len(np.intersect1d(g, t)) for g, t in zip(got, truth))
    return hits / truth.size


def sweep(search_one, test: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Run queries one at a time, returning (ids matrix, elapsed seconds)."""
    ids = np.empty((len(test), k), dtype=np.int64)
    t0 = time.perf_counter()
    for i, q in enumerate(test):
        ids[i] = search_one(q, k)
    return ids, time.perf_counter() - t0


def bench_faiss_flat(train, test, truth, k):
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(train.shape[1])
    index.add(train)
    build_s = time.perf_counter() - t0

    def search_one(q, k):
        _, ids = index.search(q.reshape(1, -1), k)
        return ids[0]

    ids, elapsed = sweep(search_one, test, k)
    return {
        "name": "faiss IndexFlatL2 (exact)",
        "build_s": build_s,
        "recall": recall_at_k(ids, truth),
        "qps": len(test) / elapsed,
    }


def bench_hnswlib(train, test, truth, k, M=16, ef_construction=200, ef_search=(10, 20, 40, 80, 160)):
    t0 = time.perf_counter()
    index = hnswlib.Index(space="l2", dim=train.shape[1])
    index.init_index(max_elements=len(train), M=M, ef_construction=ef_construction)
    index.add_items(train, num_threads=1)
    build_s = time.perf_counter() - t0

    rows = []
    for ef in ef_search:
        index.set_ef(ef)

        def search_one(q, k):
            ids, _ = index.knn_query(q, k=k, num_threads=1)
            return ids[0]

        ids, elapsed = sweep(search_one, test, k)
        rows.append({
            "name": f"hnswlib (M={M}, ef={ef})",
            "build_s": build_s,
            "recall": recall_at_k(ids, truth),
            "qps": len(test) / elapsed,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/fashion-mnist-784-euclidean.hdf5")
    ap.add_argument("--queries", type=int, default=1000,
                    help="number of queries (0 = all)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ours", type=float, default=None,
                    help="vdb QPS from ./build/release/vdb_bench, for %% comparison")
    ap.add_argument("--ours-recall", type=float, default=1.0)
    args = ap.parse_args()

    faiss.omp_set_num_threads(1)  # single-threaded, like vdb_bench

    train, test, truth = load_dataset(args.dataset, args.k, args.queries)
    print(f"dataset: {args.dataset}")
    print(f"{len(train)} base vectors, {len(test)} queries, "
          f"dim={train.shape[1]}, k={args.k}, single-threaded\n")

    results = [bench_faiss_flat(train, test, truth, args.k)]
    results += bench_hnswlib(train, test, truth, args.k)
    if args.ours is not None:
        results.append({"name": "vdb (ours)", "build_s": float("nan"),
                        "recall": args.ours_recall, "qps": args.ours})

    flat_qps = results[0]["qps"]
    print(f"{'algorithm':<30} {'build':>7} {'recall@'+str(args.k):>10} "
          f"{'qps':>10} {'vs faiss-flat':>13}")
    for r in sorted(results, key=lambda r: r["qps"]):
        rel = r["qps"] / flat_qps
        print(f"{r['name']:<30} {r['build_s']:>6.1f}s {r['recall']:>10.4f} "
              f"{r['qps']:>10.1f} {rel:>12.1%}")

    if args.ours is not None:
        print(f"\nvdb is at {args.ours / flat_qps:.2%} of faiss-flat "
              f"(the exact-search target)")
        best = max((r for r in results if r["name"].startswith("hnswlib")),
                   key=lambda r: r["qps"])
        print(f"vdb is at {args.ours / best['qps']:.3%} of {best['name']}")


if __name__ == "__main__":
    main()
