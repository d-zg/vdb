#pragma once

#include "vdb/vector_store.hpp"

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <span>
#include <vector>

namespace vdb {

// --- Timing -----------------------------------------------------------------
// A minimal RAII-friendly stopwatch over a monotonic clock. Construction (or
// reset()) marks t0; elapsed() reports time since. steady_clock is used
// because it never jumps backwards (unlike system_clock), which is exactly
// what you want for measuring durations.
class Stopwatch {
public:
    using clock = std::chrono::steady_clock;

    Stopwatch() noexcept : start_(clock::now()) {}

    void reset() noexcept { start_ = clock::now(); }

    [[nodiscard]] std::chrono::duration<double> elapsed() const noexcept {
        return clock::now() - start_;
    }

    [[nodiscard]] double elapsed_seconds() const noexcept {
        return elapsed().count();
    }

private:
    clock::time_point start_;
};

// --- Dataset ----------------------------------------------------------------
// A benchmark dataset: a base set to index, a query set to search with, and
// the precomputed exact k-NN ("ground truth") for each query. Vectors are
// stored flat (row-major) for cache-friendliness; the accessors hand back a
// non-owning span over the right row.
struct Dataset {
    std::size_t dim = 0;
    std::size_t k = 0;  // neighbours per query in ground_truth
    std::vector<float> base;   // num_base * dim
    std::vector<float> query;  // num_query * dim
    std::vector<std::vector<VectorStore::Id>> ground_truth;  // per query

    [[nodiscard]] std::size_t num_base() const noexcept { return base.size() / dim; }
    [[nodiscard]] std::size_t num_query() const noexcept { return query.size() / dim; }

    [[nodiscard]] std::span<const float> base_vec(std::size_t i) const {
        return {base.data() + i * dim, dim};
    }
    [[nodiscard]] std::span<const float> query_vec(std::size_t i) const {
        return {query.data() + i * dim, dim};
    }
};

// Generates a reproducible dataset of uniform-random vectors and computes
// exact ground truth via a brute-force flat scan. `seed` makes runs
// repeatable so benchmark numbers are comparable across code changes.
[[nodiscard]] inline Dataset make_random_dataset(std::size_t num_base,
                                                 std::size_t num_query,
                                                 std::size_t dim,
                                                 std::size_t k,
                                                 std::uint64_t seed = 42) {
    Dataset ds;
    ds.dim = dim;
    ds.k = k;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    ds.base.resize(num_base * dim);
    for (auto& x : ds.base) x = dist(rng);

    ds.query.resize(num_query * dim);
    for (auto& x : ds.query) x = dist(rng);

    // Build an exact oracle from the flat store and record true k-NN.
    VectorStore oracle(dim);
    for (std::size_t i = 0; i < num_base; ++i) {
        oracle.add(ds.base_vec(i));
    }
    ds.ground_truth.reserve(num_query);
    for (std::size_t q = 0; q < num_query; ++q) {
        ds.ground_truth.push_back(oracle.search(ds.query_vec(q), k));
    }
    return ds;
}

// --- Recall -----------------------------------------------------------------
// recall@k = |returned ∩ truth| / |truth|. Order within each list is ignored;
// we only care which ids were found. Both lists are assumed to address the
// same k (the caller passes matching k's).
[[nodiscard]] inline double recall_at_k(std::span<const VectorStore::Id> got,
                                        std::span<const VectorStore::Id> truth) {
    if (truth.empty()) return 1.0;

    std::vector<VectorStore::Id> a(got.begin(), got.end());
    std::vector<VectorStore::Id> b(truth.begin(), truth.end());
    std::ranges::sort(a);
    std::ranges::sort(b);

    std::vector<VectorStore::Id> common;
    std::ranges::set_intersection(a, b, std::back_inserter(common));

    return static_cast<double>(common.size()) / static_cast<double>(truth.size());
}

// --- The runner -------------------------------------------------------------
// Any index that can answer a k-NN query satisfies this concept. The flat
// VectorStore models it today; IVF / HNSW will model it later with zero
// changes to the benchmark code below.
template <typename T>
concept VectorIndex = requires(const T idx, std::span<const float> q, std::size_t k) {
    { idx.search(q, k) } -> std::same_as<std::vector<VectorStore::Id>>;
};

struct BenchmarkResult {
    double recall = 0.0;          // mean recall@k across queries, 0..1
    double qps = 0.0;             // queries per second
    double mean_latency_ms = 0.0; // mean per-query latency
    std::size_t num_query = 0;
};

// Runs every query in `ds` against `index`, timing the whole sweep and
// averaging recall. Templated on the concept so it stays index-agnostic.
template <VectorIndex Index>
[[nodiscard]] BenchmarkResult run_benchmark(const Index& index,
                                            const Dataset& ds,
                                            std::size_t k) {
    double recall_sum = 0.0;
    const std::size_t n = ds.num_query();

    Stopwatch sw;
    for (std::size_t q = 0; q < n; ++q) {
        const auto got = index.search(ds.query_vec(q), k);
        recall_sum += recall_at_k(got, ds.ground_truth[q]);
    }
    const double seconds = sw.elapsed_seconds();

    BenchmarkResult r;
    r.num_query = n;
    r.recall = n ? recall_sum / static_cast<double>(n) : 0.0;
    r.qps = seconds > 0.0 ? static_cast<double>(n) / seconds : 0.0;
    r.mean_latency_ms = n ? (seconds / static_cast<double>(n)) * 1000.0 : 0.0;
    return r;
}

}  // namespace vdb
