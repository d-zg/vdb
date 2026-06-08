#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vdb {

// A flat (brute-force) in-memory store of fixed-dimension float vectors.
//
// NOTE: this baseline implementation is *deliberately* unoptimized. It copies
// freely, stores vectors as a vector-of-vectors (one heap allocation each, no
// locality), and does redundant work in the hot path. That is on purpose: it
// gives the benchmark harness a slow starting point so every future
// optimization shows up as a visible jump in QPS. Do not "fix" it casually —
// improving it is the whole exercise.
//
// Header-only: every method is defined in-class, so it is implicitly inline
// and can live entirely in this header with no matching .cpp.
class VectorStore {
public:
    using Id = std::uint64_t;

    explicit VectorStore(std::size_t dim) : dim_(dim) {
        if (dim_ == 0) {
            throw std::invalid_argument("VectorStore dimension must be > 0");
        }
    }

    // Adds a vector and returns its assigned id. Throws if size != dim().
    Id add(std::span<const float> vec) {
        if (vec.size() != dim_) {
            throw std::invalid_argument("vector size does not match store dimension");
        }
        const Id id = next_id_++;
        ids_.push_back(id);
        // Copy the span into its own heap-allocated row. No reserve, no flat
        // buffer — every insert allocates.
        std::vector<float> row(vec.begin(), vec.end());
        vectors_.push_back(row);  // copy again into the container
        return id;
    }

    // Returns the ids of the k nearest neighbours to `query` by L2 distance,
    // closest first. k is clamped to the number of stored vectors.
    [[nodiscard]] std::vector<Id> search(std::span<const float> query,
                                         std::size_t k) const {
        if (query.size() != dim_) {
            throw std::invalid_argument("query size does not match store dimension");
        }

        // No reserve. Build (distance, id) pairs, copying the query for every
        // single comparison.
        std::vector<std::pair<double, Id>> scored;
        for (std::size_t i = 0; i < vectors_.size(); ++i) {
            std::vector<float> query_copy(query.begin(), query.end());
            const double d = l2_distance(query_copy, vectors_[i]);
            scored.push_back(std::make_pair(d, ids_[i]));
        }

        // Full sort of everything, even though we only need the top k.
        std::sort(scored.begin(), scored.end());

        k = std::min(k, scored.size());
        std::vector<Id> result;
        for (std::size_t i = 0; i < k; ++i) {
            result.push_back(scored[i].second);
        }
        return result;
    }

    [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
    [[nodiscard]] std::size_t size() const noexcept { return ids_.size(); }

private:
    // Takes both operands by value (full copies) and uses std::pow + std::sqrt
    // in the inner loop. About as slow as an L2 distance can reasonably get.
    static double l2_distance(std::vector<float> a, std::vector<float> b) {
        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            sum += std::pow(static_cast<double>(a[i]) - static_cast<double>(b[i]), 2.0);
        }
        return std::sqrt(sum);
    }

    std::size_t dim_;
    Id next_id_ = 0;
    std::vector<Id> ids_;
    std::vector<std::vector<float>> vectors_;  // one heap alloc per vector
};

}  // namespace vdb
