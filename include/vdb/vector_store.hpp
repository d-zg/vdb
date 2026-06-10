#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <array>

namespace vdb {

template <std::size_t N>
class ColumnVector {
public:
    template <typename It>
    ColumnVector(It first, It last) {
        std::copy(first, last, data_.begin()); 
    }

    float operator[](size_t idx) const {
        return data_[idx];
    }
private:
    std::array<float, N> data_; 
};

template<size_t dim> 
float l2_distance(const ColumnVector<dim>& a, const ColumnVector<dim>& b) {
    const size_t kLanes = 8;
    std::array<float, kLanes> s{};
    for (std::size_t i = 0; i < dim; i += kLanes) {
        for (std::size_t j = 0; j < kLanes; ++j) {
            s[j] += std::pow((a[i+j]) - (b[i+j]), 2.0f);
        }
    }
    return std::reduce(s.begin(), s.end(), 0.0f);
}

using Id = std::uint64_t;
// A flat (brute-force) in-memory store of fixed-dimension float vectors.
//
// Header-only: every method is defined in-class, so it is implicitly inline
// and can live entirely in this header with no matching .cpp.
template <std::size_t dim> 
class VectorStore {
public:

    // Adds a vector and returns its assigned id. Throws if size != dim().
    Id add(std::span<const float> vec) {
        if (vec.size() != dim) {
            throw std::invalid_argument("vector size does not match store dimension");
        }
        const Id id = next_id_++;
        ids_.push_back(id);
        ColumnVector<dim> row(vec.begin(), vec.end());
        vectors_.push_back(row);
        return id;
    }

    // Returns the ids of the k nearest neighbours to `query` by L2 distance,
    // closest first. k is clamped to the number of stored vectors.
    [[nodiscard]] std::vector<Id> search(std::span<const float> query,
                                         std::size_t k) const {
        if (query.size() != dim) {
            throw std::invalid_argument("query size does not match store dimension");
        }

        auto queryVector = ColumnVector<dim>(query.begin(), query.end()); 
        std::vector<std::pair<float, Id>> scored;
        scored.reserve(size());
        for (std::size_t i = 0; i < vectors_.size(); ++i) {
            auto d = l2_distance(queryVector, vectors_[i]);
            scored.push_back(std::make_pair(d, ids_[i]));
        }

        k = std::min(k, scored.size());
        std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k), scored.end());
        std::vector<Id> result; 
        result.reserve(k);
        for (size_t i = 0; i < k; ++i) {
            result.push_back(scored[i].second);
        };
        return result;
    }

    [[nodiscard]] std::size_t size() const noexcept { return ids_.size(); }

private:
    Id next_id_ = 0;
    std::vector<Id> ids_;
    std::vector<ColumnVector<dim>> vectors_;

};

// Defined only by .clangd, never by real builds. Template bodies above are
// only type-checked when instantiated; this gives the IDE an instantiation
// to chew on so errors get squiggles at their real locations.
#ifdef VDB_IDE_CHECKS
template class VectorStore<128>;
#endif

}  // namespace vdb
