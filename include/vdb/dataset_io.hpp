#pragma once

// Loader for ann-benchmarks HDF5 datasets (https://ann-benchmarks.com).
//
// Their files contain four datasets:
//   train      — float32 [n_train x dim]   vectors to index
//   test       — float32 [n_test x dim]    query vectors
//   neighbors  — int32   [n_test x 100]    true nearest-neighbour *row indices*
//                                          into train, closest first
//   distances  — float32 [n_test x 100]    matching distances (unused here)
//
// The row indices line up with our VectorStore ids as long as vectors are
// added in file order — which is exactly what the benchmark runner does.
//
// This header is the only place vdb touches HDF5, so only targets that load
// datasets (vdb_bench) need to link HighFive.

#include "vdb/benchmark.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

namespace vdb {

// Loads an ann-benchmarks file, keeping ground truth for the first `k`
// neighbours and at most `max_queries` queries (0 = all). Capping queries
// keeps iteration fast while the index is still slow.
[[nodiscard]] inline Dataset load_ann_benchmarks(const std::string& path,
                                                 std::size_t k,
                                                 std::size_t max_queries = 0) {
    const HighFive::File file(path, HighFive::File::ReadOnly);

    const auto train = file.getDataSet("train")
                           .read<std::vector<std::vector<float>>>();
    auto test = file.getDataSet("test")
                    .read<std::vector<std::vector<float>>>();
    auto neighbors = file.getDataSet("neighbors")
                         .read<std::vector<std::vector<std::int32_t>>>();

    if (train.empty() || test.empty()) {
        throw std::runtime_error("dataset is empty: " + path);
    }
    if (k > neighbors.front().size()) {
        throw std::runtime_error("k exceeds ground-truth depth in " + path);
    }

    if (max_queries != 0 && max_queries < test.size()) {
        test.resize(max_queries);
        neighbors.resize(max_queries);
    }

    Dataset ds;
    ds.dim = train.front().size();
    ds.k = k;

    ds.base.reserve(train.size() * ds.dim);
    for (const auto& row : train) {
        ds.base.insert(ds.base.end(), row.begin(), row.end());
    }
    ds.query.reserve(test.size() * ds.dim);
    for (const auto& row : test) {
        ds.query.insert(ds.query.end(), row.begin(), row.end());
    }

    ds.ground_truth.reserve(neighbors.size());
    for (const auto& row : neighbors) {
        std::vector<VectorStore::Id> truth;
        truth.reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            truth.push_back(static_cast<VectorStore::Id>(row[i]));
        }
        ds.ground_truth.push_back(std::move(truth));
    }
    return ds;
}

}  // namespace vdb
