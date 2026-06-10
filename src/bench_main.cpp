#include "vdb/benchmark.hpp"
#include "vdb/dataset_io.hpp"
#include "vdb/vector_store.hpp"

#include <cstddef>
#include <print>
#include <stdexcept>
#include <string>

namespace {

// The set of vector dimensions we compile a VectorStore for. Because the
// dimension is now a compile-time template parameter, every dimension we want
// to benchmark must appear here so the compiler instantiates that variant.
// These cover the common ann-benchmarks datasets (e.g. glove-100=100,
// sift-128=128, gist-960=960, fashion-mnist-784=784).
#define VDB_SUPPORTED_DIMS(X) \
    X(25) X(50) X(96) X(100) X(128) X(200) X(256) X(300) X(784) X(960)

// Builds a VectorStore<Dim> from the dataset and benchmarks it. The dataset's
// runtime dim is asserted to match the compile-time Dim the caller dispatched
// to (they always will, given the switch below).
template <std::size_t Dim>
vdb::BenchmarkResult run_for_dim(const vdb::Dataset& ds, std::size_t k) {
    if (ds.dim != Dim) {
        throw std::logic_error("dispatched to wrong Dim instantiation");
    }
    vdb::VectorStore<Dim> index;
    for (std::size_t i = 0; i < ds.num_base(); ++i) {
        index.add(ds.base_vec(i));
    }
    std::println("index built ({} vectors, dim={})", index.size(), Dim);
    return vdb::run_benchmark(index, ds, k);
}

// Runtime dim -> compile-time Dim dispatch. This is where the runtime dataset
// dimension gets mapped onto one of the compiled VectorStore<Dim> variants.
vdb::BenchmarkResult run_benchmark_for(const vdb::Dataset& ds, std::size_t k) {
    switch (ds.dim) {
#define VDB_DISPATCH_CASE(D) \
    case D:                  \
        return run_for_dim<D>(ds, k);
        VDB_SUPPORTED_DIMS(VDB_DISPATCH_CASE)
#undef VDB_DISPATCH_CASE
        default:
            throw std::runtime_error(
                "dimension " + std::to_string(ds.dim) +
                " is not compiled in; add it to VDB_SUPPORTED_DIMS in "
                "bench_main.cpp");
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Usage: vdb_bench [dataset.hdf5] [max_queries]
    const std::string path =
        argc > 1 ? argv[1] : "data/fashion-mnist-784-euclidean.hdf5";
    const std::size_t max_queries =
        argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 100;
    constexpr std::size_t kK = 10;

    std::println("loading {} (max_queries={}) ...", path, max_queries);
    const auto ds = vdb::load_ann_benchmarks(path, kK, max_queries);
    std::println("loaded: {} base vectors, {} queries, dim={}",
                 ds.num_base(), ds.num_query(), ds.dim);

    const auto r = run_benchmark_for(ds, kK);

    std::println("");
    std::println("=== scoreboard ===");
    std::println("queries        : {}", r.num_query);
    std::println("recall@{:<8}: {:.4f}", kK, r.recall);
    std::println("qps            : {:.1f}", r.qps);
    std::println("mean latency   : {:.3f} ms", r.mean_latency_ms);
    return 0;
}
