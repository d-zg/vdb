#include "vdb/benchmark.hpp"
#include "vdb/dataset_io.hpp"
#include "vdb/vector_store.hpp"

#include <print>
#include <string>

int main(int argc, char** argv) {
    // Usage: vdb_bench [dataset.hdf5] [max_queries]
    // Defaults to the fashion-mnist ann-benchmarks file. max_queries caps the
    // query sweep so runs stay quick while the index is still slow; raise it
    // (or pass 0 for all 10k) as the index gets faster.
    const std::string path =
        argc > 1 ? argv[1] : "data/fashion-mnist-784-euclidean.hdf5";
    const std::size_t max_queries =
        argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 100;
    constexpr std::size_t kK = 10;

    std::println("loading {} (max_queries={}) ...", path, max_queries);
    const auto ds = vdb::load_ann_benchmarks(path, kK, max_queries);
    std::println("loaded: {} base vectors, {} queries, dim={}",
                 ds.num_base(), ds.num_query(), ds.dim);

    vdb::VectorStore index(ds.dim);
    for (std::size_t i = 0; i < ds.num_base(); ++i) {
        index.add(ds.base_vec(i));
    }
    std::println("index built ({} vectors)", index.size());

    const auto r = vdb::run_benchmark(index, ds, kK);

    std::println("");
    std::println("=== scoreboard ===");
    std::println("queries        : {}", r.num_query);
    std::println("recall@{:<8}: {:.4f}", kK, r.recall);
    std::println("qps            : {:.1f}", r.qps);
    std::println("mean latency   : {:.3f} ms", r.mean_latency_ms);
    return 0;
}
