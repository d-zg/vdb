#include "vdb/vector_store.hpp"

#include <array>
#include <print>

int main() {
    vdb::VectorStore store(3);

    store.add(std::array{1.0f, 0.0f, 0.0f});
    store.add(std::array{0.0f, 1.0f, 0.0f});
    store.add(std::array{0.9f, 0.1f, 0.0f});

    const auto query = std::array{1.0f, 0.0f, 0.0f};
    const auto results = store.search(query, 2);

    std::println("stored {} vectors of dim {}", store.size(), store.dim());
    std::print("nearest ids:");
    for (const auto id : results) {
        std::print(" {}", id);
    }
    std::println("");
    return 0;
}
