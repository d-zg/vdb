#include "vdb/vector_store.hpp"

#include <array>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("add assigns sequential ids and grows size") {
    vdb::VectorStore<2> store;
    REQUIRE(store.size() == 0);

    const auto a = store.add(std::array{1.0f, 2.0f});
    const auto b = store.add(std::array{3.0f, 4.0f});

    REQUIRE(a == 0);
    REQUIRE(b == 1);
    REQUIRE(store.size() == 2);
}

TEST_CASE("add rejects wrong-dimension input") {
    vdb::VectorStore<3> store;
    const auto bad = std::array{1.0f, 2.0f};
    REQUIRE_THROWS_AS(store.add(bad), std::invalid_argument);
}

TEST_CASE("search returns nearest neighbours closest-first") {
    vdb::VectorStore<2> store;
    const auto origin = store.add(std::array{0.0f, 0.0f});
    const auto near = store.add(std::array{0.1f, 0.0f});
    const auto far = store.add(std::array{5.0f, 5.0f});

    const auto results = store.search(std::array{0.0f, 0.0f}, 2);

    REQUIRE(results.size() == 2);
    REQUIRE(results[0] == origin);
    REQUIRE(results[1] == near);
    REQUIRE(results[1] != far);
}

TEST_CASE("search clamps k to the number of stored vectors") {
    vdb::VectorStore<1> store;
    store.add(std::array{1.0f});
    const auto results = store.search(std::array{0.0f}, 10);
    REQUIRE(results.size() == 1);
}
