#include "vdb/benchmark.hpp"
#include "vdb/vector_store.hpp"

#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("recall_at_k counts overlap regardless of order") {
    const std::vector<vdb::Id> truth{1, 2, 3, 4};

    SECTION("perfect recall") {
        const std::vector<vdb::Id> got{4, 3, 2, 1};
        REQUIRE_THAT(vdb::recall_at_k(got, truth), WithinAbs(1.0, 1e-9));
    }
    SECTION("half recall") {
        const std::vector<vdb::Id> got{1, 2, 99, 98};
        REQUIRE_THAT(vdb::recall_at_k(got, truth), WithinAbs(0.5, 1e-9));
    }
    SECTION("zero recall") {
        const std::vector<vdb::Id> got{7, 8, 9};
        REQUIRE_THAT(vdb::recall_at_k(got, truth), WithinAbs(0.0, 1e-9));
    }
}

TEST_CASE("Stopwatch measures a non-negative, monotonic duration") {
    vdb::Stopwatch sw;
    volatile double sink = 0.0;
    for (int i = 0; i < 1000; ++i) sink += static_cast<double>(i);
    REQUIRE(sw.elapsed_seconds() >= 0.0);
}
