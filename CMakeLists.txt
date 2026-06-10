cmake_minimum_required(VERSION 3.24)
project(vdb LANGUAGES CXX)

# --- Global standards -------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for clangd / editor tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to a sane build type if none was given.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# --- Warnings ---------------------------------------------------------------
add_library(vdb_warnings INTERFACE)
target_compile_options(vdb_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)

# --- The library (header-only) ----------------------------------------------
# INTERFACE = no compiled sources; consumers just get the include path and any
# usage requirements. All of vdb's code lives in headers under include/vdb/.
add_library(vdb INTERFACE)
target_include_directories(vdb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- HDF5 / HighFive (for loading ann-benchmarks datasets) -------------------
# HighFive is a header-only C++ wrapper over the system HDF5 C library.
include(FetchContent)
find_package(HDF5 REQUIRED COMPONENTS C)
# HighFive v2 declares an old cmake_minimum_required that CMake 4 rejects;
# this tells CMake to treat it as 3.5-compatible instead of erroring.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  HighFive
  GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
  GIT_TAG        v2.10.1
)
FetchContent_MakeAvailable(HighFive)

# --- The benchmark harness --------------------------------------------------
add_executable(vdb_bench src/bench_main.cpp)
target_link_libraries(vdb_bench PRIVATE vdb vdb_warnings HighFive)

# --- Scratch syntax-check (dev only) ----------------------------------------
# A throwaway TU that explicitly instantiates VectorStore so clangd reports
# template-instantiation errors live in the editor. EXCLUDE_FROM_ALL so it
# never builds as part of the project; it only needs to appear in
# compile_commands.json so the language server knows its flags.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scratch/check.cpp)
  add_library(vdb_scratch_check OBJECT EXCLUDE_FROM_ALL scratch/check.cpp)
  target_link_libraries(vdb_scratch_check PRIVATE vdb)
endif()

# --- Tests ------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
