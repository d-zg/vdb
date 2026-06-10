// Fast syntax-check driver for vector_store.hpp.
//
// `template class` forces FULL instantiation of VectorStore<128>, so this
// catches template-instantiation errors that clangd silently ignores —
// without compiling the whole project. Run with:
//
//   clang++ -std=c++23 -fsyntax-only -Iinclude scratch/check.cpp
//
// No output = no errors. Add more `template class` lines for other dims.
#include "vdb/vector_store.hpp"

template class vdb::VectorStore<128>;
