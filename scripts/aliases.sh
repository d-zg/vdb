# vectorDB dev-loop aliases. All assume cwd = repo root.
# Use from your shell rc:  source <repo>/scripts/aliases.sh

# build
alias cbpd='cmake --build --preset dev'
alias cbpr='cmake --build --preset release'

# benchmark: always rebuilds release first so you can't measure stale code
alias vbench='cmake --build --preset release && ./build/release/vdb_bench'

# optimizer telemetry: vectorization/unroll remarks for our headers only
# (compiles scratch/check.cpp, which explicitly instantiates the templates)
alias vopt="clang++ -std=c++23 -O3 -Iinclude -Rpass='loop-unroll|loop-vectorize|slp-vectorize' -Rpass-missed='loop-unroll|loop-vectorize' -Rpass-analysis='loop-vectorize' -c scratch/check.cpp -o /dev/null 2>&1 | grep -E 'include/vdb/.*remark' | sort -u | sort -t: -k2 -n"

# flame graph: profile the release bench with samply (brew install samply)
alias vprof='samply record ./build/release/vdb_bench'
