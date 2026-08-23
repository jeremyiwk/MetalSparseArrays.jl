# The benchmark suite. `runbenchmarks.jl` activates the environment, includes
# this file, and runs `SUITE`.
#
# Convention: one BenchmarkGroup per operation, keyed by representation
# ("SparseArrays", "device sparse", "device dense"), format, element type, and
# problem size, so that both crossover sizes stated in docs/src/roadmap.md —
# device sparse over CPU sparse, and device sparse over device dense — can be
# read directly from the results. Device benchmarks synchronize before timing.

using BenchmarkTools
using MetalSparseArrays

const SUITE = BenchmarkGroup()

# Example shape for when the first kernel lands:
#
# SUITE["spmv"] = BenchmarkGroup()
# for n in (64, 256, 1024), Tv in (Float32,)
#     A = laplacian_2d(Tv, n)
#     x = ones(Tv, size(A, 2))
#     SUITE["spmv"]["SparseArrays", Tv, n] = @benchmarkable $A * $x
# end
