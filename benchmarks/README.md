# Benchmarks

Direct measurements of performance for the formats and operations in `src/`.

- `benchmarks.jl` defines `SUITE`, a vector of benchmark cases grouped per
  operation and keyed by representation, format, element type, and problem
  size.
- `runbenchmarks.jl` is the entry point: `julia benchmarks/runbenchmarks.jl`
  activates this environment, develops the package, and times every case.

Timing uses the standard-library macros, not BenchmarkTools: `Metal.@timed`
(the macro behind `Metal.@time`, which synchronizes the GPU before the
expression and wraps it in `Metal.@sync`) for device cases and `Base.@elapsed`
(the timing core of `Base.@time`) for host cases. Each case is compiled
untimed first, then the minimum over a fixed number of repetitions is
reported.

CI runs the suite informationally on pull requests and on a weekly schedule
(`.github/workflows/Benchmarks.yml`); timing results never block a merge, but
regressions are reviewed at PR time and the roadmap performance goals are
enforced before tagging a release.

The goals these benchmarks exist to verify:

1. Sparse operations on the device must run faster than the same operations on
   `SparseArrays` on the CPU once the problem size is large enough, with the
   crossover size measured and reported.
2. Sparse operations on the device must beat dense `MtlArray` operations on the
   same problem starting at a fairly modest size — sparsity must pay for itself
   well before problems become huge.
