# Benchmarks

Direct measurements of performance for the formats and operations in `src/`.
See [the M5 measurements](RESULTS.md) for current CPU comparisons and remaining costs.

- `benchmarks.jl` defines `SUITE`, a vector of benchmark cases grouped per
  operation and keyed by representation, format, element type, and problem
  size.
- `runbenchmarks.jl` is the entry point: `julia benchmarks/runbenchmarks.jl`
  activates this environment, develops the package, and times every case.

Timing uses the standard-library macros, not BenchmarkTools: `Metal.@timed`
(the macro behind `Metal.@time`, which synchronizes the GPU before the
expression and wraps it in `Metal.@sync`) for device cases and `Base.@elapsed`
(the timing core of `Base.@time`) for host cases. Each case is compiled
untimed first, then minimum, median and p95 over 100 repetitions are reported.
Device times include host dispatch, allocation and device completion.

Pass operation group names to select a subset, for example
`julia benchmarks/runbenchmarks.jl copy unary_scale findnz`. Results include
hardware/package versions and tab-separated minimum, median and p95 seconds,
followed by the operation and case key. Array-interface cases compare the same
index width on CPU and device and include the stored entry count.
The last column is operations per sample; times are normalized per operation.
`zero_fill_batch` executes 100 operations per synchronization to measure batching,
while the other groups report standalone latency.

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
