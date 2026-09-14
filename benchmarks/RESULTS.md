# Array-interface measurements — 2026-09-14

Apple M5 (10 GPU cores), macOS 26.6.2, Julia 1.12.6, Metal.jl 1.10.3,
GPUArrays 11.5.13. Inputs reside in their respective CPU/device memory before
timing. Times include allocations and host dispatch; each standalone device
sample waits for completion. Compilation is excluded. Reported minima are from
100 samples; scheduling variability is visible in the median and p95.

Reproduce the implemented paths with:

```sh
julia benchmarks/runbenchmarks.jl copy similar unary_scale inplace_scale findnz densify zero_fill_batch
```

The array-interface family is a square tridiagonal matrix, with diagonal 2 and
off-diagonals -1. CPU and device use matching value and index types. The following
subset uses Float32/Int32, CPU CSC and device CSR. Times are microseconds.

| Stored entries | Operation | CPU min | GPU min | GPU median | GPU p95 |
|---:|---|---:|---:|---:|---:|
| 12,286 | copy | 2.3 | 138 | 315 | 1,313 |
| 12,286 | scale | 11.3 | 134 | 295 | 520 |
| 12,286 | in-place sign change | 12.0 | 295 | 389 | 1,034 |
| 196,606 | copy | 37.2 | 189 | 308 | 1,564 |
| 196,606 | scale | 179 | 177 | 234 | 578 |
| 196,606 | in-place sign change | 181 | 390 | 496 | 865 |
| 786,430 | copy | 171 | 361 | 417 | 1,528 |
| 786,430 | scale | 777 | 370 | 448 | 717 |
| 786,430 | in-place sign change | 814 | 669 | 762 | 1,585 |

Scaling reaches approximate CPU parity around 200k entries in this family and
is 2.1 times faster at 786k entries. In-place sign change is 1.2 times faster at
786k entries. Copying remains slower. These crossovers do not generalize to
arbitrary sparsity distributions, types or machines.

The pre-change audit measured the same 786k-entry operations at 1,050 us for
copy, 1,068 us for scale and 8,299 us for in-place sign change. The corresponding
improvements are about 2.9, 2.9 and 12.4 times. These are separate warm runs on
the same hardware, not simultaneous paired measurements.

In-place assignment retains SparseArrays' numerical-zero compaction semantics.
It uses device count/scan/fill and reads back the resulting stored count. A
value-only kernel is not a valid general replacement, even though it measures
much faster. Zero scalar assignment does preserve the pattern and can be batched
without a count readback.

The same runner exposes remaining format and memory costs:

| Operation | Size / stored entries | CPU min, us | Device min, us |
|---|---|---:|---:|
| CSC findnz | 4,096 / 12,286 | 6.5 | 145 |
| CSC findnz | 262,144 / 786,430 | 435 | 423 |
| CSR findnz | 262,144 / 786,430 | 435 | 4,298 |
| COO findnz | 262,144 / 786,430 | 435 | 7,236 |
| CSR densification | 4,096 / 12,286 | 687 | 1,507 |
| CSC densification | 4,096 / 12,286 | 687 | 1,512 |
| CSR zero fill, 100 per sync | 4,096 / 12,286 | 0.29 | 3.57 |
| CSR zero fill, 100 per sync | 262,144 / 786,430 | 18.79 | 21.56 |

CSR/COO findnz still reorder through CPU CSC to provide column-major results.
Direct densification removes intermediate coordinates and extra scatter work,
but allocating and initializing a dense output remains substantial. Batched
zero-fill times are normalized per operation and must not be read as standalone
latency or as a prediction for a solver loop with dependencies.

The next performance work is device format reordering, deterministic assembly,
long-slice measurements, and allocation-free sparse matrix-vector application
measured in an actual iterative solve. Small isolated calls remain limited by
submission/completion overhead; faster kernels alone cannot close that gap.
