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

These measurements precede device reordering and coordinate assembly. Their
follow-up results appear below. Small isolated calls remain limited by
submission/completion overhead; faster kernels alone cannot close that gap.

## Device ordering and assembly follow-up

Same hardware, package versions, and synchronized minimum-of-100 methodology.
Reproduce with:

```sh
julia benchmarks/runbenchmarks.jl findnz format_reorder mixed_sparse_add coo_assembly
```

All device format conversions now preserve canonical order without copying
coordinates or values through the host. Stable numeric keys include an input
rank, so correctness does not depend on backend tie ordering. Reorders involving
CSC require Metal.jl 1.10 and support at most `typemax(Int32)` entries; Int64
coordinates use multiple sorting passes when needed to avoid key overflow.

For Float32/Int32 tridiagonal inputs, current `findnz` minima are:

| Stored entries | CPU CSC, us | Device CSR, us | Device CSC, us | Device COO, us |
|---:|---:|---:|---:|---:|
| 12,286 | 6.5 | 283 | 148 | 285 |
| 786,430 | 434 | 2,294 | 397 | 2,243 |

At 786k entries, CSR/COO improve from 4,298/7,236 us to 2,294/2,243 us,
approximately 1.9/3.2 times faster. The global sort still costs more than CPU
reordering. CSC-to-CSR conversion takes 2,385 us versus 1,156 us for the CPU
transpose-storage equivalent. Mixed CSR/CSC addition takes 2,916 us versus
661 us with both inputs already CSR; resident conversion is still worth avoiding
inside repeated computations.

`sparse(I, J, V, m, n; fmt)` now assembles device coordinates directly into the
requested format. Validation and the stored-count readback are included below.
CPU output is CSC; input value and index widths match. Times are milliseconds.

| Input pattern | Input triples | CPU | Device CSC | Device CSR | Device COO |
|---|---:|---:|---:|---:|---:|
| Reverse-ordered bands | 786,432 | 3.364 | 3.587 | 3.564 | 3.466 |
| 16 inputs per coordinate | 262,144 | 0.593 | 1.714 | 1.724 | 1.669 |
| All inputs at one coordinate | 262,144 | 0.625 | 15.796 | 15.783 | 15.780 |

The largest reverse-band case is within about 7% of CPU. Duplicate-heavy cases
remain slower. High average duplicate multiplicity uses cooperative threadgroup
loads, but one thread folds each group in original input order, preserving
SparseArrays' rounding and custom-combiner semantics. This reduced the
single-coordinate Float32 case from roughly 38 ms to 16 ms. A SIMD-shuffle
variant was slower and is not included in the implementation.

An isolated long duplicate group among many short groups can still select the
serial path: dispatch currently uses average multiplicity. More adaptive group
scheduling and cheaper ordering remain performance work, alongside structured
device constructors and sparse matrix-vector application. CPU parity has not
been established across arbitrary sparsity distributions.
