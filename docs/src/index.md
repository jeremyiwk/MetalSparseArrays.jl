# MetalSparseArrays.jl

Sparse arrays for the Apple Silicon GPU backend. Code written against
`SparseArrays` on the CPU, or against `CUDA.CUSPARSE` on NVIDIA hardware, runs
unchanged on Metal — same storage formats, same function names, same semantics —
including the operator interface that iterative solvers such as `Krylov.jl`
require.

The target API is the union of two existing, well tested APIs. Almost nothing
here is a new design; the work is faithful reimplementation on a different
device, and reuse of the validation those libraries already provide.

## Getting started

The package is registered nowhere yet; install it by developing the repository:

```julia
using Pkg
Pkg.develop(url = "https://github.com/jeremyiwk/MetalSparseArrays.jl")
using MetalSparseArrays
```

An Apple Silicon GPU is required for device operations. The test suite runs
anywhere for local development — device test sets are skipped without a GPU —
but merges are gated on CI runs where a Metal device is required to be present
and exercised, so no functionality lands unverified on device.

## Device constraints

These are properties of the hardware and shape every design decision:

- **No double precision.** Apple GPUs support `Float16`, `BFloat16`, `Float32`,
  and the complex types built on each — but not `Float64`, which appears only as
  the element type of CPU references. Supported element types are probed at run
  time.
- **32-bit indices.** The default index type is `Int32`, matching `CUSPARSE`;
  conversions from host matrices are checked against `typemax(Int32)`.
- **No scalar indexing** of device arrays in library code.
- **Asynchrony.** Kernel launches return before the work completes; results are
  synchronized before host reads.

Correctness is defined by the CPU reference: every device operation is validated
against the `SparseArrays` result for the same input — exactly for structural
results, and to a stated floating point bound otherwise.

See the [Roadmap](@ref) for the phase ordering of the work and the
[API reference](@ref) for what is implemented today.
