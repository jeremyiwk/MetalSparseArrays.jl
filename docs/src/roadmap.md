# Roadmap

A coarse ordering of the work. Phases are sequenced so that each one supplies what
the next one needs; within a phase, features are planned and implemented one at a
time via the `plan-feature` workflow.

The target API is the union of what `SparseArrays` provides on the CPU and what
`CUDA.CUSPARSE` provides on NVIDIA hardware. Code written against either should run
unchanged on Metal.

## Cross-cutting requirements

These apply to every phase rather than being phases themselves:

- **Every device result is validated against the CPU reference** — exactly for
  structural results, to a stated floating point bound otherwise — as part of the
  feature, never retroactively.
- **Benchmarks land with the feature.** Each operation adds entries to the
  `benchmarks/` suite keyed by representation, format, element type, and size, so
  both crossover measurements exist from day one.
- **Docs land with the feature.** Each format's and operation's docstring appears
  in the API reference when it merges.

## Non-goals

Stated so they are decisions, not omissions: no direct sparse factorizations
(sparse LU/Cholesky/QR — Metal has no library support and the iterative path is
the point of this package), and no `Float64` on the device, which the hardware
does not provide. Both remain available by converting to the CPU.

## Phase 0 — Foundations *(complete)*

Package scaffolding, the test harness that validates every device result against the
`SparseArrays` result for the same input, and the conventions for tolerances,
reproducibility, and skipping device test sets on machines without a GPU. Element
types are discovered by probing the device, so every operation is tested against
everything the device supports and a new element type costs no work.

## Phase 1 — Storage formats

The device-resident sparse types: CSC, CSR, and COO, with BSR later. Type hierarchy
under `AbstractSparseArray`, construction from `SparseMatrixCSC` and from raw index
and value arrays, `Adapt` rules for host and device transfer, conversion between
formats, and conversion to and from dense arrays on both host and device.

## Phase 2 — Array interface

`size`, `similar`, `copy`, `collect`, `nnz`, `nonzeros`, `rowvals`, `findnz`, the
policy for scalar `getindex` on device arrays, broadcasting over stored values, and
assembly from coordinate form including index sorting and duplicate accumulation.

## Phase 3 — Sparse BLAS

Sparse matrix-vector and matrix-matrix products (`mul!`, three and five argument
forms), the transpose, adjoint, `Symmetric`, and `Hermitian` variants, and the
sparse triangular solve (`ldiv!` on triangular sparse matrices — needed here for
completeness of the solver interface and later by the incomplete-factorization
preconditioners). Dispatch to Metal Performance Shaders where it applies and to
hand written kernels otherwise. Includes the stated policy on reproducibility of
atomic accumulation.

## Phase 4 — Iterative solver compatibility

This comes directly after sparse BLAS — not after the structural operations —
because it needs almost nothing beyond five-argument `mul!` and is the package's
headline promise. Conformance to the operator interface `Krylov.jl` requires, and
an end to end sweep of the solvers (`cg`, `minres`, `gmres`, `bicgstab`, `lsmr`)
on test problems with known solutions, checked against residual bounds and against
allocation requirements. Exit criterion: an unmodified `Krylov.jl` example runs on
a device matrix by changing only the array types.

## Phase 5 — Structural and arithmetic operations

Transpose and permutation, addition, subtraction, scaling, sparse-sparse products,
`kron`, concatenation, `dropzeros!`, `droptol!`, and norms and reductions.

## Phase 6 — Preconditioners

Jacobi, block Jacobi, and the incomplete factorizations `ic0` and `ilu0`, built on
the sparse triangular solve from Phase 3, matching the interface of
`KrylovPreconditioners.jl`.

## Phase 7 — Performance and release

With benchmarks accumulated feature by feature, this phase is tuning and
publishing: kernel optimization guided by the measured crossovers, guidance on
format selection, comparison against the documented behavior of `CUDA.CUSPARSE`,
documentation, and registration in General. Exit criterion: the performance goals
below hold for every shipped operation.

## Performance goals

Benchmarks live in `benchmarks/` and are direct measurements of performance, not
proxies. Two standing targets:

1. Device sparse operations run faster than the same operations on `SparseArrays`
   on the CPU once the problem size is large enough, with the crossover size
   measured and reported per operation, format, and element type.
2. Device sparse operations beat dense `MtlArray` operations on the same problem
   starting at a fairly modest size — sparsity must pay for itself well before
   problems become huge, or the format or kernel needs rework.
