# Roadmap

A coarse ordering of the work. Phases are sequenced so that each one supplies what
the next one needs; within a phase, features are planned and implemented one at a
time via the `plan-feature` workflow.

The target API is the union of what `SparseArrays` provides on the CPU and what
`CUDA.CUSPARSE` provides on NVIDIA hardware. Code written against either should run
unchanged on Metal.

## Phase 0 — Foundations

Package scaffolding, the test harness that validates every device result against the
`SparseArrays` result for the same input, and the conventions for tolerances,
reproducibility, and skipping device test sets on machines without a GPU. Element
types are discovered by probing the device, so every operation is tested against
everything the device supports and a new element type costs no work.

## Phase 1 — Storage formats

The device-resident sparse types: CSC, CSR, and COO, with BSR later. Type hierarchy
under `AbstractSparseArray`, construction from `SparseMatrixCSC` and from raw index
and value arrays, `Adapt` rules for host and device transfer, and conversion between
formats.

## Phase 2 — Array interface

`size`, `similar`, `copy`, `collect`, `nnz`, `nonzeros`, `rowvals`, `findnz`, the
policy for scalar `getindex` on device arrays, broadcasting over stored values, and
assembly from coordinate form including index sorting and duplicate accumulation.

## Phase 3 — Sparse BLAS

Sparse matrix-vector and matrix-matrix products (`mul!`, three and five argument
forms), the transpose, adjoint, `Symmetric`, and `Hermitian` variants, dispatching
to Metal Performance Shaders where it applies and to hand written kernels otherwise.
Includes the stated policy on reproducibility of atomic accumulation.

## Phase 4 — Structural and arithmetic operations

Transpose and permutation, addition, subtraction, scaling, sparse-sparse products,
`kron`, concatenation, `dropzeros!`, `droptol!`, and norms and reductions.

## Phase 5 — Iterative solver compatibility

Conformance to the operator interface `Krylov.jl` requires, and an end to end sweep
of the solvers (`cg`, `minres`, `gmres`, `bicgstab`, `lsmr`) on test problems with
known solutions, checked against residual bounds and against allocation
requirements.

## Phase 6 — Preconditioners

Jacobi, block Jacobi, and the incomplete factorizations `ic0` and `ilu0`, together
with the sparse triangular solve they depend on, matching the interface of
`KrylovPreconditioners.jl`.

## Phase 7 — Performance and release

Benchmarks against `SparseArrays` on the CPU and against the documented behavior of
`CUDA.CUSPARSE`, guidance on format selection, documentation, and registration.

## Performance goals

Benchmarks live in `benchmarks/` and are direct measurements of performance, not
proxies. Two standing targets:

1. Device sparse operations run faster than the same operations on `SparseArrays`
   on the CPU once the problem size is large enough, with the crossover size
   measured and reported per operation, format, and element type.
2. Device sparse operations beat dense `MtlArray` operations on the same problem
   starting at a fairly modest size — sparsity must pay for itself well before
   problems become huge, or the format or kernel needs rework.
