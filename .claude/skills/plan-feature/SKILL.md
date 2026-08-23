---
name: plan-feature
description: Plan, specify, implement, and integrate one MetalSparseArrays feature (a storage format, an array interface method, a sparse BLAS kernel, a preconditioner, or solver compatibility). Use when starting new sparse array work, when the user asks to add or plan an operation or format, or when picking up the next item from docs/roadmap.md.
---

# Planning a feature

One feature at a time. A feature is one operation or format plus the interface that
exposes it: for example the CSR type and its constructors, sparse matrix-vector
multiplication, `dropzeros!`, or the `ilu0` preconditioner. Do not start a second
feature until the current one meets its completion criteria.

The output of the planning stage is `.plan/current.md` (gitignored working memory).
The output of the implementation stage is source, tests, docstrings, and an updated
roadmap.

## 1. Scope the feature

Read `.plan/current.md` and `docs/roadmap.md`. Confirm which feature is next and
what it depends on. If a dependency is missing, that dependency is the feature.

State in one sentence what the feature computes or stores, and name the
corresponding `SparseArrays` and `CUDA.CUSPARSE` entry points.

## 2. Survey reference implementations

This is not optional and it comes before any design decision. This package is a
reimplementation of interfaces that already exist and are already well tested; the
work is to reuse those interfaces and their validation, not to invent either.

Consult, in roughly this order:

- `SparseArrays` (stdlib) — the API to replicate: function names, argument order,
  keyword arguments, the storage fields (`colptr`, `rowval`, `nzval`), the semantics
  of structural versus numerical zeros, and the behavior on empty, duplicate, and
  unsorted input. Read its tests; they define what correct means here.
- `CUDA.CUSPARSE` — the model for the device side: which formats exist, how they are
  named and constructed, index type, which operations are provided for which format,
  how transpose and adjoint are handled, and what it does about scalar indexing.
  Where `SparseArrays` and `CUSPARSE` disagree, follow `CUSPARSE` for device types
  and `SparseArrays` for anything the user sees.
- `Metal.jl` — what the device can actually do: supported element types, kernel
  launch, atomics, synchronization, unified memory, and which Metal Performance
  Shaders routines apply. Also read how `Metal.jl` itself structures array types and
  `Adapt` rules.
- `Krylov.jl` and `KrylovPreconditioners.jl` — the operator and preconditioner
  interfaces that must be satisfied for a feature that touches solving.
- The literature — Saad, *Iterative Methods for Sparse Linear Systems*; Davis,
  *Direct Methods for Sparse Linear Systems*; and the published treatments of
  parallel SpMV and segmented reduction — for the algorithm and its cost.

Record in `.plan/current.md` what was consulted and which specific conventions,
guarantees, and test measures are being adopted from each.

## 3. Define the interface

Write the exact signatures before implementing:

- The exported function and its in-place variant, matching `SparseArrays` naming and
  argument order, and the `CUSPARSE` spelling where the operation is device-specific.
- Which storage formats the method accepts, and which combinations of
  `transpose`, `adjoint`, `Symmetric`, and `Hermitian` wrappers it must handle.
- The index types supported; `Int32` is the default.
- Nothing about the element type. Signatures are generic in `Tv<:Number`; a concrete
  floating point type must not appear in a method signature, and every constant and
  tolerance is derived from `Tv` through `realtype` and `unit_roundoff`. `Float64` is
  not available on Apple GPUs, but `Float16`, `BFloat16`, `Float32`, and the complex
  types built on them are, and all of them must work. If the operation genuinely
  cannot be written for some supported element type, that restriction is a documented
  design decision with a stated reason, not an untested assumption.
- Whether the operation is asynchronous, and where synchronization happens.

## 4. Define completion criteria

Criteria must be checkable. The primary criterion is always agreement with the CPU
reference. Typically:

- **Agreement with `SparseArrays`.** For the same input, the device result equals the
  `SparseArrays` result: exactly, for structural and integer-valued operations
  (sparsity pattern, index arrays, `nnz`, conversions, permutations); and to a stated
  bound for floating point operations, `norm(y_device - y_cpu) <= c * u * norm(A) *
  norm(x)` with `u` the unit roundoff of the element type and `c` modest.
- **Round trip.** Host to device to host reproduces the input exactly, including the
  sparsity pattern and the ordering of index arrays.
- **Format parity.** Every format implementing the operation produces the same
  result for the same mathematical input.
- **Element type coverage.** The test set passes for every type in `ELEMENT_TYPES`,
  real and complex, including `Float16` and `BFloat16`. Tolerances scale with
  `unit_roundoff(Tv)`, so the same assertion holds in every precision without a
  separate case, and low precision results are compared against a reference computed
  in `referencetype(Tv)`. A new element type must require no change to `src`.
- **Structural semantics.** Explicit stored zeros are preserved where
  `SparseArrays` preserves them and dropped where it drops them; duplicate
  coordinate entries accumulate; unsorted input is handled as `SparseArrays` handles
  it.
- **Edge cases.** Empty matrices, all-zero matrices, a single nonzero, rows or
  columns with no nonzeros, one-by-one, very tall and very wide shapes, and a
  nonzero count near the limits of the index type.
- **Determinism.** If the implementation accumulates with atomics, the
  reproducibility of the result is stated in the docstring and tested for; prefer a
  deterministic ordering unless the performance cost is measured and recorded.
- **No unintended host round trips.** Device code does not scalar index device
  arrays; check with `Metal.allowscalar(false)`.
- **Solver compatibility**, for anything reachable from a solve: the relevant
  `Krylov.jl` solvers converge on a test problem with a known solution, to the
  documented residual tolerance, without allocating per iteration.

## 5. Write the plan

Overwrite `.plan/current.md` with: the feature, its status, references consulted,
the interface, the completion criteria as a checklist, the implementation steps as a
checklist, and open questions. Keep it terse; it is a memory aid, not a document.

## 6. Implement

Follow the steps in the plan, checking them off as they land. One format or
operation group per file in `src/`, named after it. Write the fallback path that is
correct first, confirm it against `SparseArrays`, then optimize.

Observe the conventions in `CLAUDE.md`: names from `SparseArrays` and the sparse
literature, precise and concise docstrings, minimal comments.

## 7. Integrate

A feature is not done until all of the following hold:

- The file is `include`d in `src/MetalSparseArrays.jl` and its public names are
  exported.
- Every exported name has a docstring stating what is computed or stored, the
  conditions on the input, the properties of the output, and any deviation from the
  `SparseArrays` or `CUSPARSE` behavior it corresponds to.
- A matching `test/test_<feature>.jl` is included from `test/runtests.jl`, loops
  over `SPARSE_TYPES`, `ELEMENT_TYPES`, and `INDEX_TYPES`, and is guarded by
  `DEVICE_AVAILABLE` where it needs a GPU.
- `Pkg.test()` passes, on a machine with a device and on one without.
- `docs/roadmap.md` reflects the new state.
- `.plan/current.md` is updated: criteria checked off, and the next feature named.
