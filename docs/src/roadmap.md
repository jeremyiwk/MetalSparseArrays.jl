# Roadmap

A coarse ordering of the work. Phases are sequenced so that each one supplies what
the next one needs; within a phase, features are planned and implemented one at a
time via the `plan-feature` workflow.

The target API is the union of what `SparseArrays` provides on the CPU and what
`CUDA.CUSPARSE` provides on NVIDIA hardware. Code written against either should run
unchanged on Metal.

Each phase lists its **deliverables** — the concrete types and functions it adds —
and a **phase gate**: the test criteria that define completion. The gates are a
developer contract: a phase is complete only when its gate passes in CI, a gate may
be strengthened but never weakened once written, and a feature that cannot meet its
gate is redesigned, not exempted. Throughout, `u` denotes the unit roundoff of the
element type and `c` a modest constant; "reference" means the `SparseArrays` result
for the same input on the CPU. Structural results (patterns, index arrays, `nnz`)
must match the reference exactly; floating point results to the stated bound.

## Cross-cutting requirements

These apply to every phase rather than being phases themselves:

- **Every device result is validated against the CPU reference** as part of the
  feature, never retroactively. Every gate below runs for every format it applies
  to, every supported element type, both index types (`Int32`, `Int64`), and with
  `Metal.allowscalar(false)` in force.
- **Identities are asserted over the corpus, not a draw.** Algebraic invariances
  (`(αA)x = α(Ax)`, `(A + B)x = Ax + Bx`, adjoint consistency
  `⟨Ax, y⟩ = ⟨x, Aᴴy⟩`, transpose round trips) run over all test seeds, densities,
  and the adversarial patterns of the phase — a single random pattern
  demonstrates nothing about contention or accumulation-order bugs.
- **Nonfinite values follow `SparseArrays`:** stored `NaN`/`Inf` are legal,
  participate in arithmetic, and propagate; no operation checks for them. Tests
  include patterns with nonfinite stored values and assert reference-matching
  propagation.
- **Benchmarks land with the feature**, keyed by representation, format, element
  type, and size.
- **Docs land with the feature.**
- **Regression corpus** (a convention, not part of the gates): every bug that
  escapes to a reported failure gets its minimal reproducer pinned as a
  permanent named test.
- **Shape and pattern edge cases are part of every gate:** the empty matrix, a
  matrix with `nnz = 0`, fully dense stored patterns, a single stored entry,
  empty rows and columns, and stored entries whose value is zero (structural
  nonzeros holding `0` are preserved, matching `SparseArrays`).

## Non-goals

Stated so they are decisions, not omissions: no direct sparse factorizations
(sparse LU/Cholesky/QR — Metal has no library support and the iterative path is
the point of this package), and no `Float64` on the device, which the hardware
does not provide. Both remain available by converting to the CPU.

## Phase 0 — Foundations *(complete)*

Package scaffolding, the test harness that validates every device result against the
`SparseArrays` result for the same input, and the conventions for tolerances and
reproducibility. Device test sets are skipped on machines without a GPU for local
convenience only: CI requires the device (`CI_EXPECT_DEVICE`), so nothing merges
unverified on Metal. Element types are discovered by probing the device.

## Phase 1 — Storage formats *(in progress: formats and interconversions shipped; dense conversions and BSR remain)*

**Deliverables**

- `AbstractMtlSparseMatrix{Tv, Ti} <: SparseArrays.AbstractSparseMatrix{Tv, Ti}`,
  and under it the device-resident formats, mirroring `CUSPARSE` naming:
  - `MtlSparseMatrixCSC{Tv, Ti}` — `colptr`, `rowval`, `nzval` as `MtlVector`s
    plus dimensions, field names matching `SparseMatrixCSC`;
  - `MtlSparseMatrixCSR{Tv, Ti}` — `rowptr`, `colval`, `nzval`;
  - `MtlSparseMatrixCOO{Tv, Ti}` — `rowval`, `colval`, `nzval`;
  - BSR later, once a consumer for it exists.
- Constructors from `SparseMatrixCSC` (converting indices to `Ti` with an
  overflow check against `typemax(Ti)`) and from raw index and value arrays with
  validation; `MtlSparseVector` deferred until a consumer needs it.
- `Adapt.adapt_storage` rules so `adapt(MtlArray, A)` and `adapt(Array, dA)`
  transfer in both directions; `SparseMatrixCSC(dA)` back-conversion.
- Conversions between the three formats on the device, and conversion to and
  from dense arrays on both host (`Array`) and device (`MtlArray`).

**Phase gate**

- Round trip host → device → host reproduces `colptr`, `rowval`, `nzval`, and the
  dimensions bit-for-bit, for every format, `Tv`, and `Ti`.
- Every format conversion preserves the `(i, j, v)` triple set exactly, verified
  against reference by converting back to `SparseMatrixCSC`; CSC↔CSR round trips
  are exact; dense conversions agree with `Array(::SparseMatrixCSC)` and
  `sparse(::Matrix)` including the treatment of explicit zeros (densifying keeps
  values; sparsifying a dense array drops numerical zeros, matching `sparse`).
- Invalid construction is rejected with `ArgumentError` naming the violated
  invariant: non-monotonic pointer array, pointer not starting at one or ending
  at `nnz + 1`, indices out of range, unsorted or duplicate row indices within a
  column (policy matching `SparseMatrixCSC`), and index values exceeding
  `typemax(Int32)` on conversion.
- `Base.show` displays every format legibly at the REPL without scalar indexing
  (summary line plus a bounded number of entries fetched in one transfer).

## Phase 2 — Array interface

**Deliverables**

- `size`, `axes`, `eltype`, `similar` (preserving and changing `Tv`/`Ti`),
  `copy`, `collect`, `Array`.
- `nnz`, `nonzeros`, `rowvals` (CSC), `findnz` returning host-usable triples.
- The scalar `getindex`/`setindex!` policy: erroring under
  `Metal.allowscalar(false)`, following the `CUSPARSE` precedent; documented.
- Broadcasting over stored values for zero-preserving scalar functions
  (`A .* 2`, `abs.(A)`); a function with `f(0) ≠ 0` follows the `CUSPARSE`
  policy (error, do not densify silently).
- COO assembly: `sortperm` on `(j, i)`, duplicate accumulation with `+`,
  matching `sparse(I, J, V, m, n, +)`.
- Constructor parity with `SparseArrays`: the constructors `SparseArrays`
  accepts work with device storage — `sparse`/format constructors from
  structured matrices over device vectors (`Diagonal(::MtlVector)`,
  `Bidiagonal`, `Tridiagonal`, `SymTridiagonal`) and `spdiagm`-style
  construction. Also convenient for building test problems directly on device.

**Phase gate**

- Every listed method agrees with the reference exactly for structural output
  and to `c·u` for value output, across formats, `Tv`, and `Ti`.
- Structured-matrix constructors agree exactly (structure and values) with
  `sparse` applied to the same structured matrix over host vectors.
- `similar` variants produce the documented format, element type, and index
  type; `collect`/`Array` equal `Array(SparseMatrixCSC(dA))` elementwise.
- Broadcasting: zero-preserving functions preserve the pattern exactly (stored
  zeros included); non-zero-preserving functions raise the documented error; the
  whole broadcast suite runs under `allowscalar(false)`.
- COO assembly with duplicate, unsorted, and adversarially ordered input
  (reverse-sorted, all-duplicates collapsing to one entry, interleaved columns)
  matches `sparse(I, J, V, m, n, +)` exactly in structure and to `c·k·u` in
  values, `k` the largest duplicate multiplicity.

## Phase 3 — Sparse BLAS

**Deliverables**

- `mul!` in three- and five-argument forms: sparse × dense vector and sparse ×
  dense matrix, for each storage format; `Transpose`, `Adjoint`, `Symmetric`,
  and `Hermitian` wrappers.
- Sparse triangular solve (`ldiv!` on `UpperTriangular`/`LowerTriangular`-wrapped
  device sparse matrices, unit and non-unit diagonal) — placed here for
  completeness of the solver interface and needed by Phase 6's incomplete
  factorizations.
- Dispatch to Metal Performance Shaders where a matching primitive exists, hand
  written kernels otherwise; a stated, tested policy on reproducibility of
  atomic accumulation (deterministic ordering preferred; where atomics are used
  the docstring says so).

**Phase gate**

- SpMV/SpMM: `|y_device − y_ref| ≤ c·r·u·(|A|·|x|)` componentwise, `r` the
  maximum stored entries per row — the bound that stays honest for long
  accumulations; in `Float16`/`BFloat16` a widened accumulator meets the same
  bound where the docstring promises it.
- Five-argument semantics exact to convention: `β = 0` overwrites `C` even when
  `C` contains `NaN`; `α = 0` skips the product; both tested explicitly.
- Wrapper variants agree with materialized reference computations
  (`transpose(A) * x` against reference on the same triples).
- Adversarial patterns: empty rows and columns, one fully dense row (longest
  accumulation), one fully dense column (maximal write contention for CSR/COO
  atomics), diagonal, permutation matrices, `laplacian_2d`, and patterns sized
  to straddle kernel launch boundaries.
- Reproducibility: where determinism is promised, bit-identical results over
  repeated runs on the same device; where it is not, the docstring says so and
  the test asserts the accuracy bound instead.
- Triangular solve: componentwise backward error `≤ c·n·u` on well conditioned
  triangles; graded triangles (diagonal spanning `u … 1/u`) meet the normwise
  bound or fail loudly, never silently.

## Phase 4 — Iterative solver compatibility

This comes directly after sparse BLAS — not after the structural operations —
because it needs almost nothing beyond five-argument `mul!` and is the package's
headline promise.

**Deliverables**

- Conformance to the operator interface `Krylov.jl` requires (`mul!`, `eltype`,
  `size`, adjoint application) for every storage format.
- An end to end sweep: `cg` and `minres` on `laplacian_2d`; `gmres` and
  `bicgstab` on a nonsymmetric convection–diffusion operator; `lsmr` on a tall
  least squares problem — each with a known solution.

**Phase gate**

- Each solver reaches its residual tolerance on the device with an iteration
  count within a factor of two of the same solver on the CPU reference matrix
  (identical mathematics, different rounding — iteration counts may differ
  slightly, not qualitatively).
- Recovered solutions match the constructed true solutions to the tolerance
  implied by the solver's stopping criterion and `κ` of the problem.
- No scalar indexing during any solve (`allowscalar(false)`); device allocations
  per iteration are bounded and asserted, so a solver loop does not silently
  allocate per step.
- Exit criterion: an unmodified `Krylov.jl` example runs on a device matrix by
  changing only the array types. This example is a doctest in the documentation.

## Phase 5 — Structural and arithmetic operations

**Deliverables**

- Transpose and adjoint materialization; row and column permutation.
- `+`, `-`, scalar multiplication and division; sparse–sparse multiplication.
- `kron`, `hcat`/`vcat`/`hvcat`, `dropzeros!`, `droptol!`.
- Norms (`norm`, `opnorm(A, 1)`, `opnorm(A, Inf)`) and reductions (`sum`,
  `maximum`/`minimum` over stored entries with the documented empty-row policy).

**Phase gate**

- Structural operations (transpose, permutation, `kron`, concatenation,
  `dropzeros!`) match the reference pattern exactly.
- Arithmetic: values to `c·r·u` bounds as in Phase 3; pattern-union cases where
  addition cancels to numerical zero keep the stored entry, exactly as
  `SparseArrays` does.
- `droptol!` agrees with reference on entries straddling the threshold,
  including ties.
- Sparse–sparse products validated on patterns that maximize fill (outer-product
  shaped factors) and on patterns with no overlap (empty result).

## Phase 6 — Preconditioners

**Deliverables**

- Jacobi and block Jacobi preconditioners; `ic0` and `ilu0` incomplete
  factorizations, built on the Phase 3 sparse triangular solve.
- The application interface `KrylovPreconditioners.jl` expects (`ldiv!`-style
  two- and three-argument application).

**Phase gate**

- `ic0`/`ilu0` factors match a straightforward CPU reference implementation of
  the same zero-fill algorithm: identical pattern, values to `c·n·u` — validated
  entry by entry, not just through the solve.
- Breakdown (zero or negative pivot in `ic0`, zero pivot in `ilu0`) raises a
  precise error naming the pivot, matching the reference implementation's
  behavior on the same input.
- Effectiveness, not just correctness: preconditioned `cg` on `laplacian_2d`
  reduces iteration count versus unpreconditioned by the factor the literature
  leads one to expect for `ic0` at that mesh size, asserted with a documented
  margin — a preconditioner that "works" but does not accelerate fails its gate.
- A `KrylovPreconditioners.jl` example runs unmodified with these types.

## Phase 7 — Performance and release

**Deliverables**

- Kernel optimization guided by the measured crossovers; format-selection
  guidance in the documentation backed by the benchmark table.
- REPL quality-of-life display matching `SparseArrays`: the pattern/entry
  rendering `SparseMatrixCSC` gives at the REPL (braille pattern preview and
  the truncated entry listing), produced from a bounded number of entries
  fetched in one host transfer — never by scalar indexing.
- Comparison against the documented behavior of `CUDA.CUSPARSE` for API parity.
- Complete documentation with doctested examples; registration in General.

**Phase gate**

- The performance goals below hold for every shipped operation, evidenced by
  the benchmark crossover table checked into the docs.
- An API parity checklist against `SparseArrays` and `CUSPARSE` for every
  shipped feature, with every deliberate deviation documented in the docstring.
- Aqua piracy check clean; every public name docstringed and doctested; the
  registration PR passes General's automerge checks.

## Phase 8 — Autodiff compatibility

Differentiation support, added last because the rules are written against
operations that must first be stable.

**Deliverables**

- `ChainRulesCore` rules for the linear operations — SpMV, SpMM, and the solves
  through preconditioners — where the subtlety is not the derivative (the maps
  are linear) but the *projection*: gradients with respect to a sparse argument
  are projected onto its sparsity pattern, mirroring `ChainRules`'
  `ProjectTo(::SparseMatrixCSC)` semantics, and computed without scalar indexing
  so the same rule serves the device.
- Rules for structure-preserving operations (scaling, `+`, transpose) and a
  documented policy for operations whose derivative would densify.

**Phase gate**

- `ChainRulesTestUtils.test_rrule`/`test_frule` pass for every rule with the
  primal on the CPU mirror, finite differences to the element type's tolerance.
- The same rules execute on the device with gradients matching the CPU-path
  gradients to `c·r·u`, pattern preserved exactly.
- An end to end gradient through a `Krylov.jl` solve (differentiating `x(b)` for
  fixed `A`, the adjoint-solve identity) matches the CPU result — the
  integration test that the pieces compose.

## Performance goals

Benchmarks live in `benchmarks/` and are direct measurements of performance, not
proxies. Two standing targets:

1. Device sparse operations run faster than the same operations on `SparseArrays`
   on the CPU once the problem size is large enough, with the crossover size
   measured and reported per operation, format, and element type.
2. Device sparse operations beat dense `MtlArray` operations on the same problem
   starting at a fairly modest size — sparsity must pay for itself well before
   problems become huge, or the format or kernel needs rework.
