# MetalSparseArrays.jl

## What this project is

Sparse arrays for the Apple Silicon GPU backend. The goal is that code written
against `SparseArrays` on the CPU, or against `CUDA.CUSPARSE` on NVIDIA hardware,
runs unchanged on Metal — same storage formats, same function names, same semantics
— including the operator interface that iterative solvers such as `Krylov.jl`
require.

The target API is therefore the union of two existing, well tested APIs. Almost
nothing here is a new design; the work is faithful reimplementation on a different
device, and reuse of the validation those libraries already provide.

Quality is the point. Prefer a small number of formats and operations that are
correct, validated against the CPU reference, and well documented over broad
coverage that is not.

## Layout

- `src/` — one storage format or operation group per file, included from
  `src/MetalSparseArrays.jl`. `src/common.jl` holds shared utilities
  (`DEFAULT_INDEX_TYPE`, `indextype`, `realtype`, `unit_roundoff`).
- `test/` — `runtests.jl` includes one `test_<feature>.jl` per feature.
  `test/testsuite.jl` defines `DEVICE_AVAILABLE`, `SPARSE_TYPES`, `INDEX_TYPES`, the
  element type probe `supported_element_types` and the `ELEMENT_TYPES` it produces,
  and the test problem generators `testsparse` and `laplacian_2d`.
  Device test sets are guarded by `DEVICE_AVAILABLE` so the suite still runs without
  a GPU.
- `docs/roadmap.md` — the coarse phase ordering of the work.
- `.plan/current.md` — gitignored working memory for the feature in progress.
- `.claude/skills/plan-feature/` — the workflow for adding a feature.

## Workflow

Adding or planning any format or operation goes through the `plan-feature` skill. It
defines the sequence: scope the feature, survey reference implementations, fix the
interface, define completion criteria, write `.plan/current.md`, implement,
integrate. Read `.plan/current.md` at the start of a session to recover context.

Run the tests with:

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

## CI and merge standards

CI (`.github/workflows/CI.yml`) runs on every push to `main` and every pull
request, and all three jobs must pass before merging into `main`:

1. **Tests** — `Pkg.test()` on the oldest supported Julia (see `[compat]`) and the
   latest stable release, on macOS runners (the package depends on `Metal.jl`).
   Device test sets run only where `DEVICE_AVAILABLE` finds a Metal device; the
   guarded CPU-side suite must pass regardless.
2. **Aqua QA** — `Aqua.test_all(MetalSparseArrays)`: method ambiguities, unbound
   type parameters, undefined exports, stale dependencies, missing compat bounds,
   type piracy. Reproduce locally with:

   ```
   julia -e 'using Pkg; Pkg.activate(temp = true); Pkg.develop(path = "."); Pkg.add("Aqua");
             using Aqua, MetalSparseArrays; Aqua.test_all(MetalSparseArrays)'
   ```

3. **Formatting** — all Julia source must be formatted with
   [Runic](https://github.com/fredrikekre/Runic.jl). Install it once into a shared
   environment (`julia -e 'using Pkg; Pkg.activate("runic", shared = true); Pkg.add("Runic")'`),
   then check with `julia --project=@runic -m Runic --check --diff .` and fix with
   `julia --project=@runic -m Runic --inplace .`.

Run all three locally before pushing; do not merge with a red check. New compat
entries must be bounded (Aqua enforces this).

## Development best practices

### 1. Follow existing high quality libraries

Before designing anything, look at how it has already been done well, and reuse it.
`SparseArrays` (stdlib) defines the user-facing API: function names, argument order,
keyword arguments, storage fields, and the semantics of structural versus numerical
zeros. `CUDA.CUSPARSE` defines the device side: which formats exist, how they are
constructed and named, the index type, and which operations each format supports.
`Metal.jl` defines what the hardware can do. `Krylov.jl` and
`KrylovPreconditioners.jl` define the operator and preconditioner interfaces that
must be satisfied.

Where `SparseArrays` and `CUSPARSE` disagree, follow `CUSPARSE` for device-specific
types and `SparseArrays` for anything a user of this package sees. Reuse their
correctness, validation, and robustness measures — read their test suites and adopt
the cases they cover — rather than inventing our own. A deviation from an
established interface needs a stated reason and a docstring saying so.

### 2. Mathematical clarity in names

Function and variable names match the nomenclature of `SparseArrays`, of the sparse
linear algebra literature, and of the reference implementations: `colptr`, `rowval`,
`nzval`, `nnz`, `nonzeros`, `dropzeros!`, `spmv`, `csr`, `coo`. A reader who knows
Saad or the CSR format should recognize the code. Do not rename an established
concept, and do not abbreviate past the point of recognition.

Avoid special symbols and non-ASCII characters unless they are already used for the
same purpose in the standard library (`I`, `⋅`, `'` are fine because stdlib uses
them; `α`, `β` as variable names are not — write `alpha`, `beta`).

### 3. Docstrings and comments

Docstrings are mathematically precise and concise. State what is computed or stored,
the conditions on the input, and the properties of the output. Use a term only in
its precise sense: *structural zero*, *stored entry*, *sorted*, *symmetric*,
*Hermitian*, *positive definite*, *rank* mean exactly what they mean in the
literature, and *nonzero* means a stored entry, which may hold the value zero. Do
not use a vague word where a precise counterpart exists — not "roughly equal" but a
stated bound; not "fast" but a stated complexity; not "safe" but a stated guarantee.

State asynchrony and determinism explicitly where they apply: whether an operation
returns before the device has finished, and whether repeated calls on the same input
give bit-identical results.

No marketing, no restating the signature in words, no examples unless the usage is
genuinely non-obvious.

Code comments are the exception, not the rule. Add one only where something truly
needs explaining — an index convention, a deviation from the CPU algorithm forced by
the device, a workaround for a Metal limitation — and write it to the same precision
standard as a docstring.

### 4. Device constraints

These are properties of the hardware, not preferences, and they shape every design
decision:

- **No double precision, but everything narrower.** Apple GPUs have no `Float64`
  unit. They do support `Float16`, `BFloat16`, and `Float32`, and the complex types
  built on each, and all of them must work. `Float64` appears only as the element
  type of a CPU reference. The supported set is probed at run time by
  `supported_element_types`, not hard coded, because it varies by device and by
  Metal version.
- **32-bit indices.** `DEFAULT_INDEX_TYPE` is `Int32`, matching `CUSPARSE`. Host
  matrices default to `Int` indices, so conversion changes `Ti`; conversions must be
  checked against `typemax(Int32)`.
- **No scalar indexing.** Never scalar index a device array in library code. Test
  with `Metal.allowscalar(false)`. Follow the reference implementations' policy for
  what user-facing scalar `getindex` does.
- **Asynchrony.** Kernel launches return before the work completes. Synchronize
  before reading a result on the host, before timing, and before comparing.
- **Atomics and reproducibility.** Accumulation order on the device is not fixed.
  Prefer a deterministic ordering; where atomics are used, say so in the docstring
  and test the guarantee that is actually offered.

### 5. Element type genericity

Every format and operation must work for every element type the backend supports,
and adding a new element type must cost nothing beyond one line in
`CANDIDATE_ELEMENT_TYPES` in the test harness.

- Never dispatch on, or annotate an argument with, a concrete floating point type.
  Write `Tv<:Number` and derive everything else from it.
- Derive constants and tolerances from the element type: `zero(Tv)`, `one(Tv)`,
  `realtype(Tv)`, `unit_roundoff(Tv)`. No floating point literals in kernel code, no
  fixed tolerances such as `1e-8`, and no `eps()` without an argument.
- Do not assume the element type is a double precision hardware float. `Float16` and
  `BFloat16` have a unit roundoff of about `5e-4` and `4e-3`; a threshold or a
  convergence test tuned to double precision silently fails for them, and a long
  accumulation in them loses most of its digits.
- Where accumulation in the element type loses too much accuracy — a sparse
  matrix-vector product over a dense row in `Float16` — widen the accumulator
  deliberately and say so in the docstring. Do not widen silently and do not refuse
  the type.

The failure this guards against is real: with `Metal` and `LinearAlgebra` loaded,
`lu(Metal.ones(Float16, 10, 10))` segfaults, even though the array is valid and
arithmetic on it works. Generic fallbacks that reach code assuming BLAS element
types crash rather than erroring. Testing every operation on every supported element
type is what catches this class of failure here instead of in a user's process.

### 6. Correctness is defined by the CPU reference

Every device operation is validated against the `SparseArrays` result for the same
input: exactly for structural results (sparsity pattern, index arrays, `nnz`,
conversions, permutations) and to a stated floating point bound otherwise. A test
that only checks that an operation ran is not a test.
