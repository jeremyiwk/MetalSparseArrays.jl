# References

The bounded set of sources this project works from. The survey step of every
feature consults these and only these; adding a source is a deliberate decision
recorded here with a reason, not an ambient accumulation. The collection below is
sufficient for every phase of the roadmap.

## Literature

- **Saad, *Iterative Methods for Sparse Linear Systems* (2nd ed.)** — freely
  available from the author; the reference for sparse formats, sparse
  matrix-vector products, Krylov methods, and the `ic0`/`ilu0` preconditioners
  of Phase 6.
- **Golub & Van Loan, *Matrix Computations* (4th ed.), ch. 11** — Krylov
  methods from the dense-linear-algebra side; shared shelf with
  `AlgorithmicNLA.jl`.

## Code and documentation

- **`SparseArrays` (stdlib)** — source and documentation; defines the
  user-facing API and the semantics of structural versus numerical zeros. Its
  test suite is the model for what to validate.
- **`CUDA.jl`'s `CUSPARSE` wrappers and the NVIDIA cuSPARSE documentation** —
  define the device side: formats, naming, index types, and which operations
  each format supports; the documented behavior is the parity target.
- **`Metal.jl` and the Metal Performance Shaders documentation** — what the
  hardware and Apple's libraries actually provide; re-checked each `audit-code`
  pass because MPS coverage grows by release.
- **`Krylov.jl` and `KrylovPreconditioners.jl`** — the operator and
  preconditioner interfaces Phase 4 and Phase 6 must satisfy; their docs state
  the required methods precisely.

## Deliberately not in the set

Davis, *Direct Methods for Sparse Linear Systems* — direct sparse factorization
is a stated non-goal of the roadmap. General GPU-computing texts add nothing
over the `Metal.jl`/MPS documentation for this package's needs. Grow this page
only when a phase demonstrably needs a source the set lacks.
