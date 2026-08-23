# Benchmarks

Empty for now. This directory will hold direct measurements of performance for the
formats and operations in `src/`.

The goals these benchmarks exist to verify:

1. Sparse operations on the device must run faster than the same operations on
   `SparseArrays` on the CPU once the problem size is large enough, with the
   crossover size measured and reported.
2. Sparse operations on the device must beat dense `MtlArray` operations on the
   same problem starting at a fairly modest size — sparsity must pay for itself
   well before problems become huge.
