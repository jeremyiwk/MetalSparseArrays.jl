"""
    MetalSparseArrays

Sparse arrays for the Apple Silicon GPU backend.

The package provides the sparse storage formats, array interface, and sparse
linear algebra kernels that `SparseArrays` provides on the CPU and that
`CUDA.CUSPARSE` provides on NVIDIA hardware, so that code written against either
runs on `Metal`, including the operator interface required by iterative solvers such
as `Krylov`.

Storage formats and the operations on them are organized one per file, included
below.
"""
module MetalSparseArrays

using Adapt
using LinearAlgebra
using Metal
using SparseArrays

include("common.jl")

end # module
