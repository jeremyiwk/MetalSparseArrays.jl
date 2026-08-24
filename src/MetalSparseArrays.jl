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

using Adapt: Adapt
using LinearAlgebra: LinearAlgebra
using Metal: Metal, MtlArray, MtlMatrix, MtlVector, thread_position_in_grid
using SparseArrays: SparseArrays, AbstractSparseArray, AbstractSparseMatrix,
    SparseMatrixCSC, nnz, sparse

export AbstractMtlSparseMatrix, MtlSparseMatrixCOO, MtlSparseMatrixCSC,
    MtlSparseMatrixCSR

include("common.jl")
include("csr.jl")
include("csc.jl")
include("coo.jl")
include("conversions.jl")
include("interface.jl")
include("broadcast.jl")
include("kernels/merge_broadcast.jl")

end # module
