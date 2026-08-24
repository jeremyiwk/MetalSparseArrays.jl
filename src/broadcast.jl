# Broadcasting over stored values, following the CUSPARSE policy: one device
# sparse matrix combined with scalars, the function applied to nzval on the
# device, the pattern preserved exactly. A function that does not map zero to
# zero would densify; that is an error here, never a silent densification.
# Sparse-sparse and sparse-dense broadcasts are not implemented (the former
# needs the pattern-merge kernels of Phase 3).

"""
    MtlSparseStyle <: Broadcast.AbstractArrayStyle{2}

Broadcast style of the device sparse matrix formats. Broadcasts resolve to this
style when they combine one [`AbstractMtlSparseMatrix`](@ref) with scalars;
materializing applies the function to the stored values on the device and
keeps the sparsity pattern (stored zeros included). A function `f` with
`f(0) != 0` throws `ArgumentError` rather than densifying.
"""
struct MtlSparseStyle <: Broadcast.AbstractArrayStyle{2} end

Base.BroadcastStyle(::Type{<:AbstractMtlSparseMatrix}) = MtlSparseStyle()
MtlSparseStyle(::Val{2}) = MtlSparseStyle()

with_nzval(A::MtlSparseMatrixCSC{<:Any, Ti}, nzval::MtlVector{Tv}) where {Tv, Ti} =
    MtlSparseMatrixCSC{Tv, Ti}(A.m, A.n, copy(A.colptr), copy(A.rowval), nzval)
with_nzval(A::MtlSparseMatrixCSR{<:Any, Ti}, nzval::MtlVector{Tv}) where {Tv, Ti} =
    MtlSparseMatrixCSR{Tv, Ti}(A.m, A.n, copy(A.rowptr), copy(A.colval), nzval)
with_nzval(A::MtlSparseMatrixCOO{<:Any, Ti}, nzval::MtlVector{Tv}) where {Tv, Ti} =
    MtlSparseMatrixCOO{Tv, Ti}(A.m, A.n, copy(A.rowval), copy(A.colval), nzval)

scalar_value(a) = a
scalar_value(a::Base.RefValue) = a[]

function Base.copy(bc::Broadcast.Broadcasted{MtlSparseStyle})
    flat = Broadcast.flatten(bc)
    args = flat.args
    sparse_count = count(a -> a isa AbstractMtlSparseMatrix, args)
    sparse_count == 1 || throw(
        ArgumentError(
            "broadcast over $sparse_count device sparse matrices is not implemented; " *
                "exactly one sparse operand combined with scalars is supported"
        )
    )
    any(a -> a isa AbstractArray && !(a isa AbstractMtlSparseMatrix), args) && throw(
        ArgumentError(
            "broadcast combining a device sparse matrix with a dense array " *
                "is not implemented"
        )
    )
    A = args[findfirst(a -> a isa AbstractMtlSparseMatrix, args)]
    fzero = flat.f(
        map(a -> a isa AbstractMtlSparseMatrix ? zero(eltype(a)) : scalar_value(a), args)...
    )
    iszero(fzero) || throw(
        ArgumentError(
            "broadcast function maps zero to $fzero != 0, so the result would be " *
                "dense; broadcasting over a device sparse matrix must preserve " *
                "zeros — densify explicitly with MtlMatrix(A) instead"
        )
    )
    nzval = broadcast(flat.f, map(a -> a isa AbstractMtlSparseMatrix ? a.nzval : a, args)...)
    return with_nzval(A, nzval)
end
