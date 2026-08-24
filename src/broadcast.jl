# Broadcasting over device sparse matrices, following the CUDA.CUSPARSE
# conventions: a zero-preserving function over one sparse matrix and scalars is
# applied to nzval on the device with the pattern preserved exactly (stored
# zeros included); a function that does not map zero to zero, or a broadcast
# mixing a sparse matrix with a dense array, densifies — the result is a dense
# MtlMatrix, computed on the device. (SparseArrays instead keeps the sparse
# container and stores every entry; the dense container is the CUSPARSE
# convention, adopted by the roadmap.) Sparse-sparse broadcast keeps the union
# pattern in both references and needs a pattern-merge kernel; until that
# lands (pending the Metal.MPS survey) it throws.

"""
    MtlSparseStyle <: Broadcast.AbstractArrayStyle{2}

Broadcast style of the device sparse matrix formats. A broadcast of exactly one
[`AbstractMtlSparseMatrix`](@ref) with scalars materializes sparse when the
function maps zero to zero — applied to the stored values on the device, the
pattern preserved exactly, stored zeros included — and otherwise densifies to a
dense `MtlMatrix` on the device, as `CUDA.CUSPARSE` does (`A .+ 1`, `cos.(A)`,
and `A .* NaN` are dense results, never silent errors). A broadcast combining a
sparse matrix with a dense device array densifies likewise. Broadcasts over
more than one sparse matrix are not yet implemented and throw.
"""
struct MtlSparseStyle <: Broadcast.AbstractArrayStyle{2} end

Base.BroadcastStyle(::Type{<:AbstractMtlSparseMatrix}) = MtlSparseStyle()
MtlSparseStyle(::Val{2}) = MtlSparseStyle()

# A sparse-dense mix resolves to the sparse style, whose materialization
# densifies (below); without this rule the two styles conflict and broadcast
# refuses outright.
Base.BroadcastStyle(s::MtlSparseStyle, ::Metal.MtlArrayStyle) = s

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
            "broadcast over $sparse_count device sparse matrices is not yet " *
                "implemented; one sparse operand combined with scalars or dense " *
                "device arrays is supported"
        )
    )
    A = args[findfirst(a -> a isa AbstractMtlSparseMatrix, args)]
    has_dense = any(a -> a isa AbstractArray && !(a isa AbstractMtlSparseMatrix), args)
    if !has_dense
        fzero = flat.f(
            map(
                a -> a isa AbstractMtlSparseMatrix ? zero(eltype(a)) : scalar_value(a),
                args
            )...
        )
        if iszero(fzero)
            nzval = broadcast(
                flat.f, map(a -> a isa AbstractMtlSparseMatrix ? a.nzval : a, args)...
            )
            return with_nzval(A, nzval)
        end
    end
    # Densifying path (CUSPARSE convention): a non-zero-preserving function or
    # a dense operand gives a dense device result, delegated to Metal's own
    # broadcast machinery.
    mapped = map(a -> a isa AbstractMtlSparseMatrix ? MtlMatrix(a) : a, args)
    return Broadcast.materialize(Broadcast.broadcasted(flat.f, mapped...))
end
