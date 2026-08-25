# Broadcasting over device sparse matrices, following the CUDA.CUSPARSE
# conventions: a zero-preserving function over one sparse matrix and scalars is
# applied to nzval on the device with the pattern preserved exactly (stored
# zeros included); a function that does not map zero to zero, or a broadcast
# mixing a sparse matrix with a dense array, densifies — the result is a dense
# MtlMatrix, computed on the device. (SparseArrays instead keeps the sparse
# container and stores every entry; the dense container is the CUSPARSE
# convention, adopted by the roadmap.) Sparse-sparse broadcast keeps the union
# pattern, matching SparseArrays exactly, computed by the device pattern merge
# in src/kernels/merge_broadcast.jl where it applies and by the host fallback
# below otherwise (Metal.MPS has no sparse primitive — surveyed).
# In-place `A .= rhs` follows SparseArrays semantics by running the stdlib
# broadcast on a host mirror and rebinding the destination's storage arrays.

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

# The pattern of `A` (index arrays prefix-copied, so the result is compact)
# with the given value array of length `nnz(A)`.
with_nzval(A::MtlSparseMatrixCSC{<:Any, Ti}, nzval::MtlVector{Tv}) where {Tv, Ti} =
    MtlSparseMatrixCSC{Tv, Ti}(
    A.m, A.n, device_copy(A.colptr), device_copy(view(A.rowval, 1:nnz(A))), nzval
)
with_nzval(A::MtlSparseMatrixCSR{<:Any, Ti}, nzval::MtlVector{Tv}) where {Tv, Ti} =
    MtlSparseMatrixCSR{Tv, Ti}(
    A.m, A.n, device_copy(A.rowptr), device_copy(view(A.colval, 1:nnz(A))), nzval
)
with_nzval(A::MtlSparseMatrixCOO{<:Any, Ti}, nzval::MtlVector{Tv}) where {Tv, Ti} =
    MtlSparseMatrixCOO{Tv, Ti}(A.m, A.n, device_copy(A.rowval), device_copy(A.colval), nzval)

scalar_value(a) = a
scalar_value(a::Base.RefValue) = a[]

function Base.copy(bc::Broadcast.Broadcasted{MtlSparseStyle})
    flat = Broadcast.flatten(bc)
    args = flat.args
    sparse_count = count(a -> a isa AbstractMtlSparseMatrix, args)
    A = args[findfirst(a -> a isa AbstractMtlSparseMatrix, args)]
    has_dense = any(a -> a isa AbstractArray && !(a isa AbstractMtlSparseMatrix), args)
    if !has_dense
        fzero = flat.f(
            map(
                a -> a isa AbstractMtlSparseMatrix ? zero(eltype(a)) : scalar_value(a),
                args
            )...
        )
        if iszero(fzero) && sparse_count == 1
            nzval = broadcast(
                flat.f, map(a -> a isa AbstractMtlSparseMatrix ? stored_nzval(a) : a, args)...
            )
            return with_nzval(A, nzval)
        elseif iszero(fzero)
            # Union-pattern sparse-sparse broadcast, matching SparseArrays
            # exactly. Result format and Ti follow the first sparse operand.
            merged = try_merge_broadcast(flat.f, args)
            merged === nothing || return merged
            return host_union_broadcast(flat.f, args, A)
        end
    end
    # Densifying path (CUSPARSE convention): a non-zero-preserving function or
    # a dense operand gives a dense device result, delegated to Metal's own
    # broadcast machinery.
    mapped = map(a -> a isa AbstractMtlSparseMatrix ? MtlMatrix(a) : a, args)
    return Broadcast.materialize(Broadcast.broadcasted(flat.f, mapped...))
end

# Sparse-sparse broadcast on the host, for the cases outside the domain of
# `try_merge_broadcast`: the stdlib broadcast over host mirrors, moved back in
# the format and index type of `A`.
function host_union_broadcast(f, args::Tuple, A::AbstractMtlSparseMatrix)
    hosts = map(a -> a isa AbstractMtlSparseMatrix ? SparseMatrixCSC(a) : a, args)
    return format_like(A, Broadcast.materialize(Broadcast.broadcasted(f, hosts...)))
end

format_like(::MtlSparseMatrixCSC{<:Any, Ti}, A::SparseMatrixCSC{Tv}) where {Tv, Ti} =
    MtlSparseMatrixCSC{Tv, Ti}(A)
format_like(::MtlSparseMatrixCSR{<:Any, Ti}, A::SparseMatrixCSC{Tv}) where {Tv, Ti} =
    MtlSparseMatrixCSR{Tv, Ti}(A)
format_like(::MtlSparseMatrixCOO{<:Any, Ti}, A::SparseMatrixCSC{Tv}) where {Tv, Ti} =
    MtlSparseMatrixCOO{Tv, Ti}(A)

"""
    copyto!(dest::AbstractMtlSparseMatrix, bc::Broadcasted)

In-place broadcast assignment `dest .= ...` with the exact semantics of
`SparseArrays` for a sparse destination: assigning `0` keeps the pattern with
stored zeros, a nonzero scalar stores every entry, a dense right-hand side
takes the union of the old pattern and the dense nonzeros, and a sparse
right-hand side replaces the pattern. Computed by running the stdlib broadcast
on a host mirror and rebinding the destination's storage arrays (device arrays
cannot resize, so a pattern change rebinds); values are converted to the
destination's element type as the stdlib does.
"""
function Base.copyto!(
        dest::AbstractMtlSparseMatrix, bc::Broadcast.Broadcasted{MtlSparseStyle}
    )
    return host_materialize!(dest, bc)
end

# Base and GPUArrays both have entry points that bypass the destination's
# broadcast style: Base fast-paths a scalar right-hand side (`A .= 0`) through
# fill!, the generic AbstractArray path scalar-indexes (`A .= Matrix`), and
# GPUArrays claims any destination when the right-hand side carries the dense
# device style (`A .= MtlMatrix`). Intercept them all on the destination type
# and route to the host mirror.
function Base.copyto!(
        dest::AbstractMtlSparseMatrix,
        bc::Broadcast.Broadcasted{<:Broadcast.DefaultArrayStyle}
    )
    return host_materialize!(dest, bc)
end

function Base.copyto!(
        dest::AbstractMtlSparseMatrix, bc::Broadcast.Broadcasted{<:Metal.MtlArrayStyle}
    )
    return host_materialize!(dest, bc)
end

function host_materialize!(dest::AbstractMtlSparseMatrix, bc::Broadcast.Broadcasted)
    flat = Broadcast.flatten(bc)
    hostdest = SparseMatrixCSC(dest)
    hosts = map(
        a -> a isa AbstractMtlSparseMatrix ? SparseMatrixCSC(a) :
            a isa MtlArray ? Array(a) : a,
        flat.args
    )
    Broadcast.materialize!(hostdest, Broadcast.broadcasted(flat.f, hosts...))
    rebind!(dest, hostdest)
    return dest
end

# Replace the destination's storage arrays with the (validated) arrays of the
# host result converted to the destination's format and index type.
function rebind!(dest::F, A::SparseMatrixCSC) where {F <: AbstractMtlSparseMatrix}
    tmp = format_like(dest, A)
    if dest isa MtlSparseMatrixCSC
        dest.colptr = tmp.colptr
        dest.rowval = tmp.rowval
        dest.nnz = tmp.nnz
    elseif dest isa MtlSparseMatrixCSR
        dest.rowptr = tmp.rowptr
        dest.colval = tmp.colval
        dest.nnz = tmp.nnz
    else
        dest.rowval = tmp.rowval
        dest.colval = tmp.colval
    end
    dest.nzval = tmp.nzval
    return dest
end
