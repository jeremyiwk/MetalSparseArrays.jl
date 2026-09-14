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
# below otherwise (the surveyed Metal wrappers provide no sparse merge).
# In-place `A .= rhs` uses device paths for scalar zero and zero-preserving
# self broadcasts, and a host mirror for the remaining assignment semantics.

"""
    MtlSparseStyle <: Broadcast.AbstractArrayStyle{2}

Broadcast style of the device sparse matrix formats. A broadcast of exactly one
[`AbstractMtlSparseMatrix`](@ref) with scalars materializes sparse when the
function maps zero to zero — applied to the stored values on the device, the
pattern preserved exactly, stored zeros included — and otherwise densifies to a
dense `MtlMatrix` on the device, as `CUDA.CUSPARSE` does (`A .+ 1`, `cos.(A)`,
and `A .* NaN` are dense results, never silent errors). A broadcast combining a
sparse matrix with a dense device array densifies likewise. Sparse-sparse
broadcasts merge the union pattern and drop computed zeros, using a device merge
for two equal-sized operands with matching index types.
"""
struct MtlSparseStyle <: Broadcast.AbstractArrayStyle{2} end

Base.BroadcastStyle(::Type{<:AbstractMtlSparseMatrix}) = MtlSparseStyle()
MtlSparseStyle(::Val{2}) = MtlSparseStyle()

# A sparse-dense mix resolves to the sparse style, whose materialization
# densifies (below); without this rule the two styles conflict and broadcast
# refuses outright.
Base.BroadcastStyle(s::MtlSparseStyle, ::Metal.MtlArrayStyle) = s

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
right-hand side replaces the pattern. Zero scalar assignment updates stored values
asynchronously; zero-preserving unary/scalar expressions on the destination use
device compaction, dropping computed zeros and reading back the resulting count.
Other assignments use a host mirror. Pattern changes rebind storage arrays, so
previously obtained storage aliases do not follow the replacement. Values are
converted to the destination's element type as the stdlib does.
"""
function Base.copyto!(
        dest::AbstractMtlSparseMatrix, bc::Broadcast.Broadcasted{MtlSparseStyle}
    )
    return materialize_sparse!(dest, bc)
end

# Base and GPUArrays both have entry points that bypass the destination's
# broadcast style: Base fast-paths a scalar right-hand side (`A .= 0`) through
# fill!, the generic AbstractArray path scalar-indexes (`A .= Matrix`), and
# GPUArrays claims any destination when the right-hand side carries the dense
# device style (`A .= MtlMatrix`). Intercept them all on the destination type
# and route to the shared assignment implementation.
function Base.copyto!(
        dest::AbstractMtlSparseMatrix,
        bc::Broadcast.Broadcasted{<:Broadcast.DefaultArrayStyle}
    )
    return materialize_sparse!(dest, bc)
end

function Base.copyto!(
        dest::AbstractMtlSparseMatrix, bc::Broadcast.Broadcasted{<:Metal.MtlArrayStyle}
    )
    return materialize_sparse!(dest, bc)
end

function materialize_sparse!(dest::AbstractMtlSparseMatrix, bc::Broadcast.Broadcasted)
    flat = Broadcast.flatten(bc)
    args = flat.args
    if all(a -> !(a isa AbstractArray), args)
        value = flat.f(map(scalar_value, args)...)
        if iszero(value)
            fill!(stored_nzval(dest), value)
            return dest
        end
    elseif count(a -> a isa AbstractMtlSparseMatrix, args) == 1 &&
            all(a -> !(a isa AbstractArray) || a === dest, args)
        flat.f === identity && length(args) == 1 && return dest
        slots = map(a -> a === dest ? MergeSlot() : scalar_value(a), args)
        f = flat.f
        g = (a, b) -> f(substitute_slots(slots, (a,))...)
        if isbits(g) && iszero(g(zero(eltype(dest)), zero(eltype(dest))))
            Tv = Base.promote_op(g, eltype(dest), eltype(dest))
            if Tv === eltype(dest) &&
                    2nnz(dest) + 1 <= typemax(indextype(dest))
                # Unlike out-of-place unary broadcast, assignment drops computed zeros.
                result = merge_broadcast(g, dest, dest)
                rebind_storage!(dest, result, result.nzval)
                return dest
            end
        end
    end
    return host_materialize!(dest, bc)
end

function host_materialize!(dest::AbstractMtlSparseMatrix, bc::Broadcast.Broadcasted)
    flat = Broadcast.flatten(bc)
    hostdest = SparseMatrixCSC(dest)
    hosts = map(
        a -> a isa AbstractMtlSparseMatrix ? SparseMatrixCSC(a) :
            a isa Metal.WrappedMtlArray ? Array(a) : a,
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
    return rebind_storage!(dest, tmp, tmp.nzval)
end

function rebind_storage!(dest, tmp, values)
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
    dest.nzval = values
    return dest
end
