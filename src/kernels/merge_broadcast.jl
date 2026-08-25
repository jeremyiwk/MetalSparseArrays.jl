# The device pattern merge behind sparse-sparse broadcast: `C = g.(A, B)` for
# two matrices in the same compressed format and a scalar function `g` with
# `g(0, 0) == 0`, merged slice by slice (row by row for CSR, column by column
# for CSC) by a two-pointer merge over the sorted minor indices. This is the
# generic pointwise form of what `CUDA.CUSPARSE` provides as `csrgeam2` for
# `alpha * A + beta * B`, built as cuSPARSE builds it: a count pass, an
# exclusive scan, and a fill pass. It replaces the host `SparseArrays` fallback
# that `src/broadcast.jl` used before (retained as `host_union_broadcast` for
# the cases outside the merge's domain).
#
# The structural semantics are those of `SparseArrays`, measured rather than
# assumed: `g` is evaluated at every position of the union pattern — including
# `g(a, 0)` where only `A` stores an entry, so `NaN * 0` is `NaN` and is kept —
# and every position whose computed value satisfies `iszero` is dropped. A
# stored zero is therefore not preserved unless `g` maps it to something
# nonzero, and exact cancellation (`A .- A`) yields a structurally empty
# result. Dropping is why the count pass must evaluate `g`: the result size is
# not the union size.

"""
    MergeSlot

Placeholder marking the positions of the two sparse operands in the flattened
argument tuple of a broadcast, so that [`MergeFunction`](@ref) can rebuild the
full argument list around the two merged values.
"""
struct MergeSlot end

# Rebuild the slot tuple with the two `MergeSlot`s replaced by the entries of
# `vals` in order. Recursion over the tuple types, so the substitution is
# resolved at compile time and stays inferable inside a kernel.
substitute_slots(::Tuple{}, ::Tuple{}) = ()
function substitute_slots(slots::Tuple, vals::Tuple)
    head = first(slots)
    return head isa MergeSlot ?
        (first(vals), substitute_slots(Base.tail(slots), Base.tail(vals))...) :
        (head, substitute_slots(Base.tail(slots), vals)...)
end

"""
    MergeFunction(f, slots)

The two-argument function `(a, b) -> f(slots...)` with the two
[`MergeSlot`](@ref) entries of `slots` replaced by `a` and `b`. `slots` is the
flattened argument tuple of a broadcast with its two sparse operands replaced
by `MergeSlot()` and its scalar arguments kept by value, so the whole broadcast
expression travels into a kernel as one `isbits` value.
"""
struct MergeFunction{F, S <: Tuple} <: Function
    f::F
    slots::S
end

(g::MergeFunction)(a, b) = g.f(substitute_slots(g.slots, (a, b))...)

## COV_EXCL_START

# Two-pointer merge of slice `i` of two compressed operands: walks the union
# of the two sorted minor-index lists, evaluates `g` at every union position
# (against a zero for the side that stores nothing there), and calls
# `emit(state, j, v)` for each position whose value is kept, threading `state`
# through and returning its final value. The single traversal shared by the
# count and fill kernels is what guarantees the fill writes exactly the number
# of entries the count pass counted.
@inline function merge_slice(
        emit::E, state, g, ptrA, idxA, valA, ptrB, idxB, valB, i
    ) where {E}
    @inbounds begin
        pa, paend = Int(ptrA[i]), Int(ptrA[i + 1])
        pb, pbend = Int(ptrB[i]), Int(ptrB[i + 1])
        za, zb = zero(eltype(valA)), zero(eltype(valB))
        while pa < paend || pb < pbend
            if pb >= pbend || (pa < paend && idxA[pa] < idxB[pb])
                j = idxA[pa]
                v = g(valA[pa], zb)
                pa += 1
            elseif pa >= paend || idxB[pb] < idxA[pa]
                j = idxB[pb]
                v = g(za, valB[pb])
                pb += 1
            else
                j = idxA[pa]
                v = g(valA[pa], valB[pb])
                pa += 1
                pb += 1
            end
            iszero(v) || (state = emit(state, j, v))
        end
    end
    return state
end

function merge_count_kernel!(counts, g, ptrA, idxA, valA, ptrB, idxB, valB, major)
    i = Int(thread_position_in_grid().x)
    i <= major || return nothing
    kept = merge_slice(
        (count, j, v) -> count + one(eltype(counts)),
        zero(eltype(counts)), g, ptrA, idxA, valA, ptrB, idxB, valB, i
    )
    @inbounds counts[i] = kept
    return nothing
end

function merge_fill_kernel!(idxC, valC, g, ptrC, ptrA, idxA, valA, ptrB, idxB, valB, major)
    i = Int(thread_position_in_grid().x)
    i <= major || return nothing
    write_entry = (k, j, v) -> @inbounds begin
        idxC[k] = j
        valC[k] = v
        k + 1
    end
    merge_slice(
        write_entry, Int(@inbounds ptrC[i]),
        g, ptrA, idxA, valA, ptrB, idxB, valB, i
    )
    return nothing
end

## COV_EXCL_STOP

# One thread per compressed slice, with the threadgroup size taken from the
# compiled pipeline and capped by the work size — the launch idiom of
# `Metal.jl`'s own broadcast. A slice is merged by a single thread, so the
# merge is as slow as its fullest slice; splitting long slices across threads
# is a later optimization for the benchmark suite to motivate.
function launch_per_slice(kernel, major::Integer, args...)
    threads = min(Int(major), kernel.pipeline.maxTotalThreadsPerThreadgroup)
    groups = cld(Int(major), threads)
    kernel(args...; threads, groups)
    return nothing
end

"""
    merge_compressed(g, major, Ti, ptrA, idxA, valA, ptrB, idxB, valB)

The compressed storage arrays `(ptrC, idxC, valC)` of `C = g.(A, B)`, where the
two operands are given by the compressed arrays of a common format with `major`
slices and `g` is a two-argument function with `g(0, 0) == 0`. Minor indices
within a slice are assumed sorted ascending without duplicates — the invariant
the formats document — and the result satisfies the same invariant, so the
output arrays are valid for unchecked construction.

Two device passes with a scan between them: a count kernel sizes each slice of
the result (evaluating `g`, because computed zeros are dropped),
[`ptr_scan!`](@ref) turns the counts into `ptrC`, and a fill kernel writes
each slice at its offset. `g` is evaluated twice per union position, once per
pass; the
alternative, materializing uncompacted values, would cost union-sized device
buffers. No atomics are involved and each thread writes a disjoint output
range, so repeated calls on the same input are bit-identical. The host reads
back exactly one element (`ptrC[end]`, to size the output); that read
synchronizes, and the fill kernel launched after it is asynchronous.
"""
function merge_compressed(
        g, major::Integer, ::Type{Ti},
        ptrA::MtlVector{Ti}, idxA::MtlVector{Ti}, valA::MtlVector,
        ptrB::MtlVector{Ti}, idxB::MtlVector{Ti}, valB::MtlVector
    ) where {Ti <: Integer}
    Tv = Base.promote_op(g, eltype(valA), eltype(valB))
    ptrC = MtlVector{Ti}(undef, major + 1)
    if major > 0
        counts = MtlVector{Ti}(undef, major)
        kernel = Metal.@metal launch = false merge_count_kernel!(
            counts, g, ptrA, idxA, valA, ptrB, idxB, valB, major
        )
        launch_per_slice(kernel, major, counts, g, ptrA, idxA, valA, ptrB, idxB, valB, major)
        ptr_scan!(ptrC, counts, one(Ti))
    else
        fill!(view(ptrC, 1:1), one(Ti))
    end
    stored = Int(Array(view(ptrC, (major + 1):(major + 1)))[1]) - 1
    idxC = MtlVector{Ti}(undef, stored)
    valC = MtlVector{Tv}(undef, stored)
    if stored > 0
        kernel = Metal.@metal launch = false merge_fill_kernel!(
            idxC, valC, g, ptrC, ptrA, idxA, valA, ptrB, idxB, valB, major
        )
        launch_per_slice(
            kernel, major, idxC, valC, g, ptrC, ptrA, idxA, valA, ptrB, idxB, valB, major
        )
    end
    return ptrC, idxC, valC
end

"""
    merge_broadcast(g, A::AbstractMtlSparseMatrix, B::AbstractMtlSparseMatrix)

`C = g.(A, B)` computed on the device by [`merge_compressed`](@ref), for a
two-argument function `g` with `g(0, 0) == 0` and operands of equal dimensions
and equal index type. The result takes the format and index type of `A` and
the element type `g` produces on the operands' element types, and agrees
exactly — pattern, index arrays, and stored values — with
`broadcast(g, SparseMatrixCSC(A), SparseMatrixCSC(B))`, deterministically.

The merge runs over columns when `A` is CSC and over rows otherwise, so an
operand or result in another format is converted first; the merge itself is
always on the device, but the CSR/CSC and COO/CSC conversions currently pass
through the host, as their docstrings in `src/conversions.jl` state.
"""
function merge_broadcast(
        g, A::MtlSparseMatrixCSC{<:Any, Ti}, B::AbstractMtlSparseMatrix{<:Any, Ti}
    ) where {Ti}
    Bc = as_csc(B)
    colptr, rowval, nzval = merge_compressed(
        g, A.n, Ti, A.colptr, A.rowval, A.nzval, Bc.colptr, Bc.rowval, Bc.nzval
    )
    return MtlSparseMatrixCSC{eltype(nzval), Ti}(unchecked, A.m, A.n, colptr, rowval, nzval)
end

function merge_broadcast(
        g, A::MtlSparseMatrixCSR{<:Any, Ti}, B::AbstractMtlSparseMatrix{<:Any, Ti}
    ) where {Ti}
    Bc = as_csr(B)
    rowptr, colval, nzval = merge_compressed(
        g, A.m, Ti, A.rowptr, A.colval, A.nzval, Bc.rowptr, Bc.colval, Bc.nzval
    )
    return MtlSparseMatrixCSR{eltype(nzval), Ti}(unchecked, A.m, A.n, rowptr, colval, nzval)
end

function merge_broadcast(
        g, A::MtlSparseMatrixCOO{<:Any, Ti}, B::AbstractMtlSparseMatrix{<:Any, Ti}
    ) where {Ti}
    return MtlSparseMatrixCOO(merge_broadcast(g, MtlSparseMatrixCSR(A), B))
end

# The sparse operands of a flattened argument tuple, by recursion over the
# tuple type so the result count and types are inferable.
sparse_operands(::Tuple{}) = ()
function sparse_operands(args::Tuple)
    head = first(args)
    return head isa AbstractMtlSparseMatrix ?
        (head, sparse_operands(Base.tail(args))...) :
        sparse_operands(Base.tail(args))
end

"""
    try_merge_broadcast(f, args)

`f.(args...)` computed by [`merge_broadcast`](@ref), or `nothing` when the
device merge does not apply and the caller must use the host fallback. `args`
is the flattened argument tuple of a broadcast whose value at all-zero sparse
arguments is zero and whose non-array arguments are scalars.

The merge applies when there are exactly two sparse operands of equal
dimensions and equal index type `Ti`, every scalar argument and the resulting
[`MergeFunction`](@ref) are `isbits`, the element type `f` produces is concrete
and `isbits`, and `nnz(A) + nnz(B) + 1 <= typemax(Ti)` (an upper bound on the
result's pointer values, checked on the host because the device scan could
overflow silently). Everything else — three or more sparse operands, a
shape-expanding broadcast, a scalar or result the device cannot hold — returns
`nothing`.
"""
function try_merge_broadcast(f::F, args::Tuple) where {F}
    operands = sparse_operands(args)
    length(operands) == 2 || return nothing
    A, B = operands
    size(A) == size(B) || return nothing
    indextype(A) === indextype(B) || return nothing
    nnz(A) + nnz(B) + 1 <= typemax(indextype(A)) || return nothing
    slots = map(a -> a isa AbstractMtlSparseMatrix ? MergeSlot() : scalar_value(a), args)
    g = MergeFunction(f, slots)
    isbits(g) || return nothing
    Tv = Base.promote_op(g, eltype(A), eltype(B))
    (isconcretetype(Tv) && isbitstype(Tv)) || return nothing
    return merge_broadcast(g, A, B)
end
