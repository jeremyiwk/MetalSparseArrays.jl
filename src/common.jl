"""
    AbstractMtlSparseMatrix{Tv, Ti} <: SparseArrays.AbstractSparseMatrix{Tv, Ti}

Supertype of the device-resident sparse matrix formats. `Tv` is the element type
of the stored values and `Ti <: Integer` the type of the index arrays. Every
format stores its index and value arrays as `MtlVector`s and follows the
semantics of `SparseArrays`: a "nonzero" is a stored entry, which may hold the
value zero.
"""
abstract type AbstractMtlSparseMatrix{Tv, Ti <: Integer} <: AbstractSparseMatrix{Tv, Ti} end

# Field convention for AbstractMtlSparseMatrix subtypes, which the generic
# methods below rely on: every format stores its dimensions as `m::Int` and
# `n::Int` and its values as `nzval::MtlVector{Tv}`; the compressed formats
# additionally store their entry count as `nnz::Int` (their buffers may be
# longer) and override `SparseArrays.nnz` below.

Base.size(A::AbstractMtlSparseMatrix) = (A.m, A.n)

SparseArrays.nnz(A::AbstractMtlSparseMatrix) = length(A.nzval)

"""
    nonzeros(A::AbstractMtlSparseMatrix)

The `MtlVector` of stored values of `A`, in the storage order of the format
(row-major for CSR and COO, column-major for CSC), aliasing the matrix. Stored
entries may hold the value zero. As with `SparseMatrixCSC`, the buffer may be
longer than `nnz(A)`; entries past `nnz(A)` are unspecified.
"""
SparseArrays.nonzeros(A::AbstractMtlSparseMatrix) = A.nzval

# The stored prefix of the value buffer, for device code that maps over the
# stored entries of a possibly oversized matrix.
stored_nzval(A::AbstractMtlSparseMatrix) = view(A.nzval, 1:nnz(A))

function Base.summary(io::IO, A::AbstractMtlSparseMatrix{Tv, Ti}) where {Tv, Ti}
    k = nnz(A)
    return print(
        io, A.m, "×", A.n, " ", nameof(typeof(A)), "{", Tv, ", ", Ti, "} with ",
        k, " stored ", k == 1 ? "entry" : "entries"
    )
end

Base.show(io::IO, ::MIME"text/plain", A::AbstractMtlSparseMatrix) = summary(io, A)
Base.show(io::IO, A::AbstractMtlSparseMatrix) = summary(io, A)

Adapt.adapt_storage(::Type{Array}, A::AbstractMtlSparseMatrix) = SparseMatrixCSC(A)

function dims_check(m::Integer, n::Integer, ::Type{Ti}) where {Ti <: Integer}
    0 <= m <= typemax(Ti) ||
        throw(ArgumentError("number of rows m = $m is negative or does not fit in Ti = $Ti"))
    0 <= n <= typemax(Ti) ||
        throw(ArgumentError("number of columns n = $n is negative or does not fit in Ti = $Ti"))
    return nothing
end

"""
    compressed_check(major, minor, ptr, idx, nzval, ptrname, idxname) -> stored

Validate the compressed-storage invariants shared by the CSR (`major == m`) and
CSC (`major == n`) formats and return the stored entry count
`stored = ptr[end] - 1`: `ptr` has length `major + 1`, starts at one, and is
monotonically nondecreasing; `idx` and `nzval` have at least `stored` entries —
longer buffers are accepted and their tails ignored, exactly as
`SparseMatrixCSC` accepts `rowval`/`nzval` longer than `nnz` — and every index
in `idx[1:stored]` lies in `1:minor`. Throws `ArgumentError` naming the
violated invariant, with `ptrname`/`idxname` naming the arrays in the format's
own vocabulary. The pointer invariants are checked on a host copy of `ptr` (an
`O(major)` transfer, accepted at construction time); the index range is checked
by a device reduction, so `idx` is never transferred. Sortedness within a major
slice is an unchecked documented assumption, matching
`SparseArrays.sparse_check`.
"""
function compressed_check(
        major::Integer, minor::Integer, ptr::MtlVector{Ti}, idx::MtlVector{Ti},
        nzval::MtlVector, ptrname::String, idxname::String
    ) where {Ti <: Integer}
    length(ptr) == major + 1 ||
        throw(ArgumentError("$(length(ptr)) == length($ptrname) != $(major + 1)"))
    hostptr = Array(ptr)
    hostptr[1] == 1 || throw(ArgumentError("$(hostptr[1]) == $ptrname[1] != 1"))
    for i in 2:(major + 1)
        hostptr[i - 1] <= hostptr[i] ||
            throw(
            ArgumentError(
                "$(hostptr[i - 1]) == $ptrname[$(i - 1)] > $ptrname[$i] == $(hostptr[i])"
            )
        )
    end
    stored = Int(hostptr[major + 1]) - 1
    length(idx) >= stored ||
        throw(ArgumentError("$(length(idx)) == length($idxname) < $ptrname[end] - 1 == $stored"))
    length(nzval) >= stored ||
        throw(ArgumentError("$(length(nzval)) == length(nzval) < $ptrname[end] - 1 == $stored"))
    index_range_check(view(idx, 1:stored), minor, idxname)
    return stored
end

# Checks `1 <= idx[k] <= bound` for every entry by a device reduction; `idx` is
# never transferred to the host.
function index_range_check(idx::AbstractVector{Ti}, bound::Integer, name::String) where {Ti}
    isempty(idx) && return nothing
    b = Ti(bound)
    inrange = mapreduce(j -> (one(Ti) <= j) & (j <= b), &, idx)
    inrange || throw(ArgumentError("$name contains an index outside 1:$bound"))
    return nothing
end

"""
    device_copy(v)

A copy of the device vector (or contiguous view of one) `v` as an `MtlVector`,
computed by an identity broadcast on the device. `Base.copy` on an `MtlArray`
synchronizes the whole queue and then copies on the host (`Metal.jl`'s
GPU-to-GPU `unsafe_copyto!` is a raw memcpy that must not overlap pending
kernels), costing on the order of 100 us per call; the broadcast stays on the
device, is ordered after previously launched kernels, and returns without
waiting. Used wherever library code copies a storage array; copying a prefix
view compacts an oversized buffer.
"""
device_copy(v::Union{MtlVector, SubArray{<:Any, 1, <:MtlVector}}) = identity.(v)

"""
    Unchecked

Internal sentinel whose singleton `unchecked` selects the format inner
constructors that skip invariant validation. For device arrays produced by a
kernel that guarantees the invariants by construction, where validation would
force a host synchronization; this is the same line `SparseArrays` draws
between its validating user-facing constructors and the direct construction
its own kernels use. Public constructors always validate. Not exported.
"""
struct Unchecked end

const unchecked = Unchecked()

"""
    DEFAULT_INDEX_TYPE

The integer type used for row, column, and pointer arrays of device-resident sparse
matrices, `Int32`. This matches the index type of `CUDA.CUSPARSE` and avoids the
memory and bandwidth cost of 64-bit indices, at the cost of restricting a stored
dimension or nonzero count to `typemax(Int32)`. `SparseArrays.SparseMatrixCSC`
defaults to `Int` on the CPU, so conversion between host and device changes `Ti`.
"""
const DEFAULT_INDEX_TYPE = Int32

"""
    indextype(A)

The integer type `Ti` used for the index arrays of the sparse array `A`.
"""
indextype(::AbstractSparseArray{<:Any, Ti}) where {Ti} = Ti
indextype(::Type{<:AbstractSparseArray{<:Any, Ti}}) where {Ti} = Ti

"""
    realtype(T)

The real floating point type used for magnitudes of elements of type `T`, i.e.
`real(float(T))`. Used for norms, tolerances, and convergence criteria.
"""
realtype(::Type{T}) where {T <: Number} = real(float(T))
realtype(x) = realtype(typeof(x))

"""
    unit_roundoff(T)

The unit roundoff `u = eps(realtype(T)) / 2` of the floating point type associated
with `T`.
"""
unit_roundoff(::Type{T}) where {T <: Number} = eps(realtype(T)) / 2
unit_roundoff(x) = unit_roundoff(typeof(x))
