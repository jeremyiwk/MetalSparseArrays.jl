"""
    MtlSparseMatrixCSR{Tv, Ti} <: AbstractMtlSparseMatrix{Tv, Ti}

Device-resident sparse matrix in compressed sparse row storage, the primary
device format, corresponding to `CUDA.CUSPARSE.CuSparseMatrixCSR`. An `m`-by-`n`
matrix with `k` stored entries holds three `MtlVector`s: `rowptr` of length
`m + 1` with `rowptr[i]:rowptr[i + 1] - 1` the range of entries of row `i`
(`rowptr[1] == 1`, monotonically nondecreasing, `rowptr[m + 1] == k + 1`),
`colval` of length `k` with the column index of each entry, and `nzval` of
length `k` with its value.

Construct from a `SparseMatrixCSC` — `MtlSparseMatrixCSR(A)` with index type
[`DEFAULT_INDEX_TYPE`](@ref), or `MtlSparseMatrixCSR{Tv, Ti}(A)` — or from raw
device arrays with `MtlSparseMatrixCSR(m, n, rowptr, colval, nzval)`, which
validates the invariants above and that every column index lies in `1:n`, and
throws `ArgumentError` naming the violated invariant otherwise. Column indices
within a row are assumed sorted ascending without duplicates and this is not
validated, matching the assumption `SparseMatrixCSC` makes for its row indices.
Unlike `SparseMatrixCSC`, the index and value arrays must have exactly their
required lengths — oversized buffers are not accepted, so the stored arrays
always mean what the invariants say.

Stored entries with the value zero are preserved, matching `SparseArrays`.
Convert back with `SparseMatrixCSC(A)` or `adapt(Array, A)`; the round trip is
exact, including the sparsity pattern and index order.
"""
struct MtlSparseMatrixCSR{Tv, Ti <: Integer} <: AbstractMtlSparseMatrix{Tv, Ti}
    m::Int
    n::Int
    rowptr::MtlVector{Ti}
    colval::MtlVector{Ti}
    nzval::MtlVector{Tv}

    function MtlSparseMatrixCSR{Tv, Ti}(
            m::Integer, n::Integer, rowptr::MtlVector{Ti},
            colval::MtlVector{Ti}, nzval::MtlVector{Tv}
        ) where {Tv, Ti <: Integer}
        csr_check(m, n, rowptr, colval, nzval)
        return new{Tv, Ti}(m, n, rowptr, colval, nzval)
    end
end

function MtlSparseMatrixCSR(
        m::Integer, n::Integer, rowptr::MtlVector{Ti},
        colval::MtlVector{Ti}, nzval::MtlVector{Tv}
    ) where {Tv, Ti <: Integer}
    return MtlSparseMatrixCSR{Tv, Ti}(m, n, rowptr, colval, nzval)
end

# The pointer invariants are checked on a host copy of `rowptr` (an O(m)
# transfer, accepted at construction time); the column index range is checked by
# a device reduction, so `colval` is never transferred.
function csr_check(
        m::Integer, n::Integer, rowptr::MtlVector{Ti},
        colval::MtlVector{Ti}, nzval::MtlVector
    ) where {Ti <: Integer}
    0 <= m <= typemax(Ti) ||
        throw(ArgumentError("number of rows m = $m is negative or does not fit in Ti = $Ti"))
    0 <= n <= typemax(Ti) ||
        throw(ArgumentError("number of columns n = $n is negative or does not fit in Ti = $Ti"))
    length(rowptr) == m + 1 ||
        throw(ArgumentError("$(length(rowptr)) == length(rowptr) != m + 1 == $(m + 1)"))
    hostptr = Array(rowptr)
    hostptr[1] == 1 || throw(ArgumentError("$(hostptr[1]) == rowptr[1] != 1"))
    for i in 2:(m + 1)
        hostptr[i - 1] <= hostptr[i] ||
            throw(
            ArgumentError(
                "$(hostptr[i - 1]) == rowptr[$(i - 1)] > rowptr[$i] == $(hostptr[i])"
            )
        )
    end
    stored = Int(hostptr[m + 1]) - 1
    length(colval) == stored ||
        throw(ArgumentError("$(length(colval)) == length(colval) != rowptr[m + 1] - 1 == $stored"))
    length(nzval) == stored ||
        throw(ArgumentError("$(length(nzval)) == length(nzval) != rowptr[m + 1] - 1 == $stored"))
    if !isempty(colval)
        nTi = Ti(n)
        inrange = mapreduce(j -> (one(Ti) <= j) & (j <= nTi), &, colval)
        inrange || throw(ArgumentError("colval contains a column index outside 1:$n"))
    end
    return nothing
end

function MtlSparseMatrixCSR{Tv, Ti}(A::SparseMatrixCSC) where {Tv, Ti <: Integer}
    m, n = size(A)
    (m <= typemax(Ti) && n <= typemax(Ti) && nnz(A) + 1 <= typemax(Ti)) ||
        throw(
        ArgumentError(
            "matrix with dimensions ($m, $n) and $(nnz(A)) stored entries does not fit in Ti = $Ti"
        )
    )
    At = sparse(transpose(A))
    rowptr = MtlVector{Ti}(convert(Vector{Ti}, At.colptr))
    colval = MtlVector{Ti}(convert(Vector{Ti}, At.rowval))
    nzval = MtlVector{Tv}(convert(Vector{Tv}, At.nzval))
    return MtlSparseMatrixCSR{Tv, Ti}(m, n, rowptr, colval, nzval)
end

function MtlSparseMatrixCSR(A::SparseMatrixCSC{Tv}) where {Tv}
    return MtlSparseMatrixCSR{Tv, DEFAULT_INDEX_TYPE}(A)
end

"""
    SparseMatrixCSC(A::MtlSparseMatrixCSR{Tv, Ti}) -> SparseMatrixCSC{Tv, Ti}

The host CSC matrix equal to `A`. The conversion is exact: the sparsity pattern,
the ordering of the index arrays, and every stored value (zeros included) are
reproduced, so converting a `SparseMatrixCSC` to the device and back is the
identity up to the index type.
"""
function SparseArrays.SparseMatrixCSC(A::MtlSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    At = SparseMatrixCSC(A.n, A.m, Array(A.rowptr), Array(A.colval), Array(A.nzval))
    return sparse(transpose(At))
end

Adapt.adapt_storage(::Type{Array}, A::MtlSparseMatrixCSR) = SparseMatrixCSC(A)

Base.size(A::MtlSparseMatrixCSR) = (A.m, A.n)

SparseArrays.nnz(A::MtlSparseMatrixCSR) = length(A.nzval)

"""
    nonzeros(A::MtlSparseMatrixCSR)

The `MtlVector` of stored values of `A`, in storage (row-major) order, aliasing
the matrix. Stored entries may hold the value zero.
"""
SparseArrays.nonzeros(A::MtlSparseMatrixCSR) = A.nzval

function Base.summary(io::IO, A::MtlSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    k = nnz(A)
    return print(
        io, A.m, "×", A.n, " MtlSparseMatrixCSR{", Tv, ", ", Ti, "} with ",
        k, " stored ", k == 1 ? "entry" : "entries"
    )
end

Base.show(io::IO, ::MIME"text/plain", A::MtlSparseMatrixCSR) = summary(io, A)
Base.show(io::IO, A::MtlSparseMatrixCSR) = summary(io, A)
