"""
    MtlSparseMatrixCOO{Tv, Ti} <: AbstractMtlSparseMatrix{Tv, Ti}

Device-resident sparse matrix in coordinate storage, corresponding to
`CUDA.CUSPARSE.CuSparseMatrixCOO`. An `m`-by-`n` matrix with `k` stored entries
holds three `MtlVector`s of length `k`: `rowval` and `colval` with the row and
column index of each entry, and `nzval` with its value.

Following `CUSPARSE`, entries are sorted lexicographically by `(row, column)`
and coordinates are not duplicated; like the sortedness assumption of the
compressed formats, this is documented but not validated. Assembly from
arbitrary, duplicated coordinates is the job of `sparse(I, J, V, m, n)`, not of
this constructor.

Construct from a `SparseMatrixCSC` — `MtlSparseMatrixCOO(A)` with index type
[`DEFAULT_INDEX_TYPE`](@ref), or `MtlSparseMatrixCOO{Tv, Ti}(A)` — or from raw
device arrays with `MtlSparseMatrixCOO(m, n, rowval, colval, nzval)`, which
validates that the three arrays have equal length and every index is in range,
and throws `ArgumentError` naming the violated invariant otherwise.

Stored entries with the value zero are preserved, matching `SparseArrays`.
Convert back with `SparseMatrixCSC(A)` or `adapt(Array, A)`; the round trip is
exact.
"""
struct MtlSparseMatrixCOO{Tv, Ti <: Integer} <: AbstractMtlSparseMatrix{Tv, Ti}
    m::Int
    n::Int
    rowval::MtlVector{Ti}
    colval::MtlVector{Ti}
    nzval::MtlVector{Tv}

    function MtlSparseMatrixCOO{Tv, Ti}(
            m::Integer, n::Integer, rowval::MtlVector{Ti},
            colval::MtlVector{Ti}, nzval::MtlVector{Tv}
        ) where {Tv, Ti <: Integer}
        coo_check(m, n, rowval, colval, nzval)
        return new{Tv, Ti}(m, n, rowval, colval, nzval)
    end
end

function MtlSparseMatrixCOO(
        m::Integer, n::Integer, rowval::MtlVector{Ti},
        colval::MtlVector{Ti}, nzval::MtlVector{Tv}
    ) where {Tv, Ti <: Integer}
    return MtlSparseMatrixCOO{Tv, Ti}(m, n, rowval, colval, nzval)
end

# All three arrays live on the device and there is no pointer array, so the
# whole validation runs as device reductions; nothing is transferred.
function coo_check(
        m::Integer, n::Integer, rowval::MtlVector{Ti},
        colval::MtlVector{Ti}, nzval::MtlVector
    ) where {Ti <: Integer}
    dims_check(m, n, Ti)
    length(rowval) == length(colval) == length(nzval) ||
        throw(
        ArgumentError(
            "lengths of rowval ($(length(rowval))), colval ($(length(colval))), " *
                "and nzval ($(length(nzval))) must be equal"
        )
    )
    index_range_check(rowval, m, "rowval")
    index_range_check(colval, n, "colval")
    return nothing
end

function MtlSparseMatrixCOO{Tv, Ti}(A::SparseMatrixCSC) where {Tv, Ti <: Integer}
    m, n = size(A)
    (m <= typemax(Ti) && n <= typemax(Ti)) ||
        throw(ArgumentError("matrix with dimensions ($m, $n) does not fit in Ti = $Ti"))
    At = sparse(transpose(A))
    rowhost = Vector{Ti}(undef, nnz(A))
    for i in 1:m, k in At.colptr[i]:(At.colptr[i + 1] - 1)
        rowhost[k] = Ti(i)
    end
    rowval = MtlVector{Ti}(rowhost)
    colval = MtlVector{Ti}(convert(Vector{Ti}, At.rowval))
    nzval = MtlVector{Tv}(convert(Vector{Tv}, At.nzval))
    return MtlSparseMatrixCOO{Tv, Ti}(m, n, rowval, colval, nzval)
end

function MtlSparseMatrixCOO(A::SparseMatrixCSC{Tv}) where {Tv}
    return MtlSparseMatrixCOO{Tv, DEFAULT_INDEX_TYPE}(A)
end
