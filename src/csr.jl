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
mutable struct MtlSparseMatrixCSR{Tv, Ti <: Integer} <: AbstractMtlSparseMatrix{Tv, Ti}
    const m::Int
    const n::Int
    rowptr::MtlVector{Ti}
    colval::MtlVector{Ti}
    nzval::MtlVector{Tv}

    function MtlSparseMatrixCSR{Tv, Ti}(
            m::Integer, n::Integer, rowptr::MtlVector{Ti},
            colval::MtlVector{Ti}, nzval::MtlVector{Tv}
        ) where {Tv, Ti <: Integer}
        dims_check(m, n, Ti)
        compressed_check(m, n, rowptr, colval, nzval, "rowptr", "colval")
        return new{Tv, Ti}(m, n, rowptr, colval, nzval)
    end

    # Unchecked construction, for arrays a kernel guarantees by construction;
    # see the docstring of `Unchecked` in common.jl.
    function MtlSparseMatrixCSR{Tv, Ti}(
            ::Unchecked, m::Integer, n::Integer, rowptr::MtlVector{Ti},
            colval::MtlVector{Ti}, nzval::MtlVector{Tv}
        ) where {Tv, Ti <: Integer}
        return new{Tv, Ti}(m, n, rowptr, colval, nzval)
    end
end

function MtlSparseMatrixCSR(
        m::Integer, n::Integer, rowptr::MtlVector{Ti},
        colval::MtlVector{Ti}, nzval::MtlVector{Tv}
    ) where {Tv, Ti <: Integer}
    return MtlSparseMatrixCSR{Tv, Ti}(m, n, rowptr, colval, nzval)
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
