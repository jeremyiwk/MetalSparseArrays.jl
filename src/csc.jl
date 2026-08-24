"""
    MtlSparseMatrixCSC{Tv, Ti} <: AbstractMtlSparseMatrix{Tv, Ti}

Device-resident sparse matrix in compressed sparse column storage, mirroring
`SparseMatrixCSC` and corresponding to `CUDA.CUSPARSE.CuSparseMatrixCSC`. An
`m`-by-`n` matrix with `k` stored entries holds three `MtlVector`s: `colptr` of
length `n + 1` with `colptr[j]:colptr[j + 1] - 1` the range of entries of column
`j` (`colptr[1] == 1`, monotonically nondecreasing, `colptr[n + 1] == k + 1`),
`rowval` of length `k` with the row index of each entry, and `nzval` of length
`k` with its value.

Construct from a `SparseMatrixCSC` — `MtlSparseMatrixCSC(A)` with index type
[`DEFAULT_INDEX_TYPE`](@ref), or `MtlSparseMatrixCSC{Tv, Ti}(A)`; the transfer
copies the three arrays directly, with no reordering — or from raw device arrays
with `MtlSparseMatrixCSC(m, n, colptr, rowval, nzval)`, which validates the
invariants above and that every row index lies in `1:m`, and throws
`ArgumentError` naming the violated invariant otherwise. Row indices within a
column are assumed sorted ascending without duplicates and this is not
validated, matching `SparseMatrixCSC`. Unlike `SparseMatrixCSC`, the index and
value arrays must have exactly their required lengths — oversized buffers are
not accepted.

Stored entries with the value zero are preserved, matching `SparseArrays`.
Convert back with `SparseMatrixCSC(A)` or `adapt(Array, A)`; the round trip is
exact. `adapt(MtlArray, A)` on a host `SparseMatrixCSC` produces this format.
"""
struct MtlSparseMatrixCSC{Tv, Ti <: Integer} <: AbstractMtlSparseMatrix{Tv, Ti}
    m::Int
    n::Int
    colptr::MtlVector{Ti}
    rowval::MtlVector{Ti}
    nzval::MtlVector{Tv}

    function MtlSparseMatrixCSC{Tv, Ti}(
            m::Integer, n::Integer, colptr::MtlVector{Ti},
            rowval::MtlVector{Ti}, nzval::MtlVector{Tv}
        ) where {Tv, Ti <: Integer}
        dims_check(m, n, Ti)
        compressed_check(n, m, colptr, rowval, nzval, "colptr", "rowval")
        return new{Tv, Ti}(m, n, colptr, rowval, nzval)
    end
end

function MtlSparseMatrixCSC(
        m::Integer, n::Integer, colptr::MtlVector{Ti},
        rowval::MtlVector{Ti}, nzval::MtlVector{Tv}
    ) where {Tv, Ti <: Integer}
    return MtlSparseMatrixCSC{Tv, Ti}(m, n, colptr, rowval, nzval)
end

function MtlSparseMatrixCSC{Tv, Ti}(A::SparseMatrixCSC) where {Tv, Ti <: Integer}
    m, n = size(A)
    (m <= typemax(Ti) && n <= typemax(Ti) && nnz(A) + 1 <= typemax(Ti)) ||
        throw(
        ArgumentError(
            "matrix with dimensions ($m, $n) and $(nnz(A)) stored entries does not fit in Ti = $Ti"
        )
    )
    colptr = MtlVector{Ti}(convert(Vector{Ti}, A.colptr))
    rowval = MtlVector{Ti}(convert(Vector{Ti}, A.rowval))
    nzval = MtlVector{Tv}(convert(Vector{Tv}, A.nzval))
    return MtlSparseMatrixCSC{Tv, Ti}(m, n, colptr, rowval, nzval)
end

function MtlSparseMatrixCSC(A::SparseMatrixCSC{Tv}) where {Tv}
    return MtlSparseMatrixCSC{Tv, DEFAULT_INDEX_TYPE}(A)
end

"""
    SparseMatrixCSC(A::MtlSparseMatrixCSC{Tv, Ti}) -> SparseMatrixCSC{Tv, Ti}

The host CSC matrix equal to `A`: the three storage arrays copied to the host
unchanged. The conversion is exact, so converting a `SparseMatrixCSC` to the
device and back is the identity up to the index type.
"""
function SparseArrays.SparseMatrixCSC(A::MtlSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return SparseMatrixCSC(A.m, A.n, Array(A.colptr), Array(A.rowval), Array(A.nzval))
end

"""
    Adapt.adapt_storage(::Type{MtlArray}, A::SparseMatrixCSC)

`adapt(MtlArray, A)` transfers a host `SparseMatrixCSC` to the device as an
[`MtlSparseMatrixCSC`](@ref) with index type [`DEFAULT_INDEX_TYPE`](@ref),
preserving the storage format, exactly as `CUDA.CUSPARSE` adapts to `CuArray`.

This method is deliberate type piracy — both `MtlArray` and `SparseMatrixCSC`
are owned by other packages — committed knowingly on the `CUSPARSE` precedent
and excepted in the Aqua piracy check; it is the one pirated method this package
defines.
"""
Adapt.adapt_storage(::Type{MtlArray}, A::SparseMatrixCSC) = MtlSparseMatrixCSC(A)
