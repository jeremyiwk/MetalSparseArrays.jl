# The array interface shared by the device formats: similar, copy, collect,
# rowvals, findnz, and scalar indexing. Scalar getindex/setindex! read device
# memory elementwise, so they work under `Metal.@allowscalar` and error under
# `Metal.allowscalar(false)` — the CUSPARSE-precedent policy for device sparse
# matrices; nothing in this package calls them internally.

"""
    similar(A::AbstractMtlSparseMatrix[, Tv][, dims])

Following `SparseArrays`: `similar(A)` and `similar(A, Tv)` keep the sparsity
pattern of `A` (index arrays copied) with an uninitialized value array of the
requested element type; `similar(A, [Tv,] dims)` gives an empty (no stored
entries) matrix of the same format with the requested dimensions.
"""
function Base.similar(A::MtlSparseMatrixCSC{<:Any, Ti}, ::Type{Tv}) where {Tv, Ti}
    nzval = MtlVector{Tv}(undef, nnz(A))
    return MtlSparseMatrixCSC{Tv, Ti}(A.m, A.n, device_copy(A.colptr), device_copy(A.rowval), nzval)
end

function Base.similar(A::MtlSparseMatrixCSR{<:Any, Ti}, ::Type{Tv}) where {Tv, Ti}
    nzval = MtlVector{Tv}(undef, nnz(A))
    return MtlSparseMatrixCSR{Tv, Ti}(A.m, A.n, device_copy(A.rowptr), device_copy(A.colval), nzval)
end

function Base.similar(A::MtlSparseMatrixCOO{<:Any, Ti}, ::Type{Tv}) where {Tv, Ti}
    nzval = MtlVector{Tv}(undef, nnz(A))
    return MtlSparseMatrixCOO{Tv, Ti}(A.m, A.n, device_copy(A.rowval), device_copy(A.colval), nzval)
end

Base.similar(A::AbstractMtlSparseMatrix{Tv}) where {Tv} = similar(A, Tv)

function Base.similar(
        A::AbstractMtlSparseMatrix{<:Any, Ti}, ::Type{Tv}, dims::Dims{2}
    ) where {Tv, Ti}
    return empty_format(typeof(A), Tv, Ti, dims...)
end

function Base.similar(A::AbstractMtlSparseMatrix{Tv}, dims::Dims{2}) where {Tv}
    return similar(A, Tv, dims)
end

function empty_format(
        ::Type{<:MtlSparseMatrixCSC}, ::Type{Tv}, ::Type{Ti}, m::Integer, n::Integer
    ) where {Tv, Ti}
    colptr = MtlVector{Ti}(ones(Ti, n + 1))
    return MtlSparseMatrixCSC{Tv, Ti}(
        m, n, colptr, MtlVector{Ti}(undef, 0), MtlVector{Tv}(undef, 0)
    )
end

function empty_format(
        ::Type{<:MtlSparseMatrixCSR}, ::Type{Tv}, ::Type{Ti}, m::Integer, n::Integer
    ) where {Tv, Ti}
    rowptr = MtlVector{Ti}(ones(Ti, m + 1))
    return MtlSparseMatrixCSR{Tv, Ti}(
        m, n, rowptr, MtlVector{Ti}(undef, 0), MtlVector{Tv}(undef, 0)
    )
end

function empty_format(
        ::Type{<:MtlSparseMatrixCOO}, ::Type{Tv}, ::Type{Ti}, m::Integer, n::Integer
    ) where {Tv, Ti}
    return MtlSparseMatrixCOO{Tv, Ti}(
        m, n, MtlVector{Ti}(undef, 0), MtlVector{Ti}(undef, 0), MtlVector{Tv}(undef, 0)
    )
end

function Base.copy(A::MtlSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return MtlSparseMatrixCSC{Tv, Ti}(
        A.m, A.n, device_copy(A.colptr), device_copy(A.rowval), device_copy(A.nzval)
    )
end

function Base.copy(A::MtlSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    return MtlSparseMatrixCSR{Tv, Ti}(
        A.m, A.n, device_copy(A.rowptr), device_copy(A.colval), device_copy(A.nzval)
    )
end

function Base.copy(A::MtlSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    return MtlSparseMatrixCOO{Tv, Ti}(
        A.m, A.n, device_copy(A.rowval), device_copy(A.colval), device_copy(A.nzval)
    )
end

Base.collect(A::AbstractMtlSparseMatrix) = Array(A)

"""
    rowvals(A::MtlSparseMatrixCSC)

The `MtlVector` of row indices of the stored entries, aliasing the matrix like
`SparseArrays.rowvals` does for `SparseMatrixCSC`.
"""
SparseArrays.rowvals(A::MtlSparseMatrixCSC) = A.rowval

"""
    findnz(A::AbstractMtlSparseMatrix{Tv, Ti}) -> (I, J, V)

The stored entries of `A` as three `MtlVector`s of row indices, column indices,
and values, in the column-major order `SparseArrays.findnz` returns, freshly
allocated. Computed via the CSC form of `A`; only index arrays pass through the
host.
"""
function SparseArrays.findnz(A::AbstractMtlSparseMatrix{Tv, Ti}) where {Tv, Ti}
    csc = as_csc(A)
    ptr = Array(csc.colptr)
    colhost = Vector{Ti}(undef, nnz(csc))
    for j in 1:csc.n, k in ptr[j]:(ptr[j + 1] - 1)
        colhost[k] = Ti(j)
    end
    return (device_copy(csc.rowval), MtlVector{Ti}(colhost), device_copy(csc.nzval))
end

as_csc(A::MtlSparseMatrixCSC) = A
as_csc(A::AbstractMtlSparseMatrix) = MtlSparseMatrixCSC(A)

# Scalar indexing. The searches below read device memory elementwise and are
# guarded by GPUArrays' allowscalar machinery through the MtlVector accesses
# themselves; the O(nnz) COO scan and O(entries per slice) compressed searches
# are the CUSPARSE-precedent convenience, not a fast path.

function Base.getindex(A::MtlSparseMatrixCSC{Tv}, i::Integer, j::Integer) where {Tv}
    @boundscheck checkbounds(A, i, j)
    for k in Int(A.colptr[j]):(Int(A.colptr[j + 1]) - 1)
        A.rowval[k] == i && return A.nzval[k]
    end
    return zero(Tv)
end

function Base.getindex(A::MtlSparseMatrixCSR{Tv}, i::Integer, j::Integer) where {Tv}
    @boundscheck checkbounds(A, i, j)
    for k in Int(A.rowptr[i]):(Int(A.rowptr[i + 1]) - 1)
        A.colval[k] == j && return A.nzval[k]
    end
    return zero(Tv)
end

function Base.getindex(A::MtlSparseMatrixCOO{Tv}, i::Integer, j::Integer) where {Tv}
    @boundscheck checkbounds(A, i, j)
    for k in 1:nnz(A)
        (A.rowval[k] == i && A.colval[k] == j) && return A.nzval[k]
    end
    return zero(Tv)
end

"""
    setindex!(A::AbstractMtlSparseMatrix, v, i, j)

Update the stored entry at `(i, j)` to `v`. Writing to a position with no
stored entry throws `ArgumentError`: changing the sparsity structure of a
device matrix is not supported (matching `CUSPARSE`) — assemble a new matrix
instead. Scalar indexing, so it requires `Metal.@allowscalar`.
"""
function Base.setindex!(A::MtlSparseMatrixCSC, v, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    for k in Int(A.colptr[j]):(Int(A.colptr[j + 1]) - 1)
        if A.rowval[k] == i
            A.nzval[k] = v
            return A
        end
    end
    throw(structural_setindex_error(i, j))
end

function Base.setindex!(A::MtlSparseMatrixCSR, v, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    for k in Int(A.rowptr[i]):(Int(A.rowptr[i + 1]) - 1)
        if A.colval[k] == j
            A.nzval[k] = v
            return A
        end
    end
    throw(structural_setindex_error(i, j))
end

function Base.setindex!(A::MtlSparseMatrixCOO, v, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    for k in 1:nnz(A)
        if A.rowval[k] == i && A.colval[k] == j
            A.nzval[k] = v
            return A
        end
    end
    throw(structural_setindex_error(i, j))
end

function structural_setindex_error(i, j)
    return ArgumentError(
        "no stored entry at ($i, $j): changing the sparsity structure of a " *
            "device matrix is not supported; assemble a new matrix instead"
    )
end
