# Validate host storage before uploading it; only the stored prefix is converted.
function host_compressed(A::SparseMatrixCSC, ::Type{Tv}, ::Type{Ti}, transposed) where {Tv, Ti}
    dims_check(A.m, A.n, Ti)
    stored = compressed_check(A.n, A.m, A.colptr, A.rowval, A.nzval, "colptr", "rowval")
    stored + 1 <= typemax(Ti) || throw(ArgumentError("stored count does not fit in Ti = $Ti"))
    B = transposed ? sparse(transpose(A)) : A
    return (
        MtlVector{Ti}(Vector{Ti}(B.colptr)),
        MtlVector{Ti}(Vector{Ti}(view(B.rowval, 1:stored))),
        MtlVector{Tv}(Vector{Tv}(view(B.nzval, 1:stored))),
    )
end

# Conversions between the device formats. COO and CSR share row-major entry
# order, so those two conversions run entirely on the device (the kernels live
# in src/kernels/conversions.jl); every conversion involving CSC reorders
# entries and currently goes through the host SparseMatrixCSC fallback (see
# each docstring). Conversions always copy: mutating the source never changes
# the result, matching CUSPARSE.

"""
    MtlSparseMatrixCSR(A::MtlSparseMatrixCOO)

Convert to CSR. Both formats store entries in row-major order, so the column
index and value arrays are copied on the device and the row pointer is built
on the device by one binary search per row over the sorted row indices
(`src/kernels/conversions.jl`); nothing touches the host. The conversion is
asynchronous and deterministic, and the `(i, j, v)` triple set is preserved
exactly.
"""
function MtlSparseMatrixCSR(A::MtlSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    return MtlSparseMatrixCSR{Tv, Ti}(
        unchecked, A.m, A.n, nnz(A), coo_rowptr(A), device_copy(A.colval), device_copy(A.nzval)
    )
end

"""
    MtlSparseMatrixCSR(A::MtlSparseMatrixCSC)

Convert to CSR. This reorders every entry from column-major to row-major, and
is currently computed through the host `SparseMatrixCSC` fallback (a full
transfer each way); a device reorder kernel is planned with the Phase 3 kernel
infrastructure. The `(i, j, v)` triple set is preserved exactly.
"""
function MtlSparseMatrixCSR(A::MtlSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return MtlSparseMatrixCSR{Tv, Ti}(SparseMatrixCSC(A))
end

"""
    SparseMatrixCSC(A::MtlSparseMatrixCSR{Tv, Ti}) -> SparseMatrixCSC{Tv, Ti}

The host CSC matrix equal to `A`. The conversion is exact: the sparsity pattern,
the ordering of the index arrays, and every stored value (zeros included) are
reproduced, so converting a `SparseMatrixCSC` to the device and back is the
identity up to the index type.
"""
function SparseArrays.SparseMatrixCSC(A::MtlSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    k = nnz(A)
    At = SparseMatrixCSC(
        A.n, A.m, Array(A.rowptr),
        Array(view(A.colval, 1:k)), Array(view(A.nzval, 1:k))
    )
    return sparse(transpose(At))
end

"""
    MtlSparseMatrixCSC(A::MtlSparseMatrixCSR)
    MtlSparseMatrixCSC(A::MtlSparseMatrixCOO)

Convert to CSC. Both reorder every entry from row-major to column-major, and
are currently computed through the host `SparseMatrixCSC` fallback (a full
transfer each way); a device reorder kernel is planned with the Phase 3 kernel
infrastructure. The `(i, j, v)` triple set is preserved exactly.
"""
function MtlSparseMatrixCSC(A::MtlSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    return MtlSparseMatrixCSC{Tv, Ti}(SparseMatrixCSC(A))
end

function MtlSparseMatrixCSC(A::MtlSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    return MtlSparseMatrixCSC{Tv, Ti}(SparseMatrixCSC(A))
end

"""
    SparseMatrixCSC(A::MtlSparseMatrixCSC{Tv, Ti}) -> SparseMatrixCSC{Tv, Ti}

The host CSC matrix equal to `A`: the three storage arrays copied to the host
unchanged. The conversion is exact, so converting a `SparseMatrixCSC` to the
device and back is the identity up to the index type.
"""
function SparseArrays.SparseMatrixCSC(A::MtlSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    k = nnz(A)
    return SparseMatrixCSC(
        A.m, A.n, Array(A.colptr),
        Array(view(A.rowval, 1:k)), Array(view(A.nzval, 1:k))
    )
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

"""
    MtlSparseMatrixCOO(A::MtlSparseMatrixCSR)

Convert to COO. Both formats store entries in row-major order, so the column
index and value arrays are copied on the device and the row indices are
expanded on the device, one thread per row writing its index over its pointer
range (`src/kernels/conversions.jl`); nothing touches the host. The
conversion is asynchronous and deterministic, and the `(i, j, v)` triple set
is preserved exactly.
"""
function MtlSparseMatrixCOO(A::MtlSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    k = nnz(A)
    rowval = expand_ptr(A.rowptr, k)
    return MtlSparseMatrixCOO{Tv, Ti}(
        unchecked, A.m, A.n, rowval,
        device_copy(view(A.colval, 1:k)), device_copy(view(A.nzval, 1:k))
    )
end

"""
    MtlSparseMatrixCOO(A::MtlSparseMatrixCSC)

Convert to COO. This reorders every entry from column-major to the row-major
order COO stores, and is currently computed through the host `SparseMatrixCSC`
fallback (a full transfer each way); a device reorder kernel is planned with
the Phase 3 kernel infrastructure. The `(i, j, v)` triple set is preserved
exactly.
"""
function MtlSparseMatrixCOO(A::MtlSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return MtlSparseMatrixCOO{Tv, Ti}(SparseMatrixCSC(A))
end

"""
    SparseMatrixCSC(A::MtlSparseMatrixCOO{Tv, Ti}) -> SparseMatrixCSC{Tv, Ti}

The host CSC matrix equal to `A`, assembled with `sparse(I, J, V, m, n)` from
host copies of the coordinate arrays. Under the format's no-duplicates
invariant the conversion is exact, stored zeros included.
"""
function SparseArrays.SparseMatrixCSC(A::MtlSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    return sparse(Array(A.rowval), Array(A.colval), Array(A.nzval), A.m, A.n)
end

# `as_csr` gives the merge a row-major representation; same-format is identity.
# `as_csc` sits with its caller in `interface.jl`.
function coo_rowptr(A::MtlSparseMatrixCOO{<:Any, Ti}) where {Ti}
    nnz(A) + 1 <= typemax(Ti) || throw(ArgumentError("stored count does not fit in Ti = $Ti"))
    rowptr = MtlVector{Ti}(undef, A.m + 1)
    kernel = Metal.@metal launch = false contract_idx_kernel!(rowptr, A.rowval, A.m, nnz(A))
    launch_per_slice(kernel, A.m + 1, rowptr, A.rowval, A.m, nnz(A))
    return rowptr
end

# Internal read-only CSR adapter borrows COO storage; public conversions copy.
function as_csr(A::MtlSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    return MtlSparseMatrixCSR{Tv, Ti}(
        unchecked, A.m, A.n, nnz(A), coo_rowptr(A), A.colval, A.nzval
    )
end

as_csr(A::MtlSparseMatrixCSR) = A
as_csr(A::AbstractMtlSparseMatrix) = MtlSparseMatrixCSR(A)

"""
    Array(A::AbstractMtlSparseMatrix{Tv}) -> Matrix{Tv}

The dense host matrix equal to `A`, agreeing with `Array(::SparseMatrixCSC)`:
every stored value is written (stored zeros included, indistinguishably), every
other entry is zero.
"""
Base.Array(A::AbstractMtlSparseMatrix) = Array(SparseMatrixCSC(A))

"""
    MtlMatrix(A::AbstractMtlSparseMatrix{Tv}) -> MtlMatrix{Tv}

The dense device matrix equal to `A`, computed asynchronously and deterministically
by scattering its stored entries into a zeroed matrix. No format conversion or
host transfer is performed. Address arithmetic uses `Int` independently of the
sparse index type. Agrees with `Array(::SparseMatrixCSC)` moved to the device.
"""
function Metal.MtlMatrix(A::AbstractMtlSparseMatrix{Tv}) where {Tv}
    D = Metal.zeros(Tv, A.m, A.n)
    nnz(A) > 0 && scatter!(D, A)
    return D
end

"""
    MtlSparseMatrixCSC(D::Union{Matrix, MtlMatrix})
    MtlSparseMatrixCSR(D::Union{Matrix, MtlMatrix})
    MtlSparseMatrixCOO(D::Union{Matrix, MtlMatrix})

Sparsify a dense host or device matrix into the named device format, with the
semantics of `sparse(::Matrix)`: numerical zeros are dropped, everything else
(nonfinite values included) becomes a stored entry. Currently computed through
host `sparse` (a device matrix is first copied to the host); the index type is
[`DEFAULT_INDEX_TYPE`](@ref).
"""
MtlSparseMatrixCSC(D::Matrix) = MtlSparseMatrixCSC(sparse(D))
MtlSparseMatrixCSC(D::MtlMatrix) = MtlSparseMatrixCSC(sparse(Array(D)))
MtlSparseMatrixCSR(D::Matrix) = MtlSparseMatrixCSR(sparse(D))
MtlSparseMatrixCSR(D::MtlMatrix) = MtlSparseMatrixCSR(sparse(Array(D)))
MtlSparseMatrixCOO(D::Matrix) = MtlSparseMatrixCOO(sparse(D))
MtlSparseMatrixCOO(D::MtlMatrix) = MtlSparseMatrixCOO(sparse(Array(D)))
