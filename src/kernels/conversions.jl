# Device kernels for the conversions between the row-major formats. CSR and
# COO order their entries identically, so converting between them only
# rewrites the row bookkeeping: CSR to COO expands the pointer array into
# per-entry row indices, and COO to CSR contracts the sorted row indices back
# into a pointer array. Both run entirely on the device with no atomics and
# disjoint writes, so they are deterministic; the callers in
# src/conversions.jl state the semantics.

## COV_EXCL_START

# Expand a compressed pointer array into per-entry major indices: one thread
# per slice writes its own index over its pointer range.
function expand_ptr_kernel!(idx, ptr, major)
    i = Int(thread_position_in_grid().x)
    i <= major || return nothing
    @inbounds for k in Int(ptr[i]):(Int(ptr[i + 1]) - 1)
        idx[k] = eltype(idx)(i)
    end
    return nothing
end

# Contract sorted per-entry major indices into a pointer array: thread `i`
# binary searches for the first entry with index at least `i` (`stored + 1`
# for `i = major + 1`, and for every empty trailing slice). The entries within
# a slice need not be sorted by minor index for this to be correct; only the
# major indices must be nondecreasing.
function contract_idx_kernel!(ptr, idx, major, stored)
    i = Int(thread_position_in_grid().x)
    i <= major + 1 || return nothing
    lo, hi = 1, stored + 1
    @inbounds while lo < hi
        mid = (lo + hi) >> 1
        if Int(idx[mid]) < i
            lo = mid + 1
        else
            hi = mid
        end
    end
    @inbounds ptr[i] = eltype(ptr)(lo)
    return nothing
end

## COV_EXCL_STOP

function expand_ptr(ptr::MtlVector{Ti}, stored::Integer) where {Ti}
    idx = MtlVector{Ti}(undef, stored)
    major = length(ptr) - 1
    if major > 0 && stored > 0
        kernel = Metal.@metal launch = false expand_ptr_kernel!(idx, ptr, major)
        launch_per_slice(kernel, major, idx, ptr, major)
    end
    return idx
end

function contract_idx(idx::MtlVector{Ti}, major::Integer) where {Ti}
    stored = length(idx)
    stored < typemax(Ti) || throw(ArgumentError("stored count does not fit in Ti = $Ti"))
    ptr = MtlVector{Ti}(undef, major + 1)
    if stored == 0
        fill!(ptr, one(Ti))
    else
        kernel = Metal.@metal launch = false contract_idx_kernel!(ptr, idx, major, stored)
        launch_per_slice(kernel, major + 1, ptr, idx, major, stored)
    end
    return ptr
end

## COV_EXCL_START

function scatter_compressed_kernel!(D, ptr, idx, val, major, column_major)
    i = Int(thread_position_in_grid().x)
    i <= major || return nothing
    @inbounds for k in Int(ptr[i]):(Int(ptr[i + 1]) - 1)
        if column_major
            D[Int(idx[k]), i] = val[k]
        else
            D[i, Int(idx[k])] = val[k]
        end
    end
    return nothing
end

function scatter_coo_kernel!(D, rows, cols, vals, stored)
    k = Int(thread_position_in_grid().x)
    k <= stored || return nothing
    @inbounds D[Int(rows[k]), Int(cols[k])] = vals[k]
    return nothing
end

## COV_EXCL_STOP

function scatter!(D, A::Union{MtlSparseMatrixCSR, MtlSparseMatrixCSC})
    column_major = A isa MtlSparseMatrixCSC
    ptr, idx, major = column_major ? (A.colptr, A.rowval, A.n) : (A.rowptr, A.colval, A.m)
    kernel = Metal.@metal launch = false scatter_compressed_kernel!(
        D, ptr, idx, A.nzval, major, column_major
    )
    launch_per_slice(kernel, major, D, ptr, idx, A.nzval, major, column_major)
    return D
end

function scatter!(D, A::MtlSparseMatrixCOO)
    kernel = Metal.@metal launch = false scatter_coo_kernel!(
        D, A.rowval, A.colval, A.nzval, nnz(A)
    )
    launch_per_slice(kernel, nnz(A), D, A.rowval, A.colval, A.nzval, nnz(A))
    return D
end
