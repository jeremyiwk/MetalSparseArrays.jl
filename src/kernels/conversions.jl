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
