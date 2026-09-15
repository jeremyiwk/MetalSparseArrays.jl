## COV_EXCL_START

function order_keys_kernel!(keys, idx, perm, shift, mask, rankbits, stored)
    k = Int(thread_position_in_grid().x)
    k <= stored || return nothing
    @inbounds begin
        p = perm === nothing ? k : Int(perm[k])
        digit = ((Int64(idx[p]) - 1) >>> shift) & mask
        keys[k] = (digit << rankbits) | (k - 1)
    end
    return nothing
end

function compose_order_kernel!(next, prev, stored)
    k = Int(thread_position_in_grid().x)
    k <= stored || return nothing
    @inbounds next[k] = prev === nothing ? k : prev[Int(next[k])]
    return nothing
end

function gather_entries_kernel!(major, minor, values, oldmajor, oldminor, oldvalues, perm, stored)
    k = Int(thread_position_in_grid().x)
    k <= stored || return nothing
    @inbounds begin
        p = perm === nothing ? k : Int(perm[k])
        major[k] = oldmajor[p]
        minor[k] = oldminor[p]
        values[k] = oldvalues[p]
    end
    return nothing
end

## COV_EXCL_STOP

"""
    stable_index_sortperm(idx, bound, perm=nothing)

Stably sort positive device indices bounded by `bound`, optionally starting in
the order `perm`. Return an Int32 permutation of the original entries. Each
MPSGraph sort uses unique Int64 keys: index digits followed by the current rank.
Thus ties preserve input order without relying on backend sort stability. Large
Int64 indices use multiple least-significant-digit passes, avoiding packed-key
overflow. At most `typemax(Int32)` entries are supported by MPSGraph argSort.
The operation is asynchronous; no indices are read on the host.
"""
function stable_index_sortperm(idx, bound::Integer, perm = nothing)
    stored = length(idx)
    stored <= typemax(Int32) ||
        throw(ArgumentError("device sorting supports at most $(typemax(Int32)) entries"))
    stored == 0 && return MtlVector{Int32}(undef, 0)
    stored == 1 && return perm === nothing ? Metal.ones(Int32, 1) : perm
    if bound == 1
        perm !== nothing && return perm
        perm = MtlVector{Int32}(undef, stored)
        kernel = Metal.@metal launch = false compose_order_kernel!(perm, nothing, stored)
        launch_per_slice(kernel, stored, perm, nothing, stored)
        return perm
    end
    rankbits = 64 - leading_zeros(UInt64(stored - 1))
    digitbits = 63 - rankbits
    mask = (Int64(1) << digitbits) - 1
    bits = max(1, 64 - leading_zeros(UInt64(bound - 1)))
    keys = MtlVector{Int64}(undef, stored)
    for shift in 0:digitbits:(bits - 1)
        kernel = Metal.@metal launch = false order_keys_kernel!(
            keys, idx, perm, shift, mask, rankbits, stored
        )
        launch_per_slice(kernel, stored, keys, idx, perm, shift, mask, rankbits, stored)
        next = MtlVector{Int32}(undef, stored)
        sortperm!(next, keys)
        if perm !== nothing
            compose = Metal.@metal launch = false compose_order_kernel!(next, perm, stored)
            launch_per_slice(compose, stored, next, perm, stored)
        end
        perm = next
    end
    return perm
end

function reorder_entries(major, minor, values, bound)
    stored = length(major)
    perm = bound <= 1 || stored <= 1 ? nothing : stable_index_sortperm(major, bound)
    newmajor = MtlVector{eltype(major)}(undef, stored)
    newminor = MtlVector{eltype(minor)}(undef, stored)
    newvalues = MtlVector{eltype(values)}(undef, stored)
    if stored > 0
        kernel = Metal.@metal launch = false gather_entries_kernel!(
            newmajor, newminor, newvalues, major, minor, values, perm, stored
        )
        launch_per_slice(
            kernel, stored, newmajor, newminor, newvalues, major, minor, values, perm, stored
        )
    end
    return newmajor, newminor, newvalues
end
