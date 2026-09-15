## COV_EXCL_START

function coordinate_keys_kernel!(keys, major, minor, minorbits, rankbits, stored)
    k = Int(thread_position_in_grid().x)
    k <= stored || return nothing
    @inbounds begin
        coordinate = ((Int64(major[k]) - 1) << minorbits) | (Int64(minor[k]) - 1)
        keys[k] = (coordinate << rankbits) | (k - 1)
    end
    return nothing
end

function assembly_groups_kernel!(flags, major, minor, perm, stored)
    k = Int(thread_position_in_grid().x)
    k <= stored || return nothing
    @inbounds begin
        p = Int(perm[k])
        prev = k == 1 ? p : Int(perm[k - 1])
        flags[k] = k == 1 || major[p] != major[prev] || minor[p] != minor[prev]
    end
    return nothing
end

function assembly_fold_kernel!(
        newmajor, newminor, values, major, minor, oldvalues, perm, flags, positions, combine, stored
    )
    k = Int(thread_position_in_grid().x)
    k <= stored || return nothing
    @inbounds if flags[k] != 0
        p = Int(perm[k])
        dest = Int(positions[k + 1])
        newmajor[dest] = major[p]
        newminor[dest] = minor[p]
        value = oldvalues[p]
        while k < stored && flags[k + 1] == 0
            k += 1
            value = eltype(values)(combine(value, oldvalues[Int(perm[k])]))
        end
        values[dest] = value
    end
    return nothing
end

function assembly_starts_kernel!(starts, flags, positions, stored)
    k = Int(thread_position_in_grid().x)
    k <= stored || return nothing
    @inbounds begin
        flags[k] != 0 && (starts[Int(positions[k + 1])] = k)
        k == stored && (starts[end] = stored + 1)
    end
    return nothing
end

# Coalesced loads avoid one global-memory dependency per duplicate. The leader
# still folds each group sequentially, preserving the CPU accumulation order.
function assembly_fold_cooperative_kernel!(
        newmajor, newminor, values, major, minor, oldvalues, perm, starts, combine
    )
    group = Int(threadgroup_position_in_grid().x)
    t = Int(thread_position_in_threadgroup().x)
    threads = Int(threads_per_threadgroup().x)
    buffer = MtlThreadGroupArray(eltype(values), 256)
    @inbounds first, last = Int(starts[group]), Int(starts[group + 1]) - 1
    value = zero(eltype(values))
    base = first
    while base <= last
        k = base + t - 1
        if k <= last
            @inbounds buffer[t] = oldvalues[Int(perm[k])]
        end
        threadgroup_barrier(MemoryFlagThreadGroup)
        if t == 1
            @inbounds for j in 1:min(threads, last - base + 1)
                value = base == first && j == 1 ? buffer[j] :
                    eltype(values)(combine(value, buffer[j]))
            end
        end
        threadgroup_barrier(MemoryFlagThreadGroup)
        base += threads
    end
    if t == 1
        @inbounds begin
            p = Int(perm[first])
            newmajor[group] = major[p]
            newminor[group] = minor[p]
            values[group] = value
        end
    end
    return nothing
end

## COV_EXCL_STOP

function assembly_fold!(newmajor, newminor, values, major, minor, oldvalues, perm, flags, positions, combine)
    stored, count = length(major), length(values)
    if stored >= 1024 && stored >= 32count
        starts = MtlVector{eltype(positions)}(undef, count + 1)
        mark = Metal.@metal launch = false assembly_starts_kernel!(starts, flags, positions, stored)
        launch_per_slice(mark, stored, starts, flags, positions, stored)
        fold = Metal.@metal launch = false assembly_fold_cooperative_kernel!(
            newmajor, newminor, values, major, minor, oldvalues, perm, starts, combine
        )
        threads = min(256, fold.pipeline.maxTotalThreadsPerThreadgroup)
        fold(newmajor, newminor, values, major, minor, oldvalues, perm, starts, combine; threads, groups = count)
    else
        fold = Metal.@metal launch = false assembly_fold_kernel!(
            newmajor, newminor, values, major, minor, oldvalues, perm, flags, positions, combine, stored
        )
        launch_per_slice(
            fold, stored, newmajor, newminor, values, major, minor, oldvalues, perm, flags, positions, combine, stored
        )
    end
    return nothing
end

function coordinate_sortperm(major, minor, nmajor, nminor)
    stored = length(major)
    stored == 1 && return Metal.ones(Int32, 1)
    nmajor == nminor == 1 && return stable_index_sortperm(major, 1)
    rankbits = 64 - leading_zeros(UInt64(stored - 1))
    minorbits = 64 - leading_zeros(UInt64(nminor - 1))
    majorbits = 64 - leading_zeros(UInt64(nmajor - 1))
    if rankbits + minorbits + majorbits <= 63
        keys = MtlVector{Int64}(undef, stored)
        perm = MtlVector{Int32}(undef, stored)
        kernel = Metal.@metal launch = false coordinate_keys_kernel!(
            keys, major, minor, minorbits, rankbits, stored
        )
        launch_per_slice(kernel, stored, keys, major, minor, minorbits, rankbits, stored)
        sortperm!(perm, keys)
        return perm
    end
    perm = stable_index_sortperm(minor, nminor)
    return stable_index_sortperm(major, nmajor, perm)
end
