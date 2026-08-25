# Device prefix sum for pointer arrays. `ptr_scan!` turns per-slice entry
# counts into a compressed pointer array in three asynchronous kernel
# launches, replacing the generic multi-kernel `accumulate!` whose fixed cost
# dominated the merge (measured 121 vs 162 us at 4096 counts and 179 vs 457 us
# at 262144, synchronized timings on Apple M-series). The classic two-level
# scan: every threadgroup scans one block of counts, one threadgroup scans the
# block totals into per-block carries, and a final pass adds each block's
# carry. All arithmetic is integer and every reduction tree is fixed, so the
# result is deterministic.

const SCAN_BLOCK = 1024

## COV_EXCL_START

# Inclusive scan of `x` across the threadgroup: a shuffle-based scan within
# each simdgroup, the simdgroup totals scanned by the first simdgroup through
# threadgroup memory, then each lane offset by its simdgroup's prefix. `buf`
# needs one slot per simdgroup (at most `SCAN_BLOCK ÷ 32`).
@inline function threadgroup_scan(x, buf)
    width = Int(threads_per_simdgroup())
    lane = Int(thread_index_in_simdgroup())
    sg = Int(simdgroup_index_in_threadgroup())
    nsg = Int(simdgroups_per_threadgroup())
    d = 1
    while d < width
        y = simd_shuffle_up(x, d)
        if lane > d
            x += y
        end
        d <<= 1
    end
    if lane == width
        @inbounds buf[sg] = x
    end
    threadgroup_barrier(MemoryFlagThreadGroup)
    if sg == 1
        s = lane <= nsg ? (@inbounds buf[lane]) : zero(x)
        d = 1
        while d < width
            y = simd_shuffle_up(s, d)
            if lane > d
                s += y
            end
            d <<= 1
        end
        if lane <= nsg
            @inbounds buf[lane] = s
        end
    end
    threadgroup_barrier(MemoryFlagThreadGroup)
    if sg > 1
        x += @inbounds buf[sg - 1]
    end
    return x
end

# Each threadgroup g inclusively scans its block of counts into ptr[i + 1]
# (block-local) and writes the block total to sums[g].
function scan_blocks_kernel!(ptr, sums, counts, n)
    t = Int(thread_position_in_threadgroup().x)
    T = Int(threads_per_threadgroup().x)
    g = Int(threadgroup_position_in_grid().x)
    base = (g - 1) * T
    i = base + t
    buf = MtlThreadGroupArray(eltype(ptr), 32)
    x = i <= n ? (@inbounds counts[i]) : zero(eltype(ptr))
    x = threadgroup_scan(x, buf)
    if i <= n
        @inbounds ptr[i + 1] = x
    end
    if t == min(T, n - base)
        @inbounds sums[g] = x
    end
    return nothing
end

# One threadgroup walks the block totals in chunks, carrying the running
# prefix in a register, and writes carries[g] = init + sum(sums[1:g - 1]).
# The last threadgroup-memory slot broadcasts each chunk's total.
function scan_carries_kernel!(carries, sums, nblocks, init)
    t = Int(thread_position_in_threadgroup().x)
    T = Int(threads_per_threadgroup().x)
    buf = MtlThreadGroupArray(eltype(carries), 33)
    carry = init
    base = 0
    while base < nblocks
        i = base + t
        x = i <= nblocks ? (@inbounds sums[i]) : zero(eltype(carries))
        inc = threadgroup_scan(x, buf)
        if i <= nblocks
            @inbounds carries[i] = carry + inc - x
        end
        if t == min(T, nblocks - base)
            @inbounds buf[33] = inc
        end
        threadgroup_barrier(MemoryFlagThreadGroup)
        carry += @inbounds buf[33]
        threadgroup_barrier(MemoryFlagThreadGroup)
        base += T
    end
    return nothing
end

# Add each block's carry to its block-local scan; the first thread writes the
# leading pointer entry, ptr[1] = carries[1] = init.
function scan_add_carries_kernel!(ptr, carries, n)
    i = Int(thread_position_in_grid().x)
    i <= n || return nothing
    T = Int(threads_per_threadgroup().x)
    g = (i - 1) ÷ T + 1
    @inbounds ptr[i + 1] += carries[g]
    if i == 1
        @inbounds ptr[1] = carries[1]
    end
    return nothing
end

## COV_EXCL_STOP

"""
    ptr_scan!(ptr, counts, init)

Fill the pointer array `ptr` (length `n + 1`) with `ptr[1] = init` and
`ptr[i + 1] = init + sum(counts[1:i])` for the device vector `counts` of
length `n >= 1`, entirely on the device in three asynchronous kernel launches;
nothing touches the host and the caller synchronizes. The element types of
`ptr` and `counts` must match, the sum must fit that type (unchecked here;
callers bound it before launching), and the result is deterministic.
"""
function ptr_scan!(
        ptr::MtlVector{Ti}, counts::MtlVector{Ti}, init::Ti
    ) where {Ti <: Integer}
    n = length(counts)
    blocks = Metal.@metal launch = false scan_blocks_kernel!(ptr, counts, counts, n)
    add = Metal.@metal launch = false scan_add_carries_kernel!(ptr, counts, n)
    threads = min(
        SCAN_BLOCK,
        blocks.pipeline.maxTotalThreadsPerThreadgroup,
        add.pipeline.maxTotalThreadsPerThreadgroup
    )
    nblocks = cld(n, threads)
    sums = MtlVector{Ti}(undef, nblocks)
    carries = MtlVector{Ti}(undef, nblocks)
    blocks(ptr, sums, counts, n; threads, groups = nblocks)
    carrykernel = Metal.@metal launch = false scan_carries_kernel!(carries, sums, nblocks, init)
    carrythreads = min(SCAN_BLOCK, carrykernel.pipeline.maxTotalThreadsPerThreadgroup)
    carrykernel(carries, sums, nblocks, init; threads = carrythreads, groups = 1)
    add(ptr, carries, n; threads, groups = nblocks)
    return ptr
end
