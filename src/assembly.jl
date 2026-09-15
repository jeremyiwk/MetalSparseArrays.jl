"""
    sparse(I::MtlVector, J::MtlVector, V::MtlVector[, m, n, combine]; fmt=:csc)
    sparse(I::MtlVector, J::MtlVector, V::MtlVector[, m, n]; fmt=:csc, combine=+)

Assemble device coordinates into `MtlSparseMatrixCSC`, or CSR/COO with `fmt=:csr`
or `:coo`. Unsorted coordinates and duplicates are accepted. Dimensions default
to the maximum indices, or zero for empty inputs. Index types are promoted from
`I` and `J`; values retain `eltype(V)`. Lengths must agree, indices must lie in
the specified dimensions, and the input count must fit the index type's pointer
range and MPSGraph's `typemax(Int32)` sorting limit.

Duplicate values are folded in original input order, converting to the value
type after each application of the GPU-compatible, isbits `combine` function.
The default is `+`, or `|` for Bool values. Numerical zeros, including cancelled
sums, remain stored. Inputs are not mutated or aliased. Ordering and accumulation
are deterministic; accumulation uses the value type, matching SparseArrays.

Coordinates and values stay on device. Validation and the resulting stored count
are read back; inferred dimensions require maximum reductions. Final output
writes are asynchronous. This deliberately extends `SparseArrays.sparse` on
Metal-owned vectors, following the CUSPARSE device assembly interface.
"""
function SparseArrays.sparse(
        I::MtlVector{Ti}, J::MtlVector{Tj}, V::MtlVector{Tv},
        m::Integer, n::Integer, combine; fmt = :csc
    ) where {Ti <: Integer, Tj <: Integer, Tv <: Number}
    F = fmt === :csc ? MtlSparseMatrixCSC : fmt === :csr ? MtlSparseMatrixCSR :
        fmt === :coo ? MtlSparseMatrixCOO :
        throw(ArgumentError("unknown sparse format $fmt; use :csc, :csr or :coo"))
    T = promote_type(Ti, Tj)
    dims_check(m, n, T)
    stored = length(I)
    stored == length(J) == length(V) ||
        throw(ArgumentError("lengths of I, J and V must agree"))
    stored < typemax(T) && stored <= typemax(Int32) ||
        throw(ArgumentError("input count exceeds the index type or device sorting limit"))
    isbits(combine) || throw(ArgumentError("combine must be an isbits GPU-compatible function"))
    stored == 0 && return empty_format(F, Tv, T, m, n)
    inrange = mapreduce(
        (i, j) -> (1 <= i <= m) & (1 <= j <= n), &, I, J
    )
    inrange || throw(ArgumentError("coordinates must satisfy 1 <= I[k] <= m and 1 <= J[k] <= n"))
    major, minor, nmajor, nminor = fmt === :csc ? (J, I, n, m) : (I, J, m, n)
    perm = coordinate_sortperm(major, minor, nmajor, nminor)
    flags = MtlVector{T}(undef, stored)
    positions = MtlVector{T}(undef, stored + 1)
    groups = Metal.@metal launch = false assembly_groups_kernel!(flags, major, minor, perm, stored)
    launch_per_slice(groups, stored, flags, major, minor, perm, stored)
    ptr_scan!(positions, flags, zero(T))
    count = Int(only(Array(view(positions, (stored + 1):(stored + 1)))))
    newmajor = MtlVector{T}(undef, count)
    newminor = MtlVector{T}(undef, count)
    values = MtlVector{Tv}(undef, count)
    assembly_fold!(
        newmajor, newminor, values, major, minor, V, perm, flags, positions, combine
    )
    if fmt === :coo
        return MtlSparseMatrixCOO{Tv, T}(unchecked, m, n, newmajor, newminor, values)
    end
    return F{Tv, T}(unchecked, m, n, count, contract_idx(newmajor, nmajor), newminor, values)
end

function SparseArrays.sparse(
        I::MtlVector{Ti}, J::MtlVector{Tj}, V::MtlVector{Tv},
        m::Integer = isempty(I) ? 0 : Int(maximum(I)),
        n::Integer = isempty(J) ? 0 : Int(maximum(J));
        fmt = :csc, combine = Tv === Bool ? (|) : (+)
    ) where {Ti <: Integer, Tj <: Integer, Tv <: Number}
    return sparse(I, J, V, m, n, combine; fmt)
end
