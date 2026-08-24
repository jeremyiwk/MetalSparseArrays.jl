# The benchmark suite. `runbenchmarks.jl` activates the environment, includes
# this file, and times every case in `SUITE`.
#
# Convention: one group per operation, each case keyed by representation
# ("SparseArrays", "device sparse", "device dense"), format, element type, and
# problem size, so that both crossover sizes stated in docs/src/roadmap.md —
# device sparse over CPU sparse, and device sparse over device dense — can be
# read directly from the results.
#
# Timing is by the standard-library macros, not BenchmarkTools: a case marked
# `device` is timed by `Metal.@timed` (the macro behind `Metal.@time`, which
# synchronizes the GPU before the expression and wraps it in `Metal.@sync`, so
# the measurement covers the complete device work and nothing else) and a host
# case by `Base.@elapsed` (the timing core of `Base.@time`). Each case is run
# once untimed to compile, then the minimum over `REPS` repetitions is
# reported; single-shot timings measured ~250 us high against minimums on this
# hardware.

using LinearAlgebra: I, kron
using Metal: Metal, MtlArray
using MetalSparseArrays
using SparseArrays: SparseMatrixCSC, sparse, spdiagm

"""
    Benchmark(group, key, device, thunk)

One benchmark case: `thunk` is the zero-argument operation to time, `group`
names the operation, `key` identifies representation, format, element type,
and size, and `device` selects GPU-synchronized timing.
"""
struct Benchmark
    group::String
    key::Tuple
    device::Bool
    thunk::Function
end

const SUITE = Benchmark[]

benchmark!(thunk::Function, group::String, key...; device::Bool = false) =
    push!(SUITE, Benchmark(group, key, device, thunk))

"""
    laplacian_2d(Tv, n)

The `n^2`-by-`n^2` five point negative Laplacian on an `n`-by-`n` grid with
homogeneous Dirichlet boundary conditions. Mirrors the definition in
`test/testsuite.jl`; repeated here because the benchmark environment does not
depend on the test environment.
"""
function laplacian_2d(::Type{Tv}, n::Integer) where {Tv}
    tridiagonal = spdiagm(
        -1 => fill(-one(Tv), n - 1),
        0 => fill(Tv(2), n),
        1 => fill(-one(Tv), n - 1)
    )
    identity_n = sparse(one(Tv) * I, n, n)
    return kron(identity_n, tridiagonal) + kron(tridiagonal, identity_n)
end

# A second merge operand whose pattern shares the Laplacian's diagonal, misses
# its off-diagonal bands, and adds a band of its own, so the merge sees all
# three kinds of union position (shared, first only, second only) rather than
# a degenerate case. Deterministic, so results are comparable across runs.
function offset_band(::Type{Tv}, N::Integer) where {Tv}
    return sparse(one(Tv) * I, N, N) + spdiagm(N, N, 2 => fill(one(Tv), N - 2))
end

# Load imbalance: every stored entry of both operands lies in one row, so the
# CSR merge gives one thread all the work. This pattern exists to measure the
# worst case of the one-thread-per-slice launch.
function dense_row_pair(::Type{Tv}, N::Integer) where {Tv}
    columns = collect(1:N)
    return (
        sparse(fill(1, N), columns, fill(one(Tv), N), N, N),
        sparse(fill(1, N), columns, fill(Tv(2), N), N, N),
    )
end

const DEVICE_FORMATS = ("CSR" => MtlSparseMatrixCSR, "CSC" => MtlSparseMatrixCSC)

# Dense operands are full m-by-n arrays, so the dense comparison is only
# affordable up to this order; the sparse entries run past it, which is where
# the roadmap's second crossover goal (device sparse over device dense) lives.
const MAX_DENSE_ORDER = 4096

function add_merge_cases!(group::String, A::SparseMatrixCSC{Tv}, B::SparseMatrixCSC{Tv}) where {Tv}
    N = size(A, 1)
    benchmark!(() -> A .+ B, group, "SparseArrays", "CSC", Tv, N)
    for (name, F) in DEVICE_FORMATS
        dA = F{Tv, Int32}(A)
        dB = F{Tv, Int32}(B)
        benchmark!(() -> dA .+ dB, group, "device sparse", name, Tv, N; device = true)
    end
    if N <= MAX_DENSE_ORDER
        dD1 = MtlArray(Matrix(A))
        dD2 = MtlArray(Matrix(B))
        benchmark!(() -> dD1 .+ dD2, group, "device dense", "dense", Tv, N; device = true)
    end
    return nothing
end

# Grid sizes reach n = 512 (N = 262144, ~1.8M stored entries in the sum)
# because the measured CPU/device crossover for the merge is near 460k stored
# entries; a size range that stops short of the crossover cannot verify the
# roadmap's performance goal.
for Tv in (Float32, Float16), n in (16, 64, 256, 512)
    A = laplacian_2d(Tv, n)
    add_merge_cases!("sparse_sparse_add", A, offset_band(Tv, size(A, 1)))
end

for Tv in (Float32,), N in (1024, 4096, 16384)
    A, B = dense_row_pair(Tv, N)
    add_merge_cases!("sparse_sparse_add_dense_row", A, B)
end
