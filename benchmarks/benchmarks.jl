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
# the measurement includes host dispatch, allocation and device completion) and a host
# case by `Base.@elapsed` (the timing core of `Base.@time`). Each case is run
# once untimed to compile, then minimum, median and p95 over `REPS` repetitions
# are reported; single-shot timings measured ~250 us high against minimums on this
# hardware.

using LinearAlgebra: I, kron
using Metal: Metal, MtlArray
using MetalSparseArrays
using SparseArrays: SparseMatrixCSC, sparse, spdiagm, nnz, findnz

"""
    Benchmark(group, key, device, thunk, evals)

One benchmark case: `thunk` is the zero-argument operation to time, `group`
names the operation, `key` identifies representation, format, element type,
and size, `device` selects GPU-synchronized timing, and `evals` counts operations
per synchronization. Reported times are normalized per operation.
"""
struct Benchmark
    group::String
    key::Tuple
    device::Bool
    thunk::Function
    evals::Int
end

# CPU CSR storage is represented by CSC of the transpose, so transpose
# materialization measures the same index/value reorder without a CSR dependency.
function add_reorder_cases!(A, pattern)
    Tv, Ti = eltype(A), eltype(A.colptr)
    m, n = size(A)
    At = sparse(transpose(A))
    csc = MtlSparseMatrixCSC{Tv, Ti}(A)
    csr = MtlSparseMatrixCSR{Tv, Ti}(A)
    coo = MtlSparseMatrixCOO{Tv, Ti}(A)
    for (name, source, F, host) in (
            ("CSC_to_CSR", csc, MtlSparseMatrixCSR, A),
            ("CSR_to_CSC", csr, MtlSparseMatrixCSC, At),
        )
        benchmark!(() -> sparse(transpose(host)), "format_reorder", "SparseArrays", name, Tv, Ti, m, n, nnz(A), pattern)
        benchmark!(() -> F(source), "format_reorder", "device sparse", name, Tv, Ti, m, n, nnz(A), pattern; device = true)
    end
    benchmark!(() -> MtlSparseMatrixCSC(coo), "format_reorder", "device sparse", "COO_to_CSC", Tv, Ti, m, n, nnz(A), pattern; device = true)
    benchmark!(() -> MtlSparseMatrixCOO(csc), "format_reorder", "device sparse", "CSC_to_COO", Tv, Ti, m, n, nnz(A), pattern; device = true)
    benchmark!(() -> A .+ A, "mixed_sparse_add", "SparseArrays", "CSC", Tv, Ti, m, n, nnz(A), pattern)
    benchmark!(() -> csr .+ csc, "mixed_sparse_add", "device sparse", "CSR_CSC", Tv, Ti, m, n, nnz(A), pattern; device = true)
    benchmark!(() -> csr .+ csr, "mixed_sparse_add", "device sparse", "CSR_CSR", Tv, Ti, m, n, nnz(A), pattern; device = true)
    return nothing
end

function add_assembly_cases!(I, J, V, m, n, pattern)
    Tv, Ti = eltype(V), eltype(I)
    di, dj, dv = MtlArray(I), MtlArray(J), MtlArray(V)
    benchmark!(() -> sparse(I, J, V, m, n), "coo_assembly", "SparseArrays", "CSC", Tv, Ti, m, n, length(V), pattern)
    for fmt in (:csc, :csr, :coo)
        benchmark!(() -> sparse(di, dj, dv, m, n; fmt), "coo_assembly", "device sparse", fmt, Tv, Ti, m, n, length(V), pattern; device = true)
    end
    return nothing
end

const SUITE = Benchmark[]

benchmark!(thunk::Function, group::String, key...; device::Bool = false, evals::Int = 1) =
    push!(SUITE, Benchmark(group, key, device, thunk, evals))

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

# Match CPU and device index widths when measuring array-interface overhead.
for Tv in (Float32, ComplexF32), Ti in (Int32, Int64), N in (256, 4096, 65536, 262144)
    A = SparseMatrixCSC{Tv, Ti}(
        spdiagm(
            -1 => fill(-one(Tv), N - 1), 0 => fill(Tv(2), N), 1 => fill(-one(Tv), N - 1)
        )
    )
    for (name, f) in (
            ("copy", copy), ("similar", similar),
            ("unary_scale", a -> a .* Tv(2)),
            ("inplace_scale", a -> (a .*= -one(Tv))), ("findnz", findnz),
        )
        benchmark!(() -> f(A), name, "SparseArrays", "CSC", Tv, Ti, N, nnz(A))
        for F in (MtlSparseMatrixCSR, MtlSparseMatrixCSC, MtlSparseMatrixCOO)
            dA = F{Tv, Ti}(A)
            benchmark!(() -> f(dA), name, "device sparse", nameof(F), Tv, Ti, N, nnz(A); device = true)
        end
    end
    if N <= MAX_DENSE_ORDER
        benchmark!(() -> Matrix(A), "densify", "SparseArrays", "CSC", Tv, Ti, N, nnz(A))
        for F in (MtlSparseMatrixCSR, MtlSparseMatrixCSC, MtlSparseMatrixCOO)
            dA = F{Tv, Ti}(A)
            benchmark!(() -> Metal.MtlMatrix(dA), "densify", "device sparse", nameof(F), Tv, Ti, N, nnz(A); device = true)
        end
    end
    benchmark!(() -> (A .= zero(Tv)), "zero_fill_batch", "SparseArrays", "CSC", Tv, Ti, N, nnz(A); evals = 100)
    for F in (MtlSparseMatrixCSR, MtlSparseMatrixCSC, MtlSparseMatrixCOO)
        dA = F{Tv, Ti}(A)
        benchmark!(() -> (dA .= zero(Tv)), "zero_fill_batch", "device sparse", nameof(F), Tv, Ti, N, nnz(A); device = true, evals = 100)
    end
end

for Tv in (Float32, ComplexF32), Ti in (Int32, Int64)
    for N in (4096, 65536, 262144)
        A = spdiagm(-1 => fill(-one(Tv), N - 1), 0 => fill(Tv(2), N), 1 => fill(-one(Tv), N - 1))
        add_reorder_cases!(SparseMatrixCSC{Tv, Ti}(A), "tridiagonal")
    end
    A, _ = dense_row_pair(Tv, 65536)
    for (name, B) in (("dense_row", A), ("dense_column", sparse(transpose(A))))
        add_reorder_cases!(SparseMatrixCSC{Tv, Ti}(B), name)
    end
    A = sparse(collect(1:65536), mod1.(collect(1:65536), 17), fill(one(Tv), 65536), 65536, 17)
    add_reorder_cases!(SparseMatrixCSC{Tv, Ti}(A), "rectangular")
end

for Tv in (Float32, ComplexF32), Ti in (Int32, Int64)
    for N in (256, 4096, 65536, 262144)
        I = reverse(repeat(Ti.(1:N); inner = 3))
        J = reverse(Ti.(mod1.(repeat(collect(1:N); inner = 3) + repeat([-1, 0, 1], N), N)))
        V = fill(one(Tv), length(I))
        add_assembly_cases!(I, J, V, N, N, "reverse_bands")
    end
    K, N = 262144, 16384
    I = Ti.(mod1.(collect(1:K), N))
    J = Ti.(mod1.(7 .* collect(1:K), N))
    add_assembly_cases!(I, J, fill(one(Tv), K), N, N, "duplicates_16")
    add_assembly_cases!(ones(Ti, K), ones(Ti, K), fill(one(Tv), K), 1, 1, "all_duplicates")
end
