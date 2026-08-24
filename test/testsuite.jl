# Shared helpers for running every test set on the CPU reference and on the device.

using MetalSparseArrays
using Adapt
using LinearAlgebra
using Metal
using Random
using SparseArrays
using StableRNGs
using Test

const BFloat16 = Metal.BFloat16

"""
    DEVICE_AVAILABLE

Whether a Metal device is present and usable. Test sets that require a device are
skipped when this is `false`, so the suite still runs on machines without one.
"""
const DEVICE_AVAILABLE = Metal.functional()

# Guard against silently reduced coverage: an environment that promises a device
# (CI on Apple Silicon runners) fails loudly if none is found, instead of green
# runs that quietly skipped every device test set.
if get(ENV, "CI_EXPECT_DEVICE", "") == "true" && !DEVICE_AVAILABLE
    error(
        "CI_EXPECT_DEVICE=true but Metal.functional() is false: the device test " *
            "sets would be skipped. Fix the runner or unset CI_EXPECT_DEVICE."
    )
end

"""
    CANDIDATE_ELEMENT_TYPES

Every element type the suite knows how to build a test problem for. Which of them a
given backend actually supports is decided at run time by `supported_element_types`;
supporting a new element type means adding it to this tuple and nothing else.
"""
const CANDIDATE_ELEMENT_TYPES = (
    Float16, Float32, Float64, BFloat16,
    ComplexF16, ComplexF32, ComplexF64,
    Complex{BFloat16},
)

"""
    supported_element_types(ArrayType)

Those of `CANDIDATE_ELEMENT_TYPES` that `ArrayType` can store, broadcast over, and
reduce, determined by trying each one. The set depends on the device and on the
Metal version, so it is probed rather than fixed: Apple GPUs have no double
precision unit, but they do support `Float16` and `BFloat16` alongside `Float32`,
and every complex type built on those.
"""
function supported_element_types(::Type{AT}) where {AT}
    return filter(collect(CANDIDATE_ELEMENT_TYPES)) do T
        try
            a = AT(ones(T, 4))
            sum(a .+ a)
            true
        catch
            false
        end
    end
end

"""
    ELEMENT_TYPES

Element types the device supports, or all candidates when no device is present.
Every test set loops over this list, so an element type is covered by the whole
suite as soon as the backend accepts it.
"""
const ELEMENT_TYPES = DEVICE_AVAILABLE ? supported_element_types(MtlArray) :
    collect(CANDIDATE_ELEMENT_TYPES)

"""
    INDEX_TYPES

Index types (`Ti`) that device-resident sparse formats support. `Int32` is the
default and matches `CUDA.CUSPARSE`.
"""
const INDEX_TYPES = (Int32,)

# Make the exercised configuration visible in every test log, so a coverage
# regression (a type silently dropping out of the probed set) is observable.
@info "MetalSparseArrays test configuration" DEVICE_AVAILABLE ELEMENT_TYPES INDEX_TYPES

"""
    SPARSE_TYPES

Device sparse matrix types the suite runs against. Each format is appended here as
it is implemented, and every format-independent test set loops over this list.
"""
const SPARSE_TYPES = Any[MtlSparseMatrixCOO, MtlSparseMatrixCSC, MtlSparseMatrixCSR]

# Cross-cutting requirement: library code never scalar indexes a device array.
# Any code path that tries fails the suite instead of silently round tripping
# through the host.
DEVICE_AVAILABLE && Metal.allowscalar(false)

"""
    exact_equal(A, B)

Structural and value equality of two `SparseMatrixCSC`s: equal dimensions and
elementwise equal `colptr`, `rowval`, and `nzval` (compared with `isequal`, so
stored `NaN`s compare equal and stored zeros are distinguished from structural
zeros). This is the round-trip criterion for host-device-host conversion, which
is exact, not approximate; index arrays may differ in integer type but not in
value.
"""
function exact_equal(A::SparseMatrixCSC, B::SparseMatrixCSC)
    return size(A) == size(B) && A.colptr == B.colptr && A.rowval == B.rowval &&
        isequal(A.nzval, B.nzval)
end

"""
    same_pattern(A, dB)

Whether the device matrix `dB` has exactly the sparsity pattern of the host
`A`: equal dimensions and index arrays, values ignored.
"""
function same_pattern(A::SparseMatrixCSC, dB::AbstractMtlSparseMatrix)
    B = SparseMatrixCSC(dB)
    return size(A) == size(B) && A.colptr == B.colptr && A.rowval == B.rowval
end

"""
    device_supports(f, Tv)

Whether broadcasting the scalar function `f` over a dense device vector of
element type `Tv` compiles and runs. When it does not (for example `abs` on
`Complex{BFloat16}`: `hypot` fails to compile in `Metal.jl` itself — see
`metal_bfloat16_hypot_bug.jl`), the sparse case is skipped with a notice
rather than failed: the limitation is upstream, and a sparse-side regression
would still fail because the probe uses the dense array path only.
"""
function device_supports(f, ::Type{Tv}) where {Tv}
    return try
        Array(f.(MtlVector{Tv}([one(Tv)])))
        true
    catch
        false
    end
end

"""
    pattern_corpus(Tv)

The corpus of host matrices format round-trip tests run over: random patterns
over several seeds and densities, the shape edge cases, a dense row and a dense
column, explicit stored zeros, and (for float types) stored nonfinite values.
"""
function pattern_corpus(::Type{Tv}) where {Tv}
    patterns = SparseMatrixCSC{Tv, Int}[]
    for seed in 0:3, density in (0.05, 0.3)
        push!(patterns, testsparse(Tv, Int, 23, 31; density, seed))
    end
    push!(patterns, spzeros(Tv, 0, 0))
    push!(patterns, spzeros(Tv, 0, 5))
    push!(patterns, spzeros(Tv, 5, 0))
    push!(patterns, spzeros(Tv, 6, 9))
    push!(patterns, sparse([1], [1], [Tv(3)], 1, 1))
    push!(patterns, sparse([4], [2], [Tv(-1)], 9, 7))
    push!(patterns, sparse(fill(3, 10), collect(1:10), fill(Tv(2), 10), 10, 10))
    push!(patterns, sparse(collect(1:10), fill(4, 10), fill(Tv(2), 10), 10, 10))
    push!(patterns, testsparse(Tv, Int, 200, 3; density = 0.2, seed = 4))
    push!(patterns, testsparse(Tv, Int, 3, 200; density = 0.2, seed = 5))
    # Explicit stored zero: sparse(I, J, V) keeps numerical zeros it is given.
    push!(patterns, sparse([1, 2, 5], [1, 1, 3], [zero(Tv), one(Tv), zero(Tv)], 5, 5))
    if Tv <: Union{AbstractFloat, Complex}
        push!(patterns, sparse([1, 3], [2, 2], [Tv(NaN), Tv(Inf)], 4, 4))
    end
    return patterns
end

"""
    referencetype(T)

The double precision CPU element type corresponding to `T`: `Float64` for real `T`
and `ComplexF64` for complex `T`. A result computed in a low precision element type
is validated against a reference computed in this type, with the tolerance scaled by
`unit_roundoff(T)` rather than by a constant.
"""
referencetype(::Type{<:Real}) = Float64
referencetype(::Type{<:Complex}) = ComplexF64

"""
    uniform(rng, T, dims...)

An array of independent entries uniform on `[-1, 1]` for real `T`, and with real and
imaginary parts independent and uniform on `[-1, 1]` for complex `T`.
"""
function uniform(rng::AbstractRNG, ::Type{R}, dims::Integer...) where {R <: Real}
    return 2 .* rand(rng, R, dims...) .- one(R)
end

function uniform(rng::AbstractRNG, ::Type{Complex{R}}, dims::Integer...) where {R <: Real}
    return complex.(uniform(rng, R, dims...), uniform(rng, R, dims...))
end

"""
    testsparse(Tv, Ti, m, n; density, seed)

An `m`-by-`n` `SparseMatrixCSC{Tv,Ti}` with approximately `density * m * n` stored
entries drawn uniformly from `[-1, 1]` (or its complex analogue). The pattern and
the values are generated in double precision and then rounded to `Tv`, so the same
`seed` gives the same matrix in every element type and results across precisions are
directly comparable. Used as the CPU reference that device results are validated
against.
"""
function testsparse(
        ::Type{Tv}, ::Type{Ti}, m::Integer, n::Integer;
        density = 0.1, seed = 0
    ) where {Tv, Ti}
    rng = StableRNG(seed)
    A = sprand(rng, referencetype(Tv), m, n, density)
    nonzeros(A) .= uniform(rng, referencetype(Tv), nnz(A))
    return SparseMatrixCSC{Tv, Ti}(A)
end

"""
    laplacian_2d(Tv, n)

The `n^2`-by-`n^2` sparse matrix of the five point finite difference discretization
of the negative Laplacian on an `n`-by-`n` grid with homogeneous Dirichlet boundary
conditions. Symmetric positive definite with condition number growing like `n^2`;
the standard test problem for conjugate gradient and for preconditioners.
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
