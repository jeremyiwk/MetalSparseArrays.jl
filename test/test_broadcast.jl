# Broadcasting over stored values: pattern preserved exactly for
# zero-preserving functions, ArgumentError for densifying ones.

"""
    broadcast_reference(f, A)

The expected result of a zero-preserving broadcast over the device form of
`A`: the same pattern with the scalar function `f` applied to the stored
values on the host.
"""
function broadcast_reference(f, A::SparseMatrixCSC{<:Any, Ti}) where {Ti}
    return SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), f.(nonzeros(A)))
end

"""
    pattern_and_values_close(expected, B; rtol)

Pattern exactly equal and stored values within `rtol` relative (elementwise,
`isequal` for nonfinite). The comparison for broadcasts of libm-backed
functions (`abs` of a complex number is `hypot`), whose device implementation
may differ from the host by a few ulp; exact arithmetic still uses
`exact_equal`.
"""
function pattern_and_values_close(expected::SparseMatrixCSC, B::SparseMatrixCSC; rtol)
    size(expected) == size(B) || return false
    expected.colptr == B.colptr && expected.rowval == B.rowval || return false
    return all(zip(expected.nzval, B.nzval)) do (x, y)
        isfinite(x) && isfinite(y) ? abs(x - y) <= rtol * max(abs(x), one(abs(x))) :
            isequal(x, y)
    end
end

"""
    device_supports(f, Tv)

Whether broadcasting the scalar function `f` over a dense device vector of
element type `Tv` compiles and runs. When it does not (for example `abs` on
`Complex{BFloat16}`: `hypot` generates invalid IR in `Metal.jl` itself), the
sparse broadcast cannot be expected to work either — the limitation is
upstream, and the sparse case is skipped with a notice rather than failed.
"""
function device_supports(f, ::Type{Tv}) where {Tv}
    return try
        Array(f.(MtlVector{Tv}([one(Tv)])))
        true
    catch
        false
    end
end

@testset "broadcast" begin
    if DEVICE_AVAILABLE
        @testset "zero-preserving $F Tv=$Tv" for F in SPARSE_TYPES, Tv in ELEMENT_TYPES
            two = Tv(2)
            # Each case pairs the device broadcast with the scalar function
            # defining its host reference, and states whether the comparison
            # is bitwise (exact arithmetic) or toleranced (libm-backed).
            cases = [
                (dA -> dA .* two, v -> v * two, :exact),
                (dA -> two .* dA, v -> two * v, :exact),
                (dA -> .-dA, v -> -v, :exact),
                (dA -> abs.(dA), abs, :libm),
                (dA -> conj.(dA), conj, :exact),
                (dA -> dA ./ two, v -> v / two, :exact),
                (dA -> two .* dA .* two, v -> two * v * two, :exact),
                (dA -> identity.(dA), identity, :exact),
            ]
            rtol = 4 * Float64(MetalSparseArrays.unit_roundoff(Tv))
            for (device_f, scalar_f, comparison) in cases
                if !device_supports(scalar_f, Tv)
                    @info "skipping broadcast case for $Tv: unsupported by Metal.jl itself" scalar_f
                    continue
                end
                for A in pattern_corpus(Tv)
                    dA = F{Tv, Int32}(A)
                    dB = device_f(dA)
                    @test dB isa F
                    expected = broadcast_reference(scalar_f, A)
                    B = SparseMatrixCSC(dB)
                    if comparison === :exact
                        @test exact_equal(expected, B)
                    else
                        @test pattern_and_values_close(expected, B; rtol)
                    end
                end
            end
        end

        @testset "element type change" begin
            A = testsparse(ComplexF32, Int, 11, 7; density = 0.3, seed = 14)
            dA = MtlSparseMatrixCSR{ComplexF32, Int32}(A)
            dB = abs2.(dA)
            @test dB isa MtlSparseMatrixCSR{Float32, Int32}
            @test exact_equal(broadcast_reference(abs2, A), SparseMatrixCSC(dB))
        end

        @testset "densifying broadcasts throw" begin
            A = testsparse(Float32, Int, 6, 6; seed = 15)
            for F in SPARSE_TYPES
                dA = F{Float32, Int32}(A)
                @test_throws ArgumentError dA .+ 1
                @test_throws ArgumentError cos.(dA)
                @test_throws ArgumentError dA .^ 0
                @test_throws ArgumentError (dA .* 2) .+ 1
            end
        end

        @testset "unsupported operand mixes throw" begin
            A = testsparse(Float32, Int, 6, 6; seed = 16)
            dA = MtlSparseMatrixCSC{Float32, Int32}(A)
            dB = MtlSparseMatrixCSC{Float32, Int32}(A)
            @test_throws ArgumentError dA .+ dB
            @test_throws ArgumentError dA .* dB
        end
    else
        @info "broadcast tests skipped: no functional Metal device"
    end
end
