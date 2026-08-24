# Broadcasting over stored values. The primary criterion is backend
# consistency: the sparse broadcast must be bit-identical to the same function
# broadcast over the same values as a dense device vector — both run the same
# Metal libm, so this isolates our layer and needs no tolerance. The device
# libm's agreement with the host is a separate, documented assumption checked
# once at the end.

@testset "broadcast" begin
    if DEVICE_AVAILABLE
        @testset "zero-preserving $F Tv=$Tv" for F in SPARSE_TYPES, Tv in ELEMENT_TYPES
            two = Tv(2)
            # Each case pairs the device broadcast with the scalar function
            # that defines its reference.
            # `:exact` cases are exact arithmetic, bit-identical between host
            # and device; `:libm` cases (abs of a complex number is hypot) may
            # differ from the host by a few ulp, so their host agreement is
            # checked once in the libm-assumption test set below instead.
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
            for (device_f, scalar_f, comparison) in cases
                if !device_supports(scalar_f, Tv)
                    @info "skipping broadcast case for $Tv: unsupported by Metal.jl itself" scalar_f
                    continue
                end
                for A in pattern_corpus(Tv)
                    dA = F{Tv, Int32}(A)
                    dB = device_f(dA)
                    @test dB isa F
                    # Pattern preserved exactly, stored zeros included.
                    @test same_pattern(A, dB)
                    # Backend consistency, tolerance-free: the sparse broadcast
                    # must be bit-identical to the dense device broadcast over
                    # the matrix's own stored values (same order, same libm).
                    device_reference = Array(scalar_f.(nonzeros(dA)))
                    @test isequal(Array(nonzeros(dB)), device_reference)
                    # Host agreement for exact arithmetic, in storage order.
                    if comparison === :exact
                        host_reference = scalar_f.(Array(nonzeros(dA)))
                        @test isequal(Array(nonzeros(dB)), host_reference)
                    end
                end
            end
        end

        @testset "element type change" begin
            A = testsparse(ComplexF32, Int, 11, 7; density = 0.3, seed = 14)
            dA = MtlSparseMatrixCSR{ComplexF32, Int32}(A)
            dB = abs2.(dA)
            @test dB isa MtlSparseMatrixCSR{Float32, Int32}
            @test same_pattern(A, dB)
            @test Array(nonzeros(dB)) == Array(abs2.(nonzeros(dA)))
        end

        @testset "densifying broadcasts give dense device results $F" for F in SPARSE_TYPES
            A = testsparse(Float32, Int, 6, 6; seed = 15)
            dA = F{Float32, Int32}(A)
            D = Array(A)
            # Comparison is against the same expression over the dense device
            # matrix (bit-exact, same backend); host agreement additionally
            # for exact arithmetic (cos is libm-backed and may differ by ulp).
            for (device_f, host_f, comparison) in [
                    (dA -> dA .+ 1, D -> D .+ 1, :exact),
                    (dA -> cos.(dA), D -> cos.(D), :libm),
                    (dA -> dA .^ 0, D -> D .^ 0, :exact),
                    (dA -> (dA .* 2) .+ 1, D -> (D .* 2) .+ 1, :exact),
                    (dA -> dA .* NaN32, D -> D .* NaN32, :exact),
                ]
                dB = device_f(dA)
                @test dB isa MtlMatrix{Float32}
                @test isequal(Array(dB), Array(host_f(MtlArray(D))))
                comparison === :exact && @test isequal(Array(dB), host_f(D))
            end
        end

        @testset "sparse with dense operand densifies $F" for F in SPARSE_TYPES
            A = testsparse(Float32, Int, 6, 6; seed = 16)
            dA = F{Float32, Int32}(A)
            dD = MtlArray(fill(2.0f0, 6, 6))
            dB = dA .+ dD
            @test dB isa MtlMatrix{Float32}
            @test Array(dB) == Array(A) .+ 2.0f0
            dC = dA .* dD
            @test dC isa MtlMatrix{Float32}
            @test Array(dC) == Array(A) .* 2.0f0
        end

        @testset "sparse-sparse union $F Tv=$Tv" for F in SPARSE_TYPES,
                Tv in (Float32, ComplexF32)

            pairs = [
                (
                    testsparse(Tv, Int, 17, 13; density = 0.15, seed = 20),
                    testsparse(Tv, Int, 17, 13; density = 0.15, seed = 21),
                ),
                # Disjoint patterns.
                (
                    sparse(1:5, 1:5, fill(Tv(2), 5), 5, 5),
                    sparse(1:4, 2:5, fill(Tv(3), 4), 5, 5),
                ),
                # One operand empty.
                (testsparse(Tv, Int, 6, 6; seed = 22), spzeros(Tv, 6, 6)),
            ]
            for (A, B) in pairs, op in (+, -, *)
                dA = F{Tv, Int32}(A)
                dB2 = F{Tv, Int32}(B)
                dC = broadcast(op, dA, dB2)
                @test dC isa F{Tv, Int32}
                @test exact_equal(broadcast(op, A, B), SparseMatrixCSC(dC))
            end
        end

        @testset "sparse-sparse mixed format and densifying" begin
            A = testsparse(Float32, Int, 8, 8; seed = 23)
            B = testsparse(Float32, Int, 8, 8; seed = 24)
            dR = MtlSparseMatrixCSR{Float32, Int32}(A)
            dC = MtlSparseMatrixCSC{Float32, Int32}(B)
            # Result format follows the first sparse operand.
            mixed = dR .+ dC
            @test mixed isa MtlSparseMatrixCSR{Float32, Int32}
            @test exact_equal(A .+ B, SparseMatrixCSC(mixed))
            # f(0, 0) != 0 densifies, per the CUDA convention.
            eq = dR .== dC
            @test eq isa MtlMatrix{Bool}
            @test Array(eq) == (Array(A) .== Array(B))
        end

        @testset "in-place A .= rhs $F" for F in SPARSE_TYPES
            A0 = sparse([1, 3], [1, 2], Float32[1, 2], 3, 3)
            D = Float32[0 5 0; 0 0 0; 6 0 0]
            B = sparse([2], [3], Float32[9], 3, 3)
            # Each case applies the identical assignment to the device matrix
            # and to a host copy; stdlib defines correct.
            cases = [
                (dA -> dA .= 0, hA -> hA .= 0),
                (dA -> dA .= 1, hA -> hA .= 1),
                (dA -> dA .= D, hA -> hA .= D),
                (dA -> dA .= MtlArray(D), hA -> hA .= D),
                (dA -> dA .= F{Float32, Int32}(B), hA -> hA .= B),
                (dA -> dA .= dA .+ 1, hA -> hA .= hA .+ 1),
                (dA -> dA .*= 2, hA -> hA .*= 2),
            ]
            for (device_case, host_case) in cases
                dA = F{Float32, Int32}(A0)
                hA = copy(A0)
                returned = device_case(dA)
                host_case(hA)
                @test returned === dA
                @test dA isa F{Float32, Int32}
                @test exact_equal(hA, SparseMatrixCSC(dA))
            end
        end

        @testset "device libm accuracy assumption" begin
            # The one place the device libm is compared against the host: for
            # libm-backed functions (abs of complex is hypot) the two
            # implementations are each accurate to a few ulp, so they agree to
            # 4u relative. Everything else in this file is bit-exact against
            # the device reference and needs no tolerance.
            for Tv in ELEMENT_TYPES
                device_supports(abs, Tv) || continue
                rtol = 4 * Float64(MetalSparseArrays.unit_roundoff(Tv))
                values = nonzeros(testsparse(Tv, Int, 40, 40; density = 0.3, seed = 17))
                dev = Array(abs.(MtlVector{Tv}(values)))
                host = abs.(values)
                @test all(
                    abs(d - h) <= rtol * max(abs(h), one(abs(h)))
                        for (d, h) in zip(dev, host)
                )
            end
        end
    else
        @info "broadcast tests skipped: no functional Metal device"
    end
end
