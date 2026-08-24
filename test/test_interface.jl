# Array interface: similar, copy, collect, rowvals, findnz, scalar indexing.

@testset "array interface" begin
    if DEVICE_AVAILABLE
        @testset "similar $F Tv=$Tv" for F in SPARSE_TYPES, Tv in ELEMENT_TYPES
            A = testsparse(Tv, Int, 13, 17; density = 0.2, seed = 11)
            dA = F{Tv, Int32}(A)

            S = similar(dA)
            @test S isa typeof(dA)
            @test same_pattern(A, S)

            T2 = Tv <: Complex ? Float32 : ComplexF32
            S2 = similar(dA, T2)
            @test S2 isa F{T2, Int32}
            @test same_pattern(A, S2)

            S3 = similar(dA, (4, 6))
            @test S3 isa F{Tv, Int32}
            @test size(S3) == (4, 6)
            @test nnz(S3) == 0

            S4 = similar(dA, T2, (3, 2))
            @test S4 isa F{T2, Int32}
            @test size(S4) == (3, 2)
            @test nnz(S4) == 0
        end

        @testset "copy and collect $F Tv=$Tv" for F in SPARSE_TYPES, Tv in ELEMENT_TYPES
            A = testsparse(Tv, Int, 13, 17; density = 0.2, seed = 12)
            dA = F{Tv, Int32}(A)
            dB = copy(dA)
            @test exact_equal(A, SparseMatrixCSC(dB))
            nonzeros(dB) .= 0
            @test exact_equal(A, SparseMatrixCSC(dA))
            @test isequal(collect(dA), Array(A))
        end

        @testset "rowvals and findnz $F Tv=$Tv" for F in SPARSE_TYPES, Tv in ELEMENT_TYPES
            for A in pattern_corpus(Tv)
                dA = F{Tv, Int32}(A)
                I, J, V = findnz(dA)
                Iref, Jref, Vref = findnz(A)
                @test Array(I) == Iref
                @test Array(J) == Jref
                @test isequal(Array(V), Vref)
            end
            dC = MtlSparseMatrixCSC(testsparse(Tv, Int, 9, 9; seed = 13))
            @test rowvals(dC) === dC.rowval
        end

        @testset "scalar getindex $F" for F in SPARSE_TYPES
            A = sparse([1, 2, 4], [1, 3, 2], Float32[5, 0, 7], 5, 4)
            dA = F{Float32, Int32}(A)
            D = Array(A)
            Metal.@allowscalar begin
                for i in 1:5, j in 1:4
                    @test dA[i, j] == D[i, j]
                end
            end
            @test_throws BoundsError Metal.@allowscalar(dA[6, 1])
            @test_throws BoundsError Metal.@allowscalar(dA[1, 5])
            # Without @allowscalar the device read itself must throw.
            @test try
                dA[1, 1]
                false
            catch
                true
            end
        end

        @testset "scalar setindex! $F" for F in SPARSE_TYPES
            A = sparse([1, 2, 4], [1, 3, 2], Float32[5, 0, 7], 5, 4)
            dA = F{Float32, Int32}(A)
            Metal.@allowscalar begin
                dA[2, 3] = 9
                @test dA[2, 3] == 9.0f0
                @test_throws ArgumentError dA[1, 2] = 1
            end
        end
    else
        @info "array interface tests skipped: no functional Metal device"
    end
end
