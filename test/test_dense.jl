# Dense conversions: densify agrees with Array(::SparseMatrixCSC) on host and
# device; sparsify agrees with sparse(::Matrix), dropping numerical zeros.

@testset "dense conversions" begin
    if DEVICE_AVAILABLE
        @testset "densify $F Tv=$Tv" for F in SPARSE_TYPES, Tv in ELEMENT_TYPES
            for A in pattern_corpus(Tv)
                dA = F{Tv, Int32}(A)
                Dh = Array(dA)
                @test Dh isa Matrix{Tv}
                @test isequal(Dh, Array(A))
                Dd = MtlMatrix(dA)
                @test Dd isa MtlMatrix{Tv}
                @test isequal(Array(Dd), Array(A))
            end
        end

        @testset "sparsify $F Tv=$Tv" for F in SPARSE_TYPES, Tv in ELEMENT_TYPES
            for A in pattern_corpus(Tv)
                Dh = Array(A)
                reference = sparse(Dh)
                dF = F(Dh)
                @test dF isa F{Tv, MetalSparseArrays.DEFAULT_INDEX_TYPE}
                @test exact_equal(reference, SparseMatrixCSC(dF))
                dFd = F(MtlArray(Dh))
                @test exact_equal(reference, SparseMatrixCSC(dFd))
            end
        end
    else
        @info "dense conversion tests skipped: no functional Metal device"
    end
end
