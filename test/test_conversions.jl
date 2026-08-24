# Conversions between the device formats: every ordered pair preserves the
# (i, j, v) triple set exactly, and conversions copy (mutating the source
# never changes the result).

@testset "format conversions" begin
    if DEVICE_AVAILABLE
        @testset "$F1 -> $F2 Tv=$Tv Ti=$Ti" for F1 in SPARSE_TYPES,
                F2 in SPARSE_TYPES, Tv in ELEMENT_TYPES, Ti in INDEX_TYPES

            F1 === F2 && continue
            for A in pattern_corpus(Tv)
                dA1 = F1{Tv, Ti}(A)
                dA2 = F2(dA1)
                @test dA2 isa F2{Tv, Ti}
                @test size(dA2) == size(A)
                @test nnz(dA2) == nnz(A)
                @test exact_equal(A, SparseMatrixCSC(dA2))
            end
        end

        @testset "conversions copy, not alias" begin
            A = testsparse(Float32, Int, 15, 12; density = 0.2, seed = 10)
            for F1 in SPARSE_TYPES, F2 in SPARSE_TYPES
                F1 === F2 && continue
                dA1 = F1{Float32, Int32}(A)
                dA2 = F2(dA1)
                nonzeros(dA1) .= 0
                @test exact_equal(A, SparseMatrixCSC(dA2))
            end
        end
    else
        @info "format conversion tests skipped: no functional Metal device"
    end
end
