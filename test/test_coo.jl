# MtlSparseMatrixCOO: construction, validation, round trip, show, adapt.

@testset "MtlSparseMatrixCOO" begin
    if DEVICE_AVAILABLE
        @testset "round trip Tv=$Tv Ti=$Ti" for Tv in ELEMENT_TYPES, Ti in INDEX_TYPES
            for A in pattern_corpus(Tv)
                dA = MtlSparseMatrixCOO{Tv, Ti}(A)
                @test dA isa MtlSparseMatrixCOO{Tv, Ti}
                @test size(dA) == size(A)
                @test nnz(dA) == nnz(A)
                B = SparseMatrixCSC(dA)
                @test B isa SparseMatrixCSC{Tv, Ti}
                @test exact_equal(A, B)
                C = adapt(Array, dA)
                @test C isa SparseMatrixCSC{Tv, Ti}
                @test exact_equal(A, C)
            end
        end

        @testset "default index type" begin
            A = testsparse(Float32, Int, 10, 10; seed = 6)
            dA = MtlSparseMatrixCOO(A)
            @test dA isa MtlSparseMatrixCOO{Float32, MetalSparseArrays.DEFAULT_INDEX_TYPE}
            @test exact_equal(A, SparseMatrixCSC(dA))
        end

        @testset "raw construction and validation" begin
            Ti = Int32
            dev(v, T) = MtlVector{T}(T.(v))
            # A = [0 1 0; 2 0 3] in row-major coordinates.
            rowval = dev([1, 2, 2], Ti)
            colval = dev([2, 1, 3], Ti)
            nzval = dev([1, 2, 3], Float32)
            dA = MtlSparseMatrixCOO(2, 3, rowval, colval, nzval)
            @test SparseMatrixCSC(dA) ==
                sparse([1, 2, 2], [2, 1, 3], Float32[1, 2, 3], 2, 3)

            # Each invariant violated in turn; the error names the invariant.
            @test_throws ArgumentError MtlSparseMatrixCOO(
                2, 3, dev([1, 2], Ti), colval, nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCOO(
                2, 3, rowval, colval, dev([1, 2], Float32)
            )
            @test_throws ArgumentError MtlSparseMatrixCOO(
                2, 3, dev([1, 3, 2], Ti), colval, nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCOO(
                2, 3, dev([0, 2, 2], Ti), colval, nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCOO(
                2, 3, rowval, dev([2, 1, 4], Ti), nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCOO(
                -1, 3, dev(Ti[], Ti), dev(Ti[], Ti), dev(Float32[], Float32)
            )
        end

        @testset "index type overflow" begin
            m = Int64(typemax(Int32)) + 1
            A = spzeros(Float32, m, 1)
            @test_throws ArgumentError MtlSparseMatrixCOO(A)
        end

        @testset "show" begin
            A = testsparse(Float32, Int, 7, 9; seed = 7)
            dA = MtlSparseMatrixCOO(A)
            text = repr(MIME"text/plain"(), dA)
            @test occursin("7×9", text)
            @test occursin("MtlSparseMatrixCOO{Float32, Int32}", text)
            @test occursin("$(nnz(A)) stored", text)
        end

        @testset "nonzeros aliases device storage" begin
            dA = MtlSparseMatrixCOO(testsparse(Float32, Int, 8, 8; seed = 8))
            v = nonzeros(dA)
            @test v isa MtlVector{Float32}
            @test v === dA.nzval
        end
    else
        @info "MtlSparseMatrixCOO tests skipped: no functional Metal device"
    end
end
