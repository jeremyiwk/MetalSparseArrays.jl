# MtlSparseMatrixCSC: construction, validation, round trip, show, adapt.

@testset "MtlSparseMatrixCSC" begin
    if DEVICE_AVAILABLE
        @testset "round trip Tv=$Tv Ti=$Ti" for Tv in ELEMENT_TYPES, Ti in INDEX_TYPES
            for A in pattern_corpus(Tv)
                dA = MtlSparseMatrixCSC{Tv, Ti}(A)
                @test dA isa MtlSparseMatrixCSC{Tv, Ti}
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

        @testset "adapt to the device" begin
            A = testsparse(Float32, Int, 12, 8; seed = 9)
            dA = adapt(MtlArray, A)
            @test dA isa
                MtlSparseMatrixCSC{Float32, MetalSparseArrays.DEFAULT_INDEX_TYPE}
            @test exact_equal(A, adapt(Array, dA))
        end

        @testset "raw construction and validation" begin
            Ti = Int32
            dev(v, T) = MtlVector{T}(T.(v))
            # A = [0 1 0; 2 0 3] in CSC: colptr [1, 2, 3, 4], rowval [2, 1, 2].
            colptr = dev([1, 2, 3, 4], Ti)
            rowval = dev([2, 1, 2], Ti)
            nzval = dev([2, 1, 3], Float32)
            dA = MtlSparseMatrixCSC(2, 3, colptr, rowval, nzval)
            @test SparseMatrixCSC(dA) ==
                sparse([2, 1, 2], [1, 2, 3], Float32[2, 1, 3], 2, 3)

            # Oversized buffers are accepted with their tails ignored — the
            # tail values (an out-of-range index included) never participate —
            # exactly as SparseMatrixCSC accepts rowval/nzval longer than nnz.
            # nnz comes from the pointer array, and copy compacts.
            dover = MtlSparseMatrixCSC(
                2, 3, colptr, dev([2, 1, 2, 9, 7], Ti), dev([2, 1, 3, 99, 42], Float32)
            )
            @test nnz(dover) == 3
            @test exact_equal(SparseMatrixCSC(dA), SparseMatrixCSC(dover))
            compact = copy(dover)
            @test nnz(compact) == 3
            @test length(nonzeros(compact)) == 3
            @test exact_equal(SparseMatrixCSC(dA), SparseMatrixCSC(compact))

            # Each invariant violated in turn; the error names the invariant.
            @test_throws ArgumentError MtlSparseMatrixCSC(
                2, 3, dev([2, 2, 3, 4], Ti), rowval, nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSC(
                2, 3, dev([1, 3, 2, 4], Ti), rowval, nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSC(
                2, 4, colptr, rowval, nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSC(
                2, 3, colptr, dev([2, 1], Ti), nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSC(
                2, 3, colptr, rowval, dev([1, 2], Float32)
            )
            @test_throws ArgumentError MtlSparseMatrixCSC(
                2, 3, colptr, dev([2, 3, 2], Ti), nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSC(
                2, 3, colptr, dev([2, 0, 2], Ti), nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSC(
                2, -1, dev([1], Ti), dev(Ti[], Ti), dev(Float32[], Float32)
            )
        end

        @testset "index type overflow" begin
            n = Int64(typemax(Int32)) + 1
            A = spzeros(Float32, 1, n)
            @test_throws ArgumentError MtlSparseMatrixCSC(A)
        end

        @testset "show" begin
            A = testsparse(Float32, Int, 7, 9; seed = 7)
            dA = MtlSparseMatrixCSC(A)
            text = repr(MIME"text/plain"(), dA)
            @test occursin("7×9", text)
            @test occursin("MtlSparseMatrixCSC{Float32, Int32}", text)
            @test occursin("$(nnz(A)) stored", text)
        end

        @testset "nonzeros aliases device storage" begin
            dA = MtlSparseMatrixCSC(testsparse(Float32, Int, 8, 8; seed = 8))
            v = nonzeros(dA)
            @test v isa MtlVector{Float32}
            @test v === dA.nzval
        end
    else
        @info "MtlSparseMatrixCSC tests skipped: no functional Metal device"
    end
end
