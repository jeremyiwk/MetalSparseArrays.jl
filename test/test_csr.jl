# MtlSparseMatrixCSR: construction, validation, round trip, show, adapt.

"""
    csr_patterns(Tv)

The corpus of host matrices the CSR round-trip tests run over: random patterns
over several seeds and densities, the shape edge cases, a dense row and a dense
column, explicit stored zeros, and (for float types) stored nonfinite values.
"""
function csr_patterns(::Type{Tv}) where {Tv}
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

@testset "MtlSparseMatrixCSR" begin
    if DEVICE_AVAILABLE
        @testset "round trip Tv=$Tv Ti=$Ti" for Tv in ELEMENT_TYPES, Ti in INDEX_TYPES
            for A in csr_patterns(Tv)
                dA = MtlSparseMatrixCSR{Tv, Ti}(A)
                @test dA isa MtlSparseMatrixCSR{Tv, Ti}
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
            dA = MtlSparseMatrixCSR(A)
            @test dA isa MtlSparseMatrixCSR{Float32, MetalSparseArrays.DEFAULT_INDEX_TYPE}
            @test exact_equal(A, SparseMatrixCSC(dA))
        end

        @testset "raw construction and validation" begin
            Ti = Int32
            dev(v, T) = MtlVector{T}(T.(v))
            rowptr = dev([1, 2, 4], Ti)
            colval = dev([2, 1, 3], Ti)
            nzval = dev([1, 2, 3], Float32)
            dA = MtlSparseMatrixCSR(2, 3, rowptr, colval, nzval)
            @test SparseMatrixCSC(dA) ==
                sparse([1, 2, 2], [2, 1, 3], Float32[1, 2, 3], 2, 3)

            # Each invariant violated in turn; the error names the invariant.
            @test_throws ArgumentError MtlSparseMatrixCSR(
                2, 3, dev([2, 2, 4], Ti), colval, nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSR(
                2, 3, dev([1, 4, 2], Ti), colval, nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSR(
                3, 3, rowptr, colval, nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSR(
                2, 3, rowptr, dev([2, 1], Ti), nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSR(
                2, 3, rowptr, colval, dev([1, 2], Float32)
            )
            @test_throws ArgumentError MtlSparseMatrixCSR(
                2, 3, rowptr, dev([2, 1, 4], Ti), nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSR(
                2, 3, rowptr, dev([2, 0, 3], Ti), nzval
            )
            @test_throws ArgumentError MtlSparseMatrixCSR(
                -1, 3, dev([1], Ti), dev(Ti[], Ti), dev(Float32[], Float32)
            )
        end

        @testset "index type overflow" begin
            m = Int64(typemax(Int32)) + 1
            A = spzeros(Float32, m, 1)
            @test_throws ArgumentError MtlSparseMatrixCSR(A)
        end

        @testset "show" begin
            A = testsparse(Float32, Int, 7, 9; seed = 7)
            dA = MtlSparseMatrixCSR(A)
            text = repr(MIME"text/plain"(), dA)
            @test occursin("7×9", text)
            @test occursin("MtlSparseMatrixCSR{Float32, Int32}", text)
            @test occursin("$(nnz(A)) stored", text)
            one_entry = MtlSparseMatrixCSR(sparse([1], [1], [1.0f0], 2, 2))
            @test occursin("1 stored entry", repr(one_entry))
        end

        @testset "nonzeros aliases device storage" begin
            dA = MtlSparseMatrixCSR(testsparse(Float32, Int, 8, 8; seed = 8))
            v = nonzeros(dA)
            @test v isa MtlVector{Float32}
            @test v === dA.nzval
        end
    else
        @info "MtlSparseMatrixCSR tests skipped: no functional Metal device"
    end
end
