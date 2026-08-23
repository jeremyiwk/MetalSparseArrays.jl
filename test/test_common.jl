@testset "common" begin
    @testset "indextype" begin
        @test MetalSparseArrays.indextype(testsparse(Float32, Int32, 4, 4)) === Int32
        @test MetalSparseArrays.indextype(testsparse(Float32, Int, 4, 4)) === Int
        @test MetalSparseArrays.indextype(SparseMatrixCSC{Float32,Int32}) === Int32
    end

    @testset "realtype and unit_roundoff" begin
        @test MetalSparseArrays.realtype(ComplexF32) === Float32
        @test MetalSparseArrays.realtype(Complex{BFloat16}) === BFloat16
        for T in ELEMENT_TYPES
            u = MetalSparseArrays.unit_roundoff(T)
            @test u == eps(MetalSparseArrays.realtype(T)) / 2
            @test MetalSparseArrays.realtype(T)(1) + 2u > 1
        end
    end

    @testset "DEFAULT_INDEX_TYPE" begin
        @test MetalSparseArrays.DEFAULT_INDEX_TYPE === Int32
    end
end

@testset "element types" begin
    @test referencetype(Float16) === Float64
    @test referencetype(Complex{BFloat16}) === ComplexF64

    if DEVICE_AVAILABLE
        # Apple GPUs have no double precision unit, but they do support every
        # narrower floating point type, so the suite must cover all of them.
        @test Float64 ∉ ELEMENT_TYPES
        @test ComplexF64 ∉ ELEMENT_TYPES
        for T in (Float16, Float32, BFloat16, ComplexF16, ComplexF32)
            @test T ∈ ELEMENT_TYPES
        end
    end
    @test !isempty(ELEMENT_TYPES)
end

@testset "test problems" begin
    for Tv in ELEMENT_TYPES
        A = testsparse(Tv, Int32, 20, 12; density = 0.25)
        @test size(A) == (20, 12)
        @test eltype(A) === Tv
        @test 0 < nnz(A) <= 20 * 12
        entries = nonzeros(A)
        @test all(<=(one(MetalSparseArrays.realtype(Tv))), abs.(real.(entries)))
        @test all(<=(one(MetalSparseArrays.realtype(Tv))), abs.(imag.(entries)))
        @test A == testsparse(Tv, Int32, 20, 12; density = 0.25)
        @test rowvals(A) == rowvals(testsparse(referencetype(Tv), Int32, 20, 12;
                                               density = 0.25))

        n = 8
        L = laplacian_2d(Tv, n)
        @test size(L) == (n^2, n^2)
        @test L == transpose(L)
        @test nnz(L) == 5n^2 - 4n
        @test isposdef(Hermitian(Matrix{referencetype(Tv)}(L)))
    end
end
