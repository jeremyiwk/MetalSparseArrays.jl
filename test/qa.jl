using Aqua: Aqua
using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports
using MetalSparseArrays: MetalSparseArrays, MtlVector, sparse
using SparseArrays: SparseMatrixCSC
using Test: @test, @testset

# Adapt's SparseMatrixCSC bridge and the device sparse assembly entry points
# follow CUSPARSE. The method allowlist prevents the sparse exception from
# admitting unrelated new overloads.
Aqua.test_all(
    MetalSparseArrays;
    piracies = (; treat_as_own = [SparseMatrixCSC, sparse]),
)
@testset "intentional sparse entry points" begin
    vectors = (MtlVector{Int32}, MtlVector{Int32}, MtlVector{Float32})
    allowed = Set(
        which(sparse, Tuple{vectors..., extra...}) for extra in
            ((), (Int,), (Int, Int), (Int, Int, typeof(+)))
    )
    actual = Set(m for m in methods(sparse) if m.module === MetalSparseArrays)
    @test actual == allowed
end
check_no_implicit_imports(MetalSparseArrays)
check_no_stale_explicit_imports(MetalSparseArrays)
