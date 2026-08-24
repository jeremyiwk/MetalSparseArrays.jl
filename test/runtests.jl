include("testsuite.jl")

@testset "MetalSparseArrays" begin
    include("test_common.jl")
    include("test_csr.jl")
end
