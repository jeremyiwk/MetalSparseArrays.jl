include("testsuite.jl")

@testset "MetalSparseArrays" begin
    include("test_common.jl")
    include("test_csr.jl")
    include("test_csc.jl")
    include("test_coo.jl")
    include("test_conversions.jl")
    include("test_dense.jl")
    include("test_interface.jl")
    include("test_broadcast.jl")
    include("test_kernels.jl")
end
