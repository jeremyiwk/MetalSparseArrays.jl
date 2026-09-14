@testset "audit regressions" begin
    if DEVICE_AVAILABLE
        @testset "Int64 pointer scan and block boundaries" for Ti in INDEX_TYPES
            for n in (0, 1, 31, 32, 33, 1023, 1024, 1025, 1048577)
                counts = Ti.(mod.(1:n, 4))
                Ti === Int64 && n > 0 && (counts[1] = typemax(Int32))
                ptr = MtlVector{Ti}(undef, n + 1)
                MetalSparseArrays.ptr_scan!(ptr, MtlVector(counts), one(Ti))
                @test Array(ptr) == vcat(one(Ti), one(Ti) .+ cumsum(counts))
            end
        end

        @testset "self broadcast compacts numerical zeros" for F in SPARSE_TYPES,
                Ti in INDEX_TYPES, Tv in ELEMENT_TYPES
            A = sparse([1, 2, 3], [1, 2, 2], Tv[0, 2, NaN], 3, 3)
            for f in (a -> (a .*= Tv(2)), a -> (a .*= zero(Tv)), a -> (a .= .-a), a -> (a .= zero(Tv)))
                reference = copy(A)
                dA = F{Tv, Ti}(A)
                f(reference)
                f(dA)
                @test exact_equal(reference, SparseMatrixCSC(dA))
            end
            dA = F{Tv, Ti}(A)
            dA .= dA
            @test exact_equal(A, SparseMatrixCSC(dA))
        end

        @testset "unused host storage tails" for F in SPARSE_TYPES, Ti in INDEX_TYPES
            A = sparse([1], [1], [2.0f0], 2, 2)
            push!(A.rowval, typemax(Int))
            push!(A.nzval, NaN32)
            dA = F{Float32, Ti}(A)
            @test nnz(dA) == 1
            @test exact_equal(SparseMatrixCSC(dA), sparse([1], [1], [2.0f0], 2, 2))
        end

        @testset "wrapped dense assignment" for F in SPARSE_TYPES, Ti in INDEX_TYPES
            D = reshape(Float32.(1:16), 4, 4)
            dD = MtlArray(D)
            for (rhs, reference) in (
                    (view(dD, 1:2:4, 1:2:4), view(D, 1:2:4, 1:2:4)),
                    (transpose(dD), transpose(D)),
                    (adjoint(dD), adjoint(D)),
                )
                A = spzeros(Float32, size(reference)...)
                dA = F{Float32, Ti}(A)
                A .= reference
                dA .= rhs
                @test exact_equal(A, SparseMatrixCSC(dA))
            end
        end

        @testset "dimension overflow without huge pointers" begin
            n = Int64(typemax(Int32)) + 1
            @test_throws ArgumentError MetalSparseArrays.dims_check(1, n, Int32)
            A = SparseMatrixCSC(n, 1, Int64[1, 2], Int64[1], Float32[2])
            dA = MtlSparseMatrixCSC{Float32, Int64}(A)
            @test size(dA) == (n, 1)
            @test nnz(dA) == 1
            @test Array(nonzeros(dA)) == Float32[2]
        end

        @testset "assignment converts promoted host values" for F in SPARSE_TYPES
            A = sparse([1, 2], [1, 2], Float32[0, 2], 2, 2)
            dA = F(A)
            A .*= 2.0
            dA .*= 2.0
            @test exact_equal(A, SparseMatrixCSC(dA))
        end
    end
end
