@testset "COO assembly" begin
    if DEVICE_AVAILABLE
        @testset "reference Tv=$Tv Ti=$Ti fmt=$fmt" for Tv in ELEMENT_TYPES,
                Ti in INDEX_TYPES, fmt in (:csc, :csr, :coo)
            F = fmt === :csc ? MtlSparseMatrixCSC : fmt === :csr ? MtlSparseMatrixCSR : MtlSparseMatrixCOO
            for seed in 0:3, count in (0, 1, 31, 33, 1023, 1025)
                rng = StableRNG(seed + count)
                I = rand(rng, Ti(1):Ti(19), count)
                J = rand(rng, Ti(1):Ti(27), count)
                V = Tv.(uniform(rng, referencetype(Tv), count))
                seed % 2 == 0 && reverse!(I)
                di, dj, dv = MtlVector(I), MtlVector(J), MtlVector(V)
                A = sparse(di, dj, dv, 19, 27; fmt)
                @test A isa F{Tv, Ti}
                @test exact_equal(sparse(I, J, V, 19, 27), SparseMatrixCSC(A))
                @test Array(di) == I && Array(dj) == J && isequal(Array(dv), V)
            end
            for (m, n) in ((0, 0), (0, 7), (7, 0), (7, 9))
                A = sparse(MtlVector(Ti[]), MtlVector(Ti[]), MtlVector(Tv[]), m, n; fmt)
                @test exact_equal(spzeros(Tv, Ti, m, n), SparseMatrixCSC(A))
            end
            for count in (1, 33)
                I, J, V = ones(Ti, count), ones(Ti, count), ones(Tv, count)
                A = sparse(MtlVector(I), MtlVector(J), MtlVector(V), 1, 1; fmt)
                @test exact_equal(sparse(I, J, V, 1, 1), SparseMatrixCSC(A))
            end

            @testset "duplicates, zeros and original fold order" begin
                rng = StableRNG(9)
                p = randperm(rng, 2053)
                I, J = Ti.(mod1.(p, 3)), Ti.(mod1.(p, 2))
                V = Tv.(uniform(rng, referencetype(Tv), length(p)))
                for combine in (+, -)
                    A = sparse(MtlVector(I), MtlVector(J), MtlVector(V), 3, 2, combine; fmt)
                    @test exact_equal(sparse(I, J, V, 3, 2, combine), SparseMatrixCSC(A))
                end
                large = Tv(2 / MetalSparseArrays.unit_roundoff(Tv))
                for V in (
                        Tv[large, one(Tv), -large],
                        fill(one(Tv), 4097),
                        Tv[zero(Tv), -zero(Tv)],
                        Tv[Inf, -Inf, NaN],
                    )
                    I, J = fill(Ti(2), length(V)), fill(Ti(3), length(V))
                    for combine in (+, -)
                        expected = sparse(I, J, V, 4, 5, combine)
                        for _ in 1:2
                            A = sparse(MtlVector(I), MtlVector(J), MtlVector(V), 4, 5, combine; fmt)
                            @test nnz(A) == 1
                            @test exact_equal(expected, SparseMatrixCSC(A))
                        end
                    end
                end
                I, J, V = Ti[2, 1, 2, 1], Ti[3, 2, 3, 2], Tv[1, 2, -1, -2]
                A = sparse(MtlVector(I), MtlVector(J), MtlVector(V); fmt)
                @test exact_equal(sparse(I, J, V), SparseMatrixCSC(A))
                @test nnz(A) == 2
                A = sparse(MtlVector(I), MtlVector(J), MtlVector(V), 7; fmt, combine = -)
                @test exact_equal(sparse(I, J, V, 7, 3, -), SparseMatrixCSC(A))
            end
        end

        @testset "validation and Boolean default" begin
            I, J, V = MtlVector(Int32[1, 2]), MtlVector(Int32[2, 1]), MtlVector(Float32[3, 4])
            @test_throws ArgumentError sparse(I, J, V; fmt = :invalid)
            @test_throws ArgumentError sparse(I, J, MtlVector(Float32[1]), 2, 2)
            @test_throws ArgumentError sparse(I, MtlVector(Int32[1]), V, 2, 2)
            for (m, n) in ((-1, 2), (2, -1), (0, 2), (2, 0), (1, 2), (2, 1), (Int(typemax(Int32)) + 1, 2))
                @test_throws ArgumentError sparse(I, J, V, m, n)
            end
            @test_throws ArgumentError sparse(MtlVector(Int32[0, 2]), J, V, 2, 2)
            @test_throws ArgumentError sparse(I, MtlVector(Int32[-1, 1]), V, 2, 2)
            let coefficients = Float32[2]
                combine = (a, b) -> a + coefficients[1] * b
                @test_throws ArgumentError sparse(I, J, V, 2, 2, combine)
            end
            for fmt in (:csc, :csr, :coo)
                i, j, v = Int32[1, 1, 2], Int32[2, 2, 1], Bool[false, true, false]
                A = sparse(MtlVector(i), MtlVector(j), MtlVector(v); fmt)
                @test exact_equal(sparse(i, j, v), SparseMatrixCSC(A))
                duplicates = vcat(falses(1024), [true])
                A = sparse(MtlVector(ones(Int32, 1025)), MtlVector(ones(Int32, 1025)), MtlVector(duplicates), 1, 1; fmt)
                @test nnz(A) == 1 && only(Array(nonzeros(A)))
                A = sparse(MtlVector(Int64.(i)), MtlVector(j), MtlVector(Float32.(v)); fmt)
                @test MetalSparseArrays.indextype(A) === Int64
                @test exact_equal(sparse(Int64.(i), Int64.(j), Float32.(v)), SparseMatrixCSC(A))
            end
            dv = MtlVector(Float32[3, 4])
            A = sparse(I, J, dv, 2, 2)
            dv .= 0
            I .= 1
            J .= 1
            @test exact_equal(sparse(Int32[1, 2], Int32[2, 1], Float32[3, 4], 2, 2), SparseMatrixCSC(A))
            @test size(sparse(MtlVector(Int32[]), MtlVector(Int32[]), MtlVector(Float32[]))) == (0, 0)
        end

        @testset "large Int64 coordinates without packed-key overflow" begin
            m = Int64(1) << 62
            I, J, V = Int64[m, 1, m, 2], Int64[1, 2, 1, 2], Float32[1, 2, 3, 4]
            expected = SparseMatrixCSC(m, 2, Int64[1, 2, 4], Int64[m, 1, 2], Float32[4, 2, 4])
            for fmt in (:csc, :coo)
                A = sparse(MtlVector(I), MtlVector(J), MtlVector(V), m, 2; fmt)
                C = fmt === :csc ? A : MtlSparseMatrixCSC(A)
                @test exact_equal(expected, SparseMatrixCSC(C))
            end
        end
    else
        @info "COO assembly tests skipped: no functional Metal device"
    end
end
