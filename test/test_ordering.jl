@testset "device ordering" begin
    if DEVICE_AVAILABLE
        @testset "stable ties Ti=$Ti n=$n" for Ti in INDEX_TYPES,
                n in (0, 1, 31, 32, 33, 1023, 1024, 1025, 65536)
            rng = StableRNG(n)
            for idx in (rand(rng, Ti(1):Ti(97), n), fill(Ti(1), n))
                d = MtlVector(idx)
                d .+= one(Ti) # pending producer must precede the sort
                for _ in 1:2
                    perm = MetalSparseArrays.stable_index_sortperm(d, Ti(98))
                    @test Array(perm) == sortperm(idx)
                end
                @test Array(d) == idx .+ one(Ti)
            end
        end

        @testset "Int64 high bits and composed ordering" begin
            rng = StableRNG(42)
            for bound in (Int64(1) << 53, Int64(1) << 62, typemax(Int64))
                idx = rand(rng, [Int64(1), 2, bound - 1, bound], 1025)
                initial = randperm(rng, length(idx))
                d = MtlVector(idx)
                p = MetalSparseArrays.stable_index_sortperm(d, bound)
                @test Array(p) == sortperm(idx)
                p = MetalSparseArrays.stable_index_sortperm(d, bound, MtlVector(Int32.(initial)))
                @test Array(p) == initial[sortperm(idx[initial])]
            end
            @test_throws ArgumentError MetalSparseArrays.stable_index_sortperm(
                1:(Int(typemax(Int32)) + 1), typemax(Int32)
            )
        end

        @testset "reorder storage and launch boundaries Tv=$Tv Ti=$Ti" for Tv in ELEMENT_TYPES,
                Ti in INDEX_TYPES
            for n in (31, 33, 1025)
                A = SparseMatrixCSC{Tv, Ti}(testsparse(Tv, Int, n, 7; density = 0.7, seed = n))
                for F in SPARSE_TYPES, G in SPARSE_TYPES
                    F === G && continue
                    dA = F{Tv, Ti}(A)
                    if !(dA isa MtlSparseMatrixCOO)
                        indices = dA isa MtlSparseMatrixCSC ? dA.rowval : dA.colval
                        tail = MtlVector(vcat(Array(indices), [typemax(Ti)]))
                        if dA isa MtlSparseMatrixCSC
                            dA.rowval = tail
                        else
                            dA.colval = tail
                        end
                        dA.nzval = MtlVector(vcat(Array(dA.nzval), [Tv(NaN)]))
                    end
                    dB = G(dA)
                    @test exact_equal(A, SparseMatrixCSC(dB))
                    nonzeros(dA) .= zero(Tv)
                    @test exact_equal(A, SparseMatrixCSC(dB))
                    @test dB.nzval !== dA.nzval
                end
            end
        end

        @testset "huge Int64 row space without pointer allocation" begin
            m = typemax(Int64) - 1
            rows = Int64[1, m, 2, m - 1, m]
            cols = Int64[1, 1, 2, 2, 2]
            values = Float32[-0.0, NaN, Inf, -Inf, 3]
            dA = MtlSparseMatrixCSC(
                m, 2, MtlVector(Int64[1, 3, 6]), MtlVector(rows), MtlVector(values)
            )
            dB = MtlSparseMatrixCOO(dA)
            p = sortperm(eachindex(rows); by = k -> (rows[k], cols[k]))
            @test Array(dB.rowval) == rows[p]
            @test Array(dB.colval) == cols[p]
            @test isequal(Array(dB.nzval), values[p])
            dC = MtlSparseMatrixCSC(dB)
            @test Array(dC.colptr) == Int64[1, 3, 6]
            @test Array(dC.rowval) == rows
            @test isequal(Array(dC.nzval), values)
        end
    else
        @info "device ordering tests skipped: no functional Metal device"
    end
end
