# The device pattern merge behind sparse-sparse broadcast. `SparseArrays`
# defines correct: for every operand pair the device result must equal
# `broadcast(op, A, B)` on the host *exactly* — same pattern, same index
# arrays, same stored values under `isequal` — because the merge runs the same
# arithmetic on the same values in the same order and drops the same computed
# zeros. No tolerance appears in this file, and none should: a tolerance here
# would hide a structural bug. `exact_equal` also revalidates what the
# unchecked construction inside the merge skips, since it compares the whole
# pointer and index arrays against the stdlib's.

@testset "merge broadcast kernel" begin
    if DEVICE_AVAILABLE
        # Operand pairs: every matrix of the suite's pattern corpus (random
        # densities, empty and degenerate shapes, a dense row and column,
        # stored zeros, nonfinite values) paired with a fresh random pattern
        # of the same shape and with itself — the self pair is what makes `-`
        # cancel every shared position. One extra pair interleaves the two
        # patterns so the two-pointer merge alternates sides at every step.
        function merge_pairs(::Type{Tv}) where {Tv}
            corpus = pattern_corpus(Tv)
            pairs = Tuple{SparseMatrixCSC{Tv, Int}, SparseMatrixCSC{Tv, Int}}[]
            for (k, A) in enumerate(corpus)
                push!(pairs, (A, testsparse(Tv, Int, size(A)...; density = 0.25, seed = 200 + k)))
                push!(pairs, (A, A))
            end
            push!(
                pairs, (
                    sparse(fill(1, 5), 1:2:9, fill(Tv(2), 5), 3, 10),
                    sparse(fill(1, 5), 2:2:10, fill(Tv(3), 5), 3, 10),
                )
            )
            return pairs
        end

        @testset "agrees with SparseArrays $F Tv=$Tv" for F in SPARSE_TYPES,
                Tv in ELEMENT_TYPES

            for (A, B) in merge_pairs(Tv), op in (+, -, *)
                dC = broadcast(op, F{Tv, Int32}(A), F{Tv, Int32}(B))
                @test dC isa F{Tv, Int32}
                @test exact_equal(broadcast(op, A, B), SparseMatrixCSC(dC))
            end
        end

        # One thread merges one slice, so the launch is only correct if slice
        # counts at, just below, and just above every threadgroup boundary the
        # launch configuration can pick are all handled — in both compressed
        # directions, because CSR merges over rows and CSC over columns.
        @testset "threadgroup boundary sizes $F" for F in SPARSE_TYPES
            Tv = Float32
            for m in (1, 2, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025)
                A = testsparse(Tv, Int, m, 4; density = 0.3, seed = 30)
                B = testsparse(Tv, Int, m, 4; density = 0.3, seed = 31)
                dC = broadcast(+, F{Tv, Int32}(A), F{Tv, Int32}(B))
                @test exact_equal(A .+ B, SparseMatrixCSC(dC))
            end
            for n in (255, 256, 257, 1023, 1024, 1025)
                A = testsparse(Tv, Int, 4, n; density = 0.3, seed = 32)
                B = testsparse(Tv, Int, 4, n; density = 0.3, seed = 33)
                dC = broadcast(-, F{Tv, Int32}(A), F{Tv, Int32}(B))
                @test exact_equal(A .- B, SparseMatrixCSC(dC))
            end
        end

        # A compressed merge result carries bound-sized buffers with the tail
        # unspecified (as SparseMatrixCSC allows); every operation must ignore
        # the tail, nnz must report the stored count, and copy must compact.
        @testset "merge result buffers are oversized and ignored $F" for F in
            (MtlSparseMatrixCSR, MtlSparseMatrixCSC)

            Tv = Float32
            A = testsparse(Tv, Int, 25, 25; density = 0.2, seed = 45)
            B = testsparse(Tv, Int, 25, 25; density = 0.2, seed = 46)
            dC = broadcast(+, F{Tv, Int32}(A), F{Tv, Int32}(B))
            host = A .+ B
            @test nnz(dC) == nnz(host)
            @test length(nonzeros(dC)) == nnz(A) + nnz(B)
            # Further operations on the oversized result stay exact.
            @test exact_equal(host .* Tv(2), SparseMatrixCSC(dC .* Tv(2)))
            @test exact_equal(host .- host, SparseMatrixCSC(dC .- dC))
            @test exact_equal(host, SparseMatrixCSC(MtlSparseMatrixCOO(dC)))
            compact = copy(dC)
            @test length(nonzeros(compact)) == nnz(host)
            @test exact_equal(host, SparseMatrixCSC(compact))
        end

        @testset "cancellation drops entries as SparseArrays does $F" for F in SPARSE_TYPES
            Tv = Float32
            A = testsparse(Tv, Int, 30, 30; density = 0.3, seed = 40)
            dA = F{Tv, Int32}(A)
            # Exact cancellation leaves nothing stored, not stored zeros.
            @test nnz(dA .- dA) == 0
            @test exact_equal(A .- A, SparseMatrixCSC(dA .- dA))
            # A product over disjoint patterns annihilates every union
            # position, structurally emptying the result.
            B = sparse(1:5, 1:5, fill(Tv(2), 5), 6, 6)
            C = sparse(1:4, 2:5, fill(Tv(3), 4), 6, 6)
            dP = F{Tv, Int32}(B) .* F{Tv, Int32}(C)
            @test nnz(dP) == 0
            @test exact_equal(B .* C, SparseMatrixCSC(dP))
        end

        # Scalars in the broadcast expression ride into the kernel inside the
        # MergeFunction; the flattened forms below place them before, between,
        # and around the sparse operands.
        @testset "scalars in the merged expression $F" for F in SPARSE_TYPES
            Tv = Float32
            A = testsparse(Tv, Int, 12, 9; density = 0.25, seed = 50)
            B = testsparse(Tv, Int, 12, 9; density = 0.25, seed = 51)
            dA, dB = F{Tv, Int32}(A), F{Tv, Int32}(B)
            two, three = Tv(2), Tv(3)
            for expr in (
                    (a, b) -> two .* a .+ b,
                    (a, b) -> a .* two .- b .* three,
                    (a, b) -> (a .+ b) .* two,
                )
                dC = expr(dA, dB)
                @test dC isa F{Tv, Int32}
                @test exact_equal(expr(A, B), SparseMatrixCSC(dC))
            end
        end

        @testset "mixed operand formats" begin
            Tv = Float32
            A = testsparse(Tv, Int, 14, 11; density = 0.25, seed = 60)
            B = testsparse(Tv, Int, 14, 11; density = 0.25, seed = 61)
            for FA in SPARSE_TYPES, FB in SPARSE_TYPES
                dC = broadcast(+, FA{Tv, Int32}(A), FB{Tv, Int32}(B))
                # The result takes the format of the first sparse operand.
                @test dC isa FA{Tv, Int32}
                @test exact_equal(A .+ B, SparseMatrixCSC(dC))
            end
        end

        @testset "mixed element types" begin
            A = testsparse(Float32, Int, 10, 10; density = 0.3, seed = 70)
            B = testsparse(Float16, Int, 10, 10; density = 0.3, seed = 71)
            dC = MtlSparseMatrixCSR{Float32, Int32}(A) .+
                MtlSparseMatrixCSR{Float16, Int32}(B)
            @test dC isa MtlSparseMatrixCSR{Float32, Int32}
            @test exact_equal(A .+ B, SparseMatrixCSC(dC))
        end

        @testset "deterministic $F" for F in SPARSE_TYPES
            Tv = Float32
            # No atomics and no accumulation: the same input must give a
            # bit-identical result on every call, including for the imbalanced
            # dense-row pattern where threads finish in unpredictable order.
            cases = [
                (
                    testsparse(Tv, Int, 64, 64; density = 0.3, seed = 80),
                    testsparse(Tv, Int, 64, 64; density = 0.3, seed = 81),
                ),
                (
                    sparse(fill(2, 50), 1:50, fill(Tv(3), 50), 50, 50),
                    sparse(fill(2, 50), 1:50, fill(Tv(7), 50), 50, 50),
                ),
            ]
            for (A, B) in cases
                dA, dB = F{Tv, Int32}(A), F{Tv, Int32}(B)
                first_result = SparseMatrixCSC(dA .+ dB)
                for _ in 1:4
                    @test exact_equal(first_result, SparseMatrixCSC(dA .+ dB))
                end
            end
        end

        # The CSR <-> COO conversion kernels launch one thread per row (plus
        # one for the closing pointer entry), so row counts at and around the
        # threadgroup boundaries need explicit coverage; the pattern corpus in
        # test_conversions.jl covers the structural cases at small sizes.
        @testset "conversion kernels at threadgroup boundary sizes" begin
            Tv = Float32
            for m in (1023, 1024, 1025), density in (0.0, 0.02)
                A = testsparse(Tv, Int, m, 7; density, seed = m)
                dR = MtlSparseMatrixCSR{Tv, Int32}(A)
                dC = MtlSparseMatrixCOO{Tv, Int32}(A)
                @test exact_equal(A, SparseMatrixCSC(MtlSparseMatrixCOO(dR)))
                @test exact_equal(A, SparseMatrixCSC(MtlSparseMatrixCSR(dC)))
            end
        end

        @testset "host fallback retained and reachable" begin
            Tv = Float32
            A = testsparse(Tv, Int, 9, 7; density = 0.3, seed = 90)
            B = testsparse(Tv, Int, 9, 7; density = 0.3, seed = 91)
            C = testsparse(Tv, Int, 9, 7; density = 0.3, seed = 92)
            dA = MtlSparseMatrixCSR{Tv, Int32}(A)
            dB = MtlSparseMatrixCSR{Tv, Int32}(B)
            dC = MtlSparseMatrixCSR{Tv, Int32}(C)

            # Three sparse operands are outside the merge's domain: it must
            # decline, and the fallback must still give the stdlib answer.
            @test MetalSparseArrays.try_merge_broadcast(+, (dA, dB, dC)) === nothing
            @test exact_equal(A .+ B .+ C, SparseMatrixCSC(dA .+ dB .+ dC))

            # A shape-expanding broadcast is likewise outside it.
            col = testsparse(Tv, Int, 9, 1; density = 0.5, seed = 93)
            dcol = MtlSparseMatrixCSR{Tv, Int32}(col)
            @test MetalSparseArrays.try_merge_broadcast(*, (dA, dcol)) === nothing
            @test exact_equal(A .* col, SparseMatrixCSC(dA .* dcol))

            # And the merge does take the two-operand equal-shape case.
            @test MetalSparseArrays.try_merge_broadcast(+, (dA, dB)) !== nothing
        end
    else
        @info "merge broadcast kernel tests skipped: no functional Metal device"
    end
end
