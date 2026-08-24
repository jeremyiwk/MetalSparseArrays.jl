using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = dirname(@__DIR__))
Pkg.instantiate()

include("benchmarks.jl")

using Printf: @printf

# Repetitions per case; the minimum is reported. See the timing note at the
# top of benchmarks.jl. Device timings need many repetitions for the minimum
# to converge: command-buffer scheduling adds hundreds of microseconds of
# right-tailed noise per sample.
const REPS = 100

mintime(b::Benchmark) =
    b.device ? minimum((Metal.@timed b.thunk()).time for _ in 1:REPS) :
    minimum((Base.@elapsed b.thunk()) for _ in 1:REPS)

function runsuite(suite)
    group = ""
    for b in suite
        if b.group != group
            group = b.group
            println("\n== ", group, " ==")
        end
        b.thunk() # compile untimed
        b.device && Metal.synchronize()
        @printf("%12.1f us  %s\n", mintime(b) * 1.0e6, join(b.key, ", "))
    end
    return nothing
end

isempty(SUITE) ? println("Benchmark suite is empty; nothing to run.") : runsuite(SUITE)
