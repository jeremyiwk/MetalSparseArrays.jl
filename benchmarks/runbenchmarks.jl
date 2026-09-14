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

function runbatch(b::Benchmark)
    for _ in 1:b.evals
        b.thunk()
    end
    return nothing
end

function sampletimes(b::Benchmark)
    times = b.device ? [(Metal.@timed runbatch(b)).time / b.evals for _ in 1:REPS] :
        [(Base.@elapsed runbatch(b)) / b.evals for _ in 1:REPS]
    sort!(times)
    return times[1], (times[REPS ÷ 2] + times[REPS ÷ 2 + 1]) / 2, times[cld(95REPS, 100)]
end

function runsuite(suite)
    Metal.versioninfo()
    println("minimum / median / p95, synchronized seconds; samples=", REPS)
    group = ""
    for b in suite
        if b.group != group
            group = b.group
            println("\n== ", group, " ==")
        end
        b.thunk() # compile untimed
        b.device && Metal.synchronize()
        low, median, p95 = sampletimes(b)
        @printf("%.9g\t%.9g\t%.9g\t%s\t%s\t%d\n", low, median, p95, b.group, join(b.key, '\t'), b.evals)
    end
    return nothing
end

selected = isempty(ARGS) ? SUITE : filter(b -> b.group in ARGS, SUITE)
isempty(selected) ? println("No matching benchmarks.") : runsuite(selected)
