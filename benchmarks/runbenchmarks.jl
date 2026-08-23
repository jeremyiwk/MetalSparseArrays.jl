using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = dirname(@__DIR__))
Pkg.instantiate()

include("benchmarks.jl")

if isempty(SUITE)
    println("Benchmark suite is empty; nothing to run.")
else
    tune!(SUITE)
    results = run(SUITE; verbose = true)
    show(stdout, MIME"text/plain"(), results)
    println()
end
