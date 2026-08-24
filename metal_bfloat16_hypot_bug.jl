# Minimal reproduction: `hypot` over device BFloat16 arrays segfaults the Julia
# process; `abs` over Complex{BFloat16} (which calls hypot) fails kernel
# compilation with InvalidIRError. Pure Metal.jl — no other packages involved.
#
# Observed with: Julia 1.12.6, Metal.jl v1.10.3, GPUCompiler v2.2.3,
# BFloat16s v0.6.1, macOS 26.5, Apple Silicon.
#
# `Metal.BFloat16` is `BFloat16s.BFloat16` re-exported; device storage and
# arithmetic for it work (the baseline below passes). The failure is specific
# to `hypot`.
#
# Run:
#   julia --project=. metal_bfloat16_hypot_bug.jl          # catchable failure
#   julia --project=. metal_bfloat16_hypot_bug.jl crash    # SEGFAULTS the process
#
# Expected output of the default run:
#   baseline sqrt:  BFloat16[1.4140625]
#   baseline abs2:  BFloat16[5.0]
#   abs on Complex{BFloat16} FAILS: GPUCompiler.InvalidIRError
#
# The `crash` run dies with signal 11 inside LLVM's inline cost analysis
# (`CallAnalyzer::analyze`, reached from GPUCompiler's Metal optimization
# pipeline, GPUCompiler/src/metal.jl `optimize_module!`) — a fatal crash, not a
# catchable exception.

using Metal

const BF = Metal.BFloat16

# Baseline: BFloat16 device arrays work in general.
println("baseline sqrt:  ", Array(sqrt.(MtlVector{BF}([BF(2)]))))
println("baseline abs2:  ", Array(abs2.(MtlVector{Complex{BF}}([BF(1) + BF(2) * im]))))

# Catchable manifestation: abs(z) = hypot(re, im) fails to compile.
try
    Array(abs.(MtlVector{Complex{BF}}([BF(1) + BF(2) * im])))
    println("abs on Complex{BFloat16} WORKED — bug appears fixed")
catch e
    println("abs on Complex{BFloat16} FAILS: ", typeof(e))
end

# Fatal manifestation: hypot directly segfaults the process. Gated behind an
# argument because it cannot be caught.
if "crash" in ARGS
    println("about to segfault...")
    hypot.(MtlVector{BF}([BF(1)]), MtlVector{BF}([BF(2)]))
    println("hypot WORKED — bug appears fixed")
end
