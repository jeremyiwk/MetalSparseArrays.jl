# MetalSparseArrays.jl

[![CI](https://github.com/jeremyiwk/MetalSparseArrays.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jeremyiwk/MetalSparseArrays.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://jeremyiwk.github.io/MetalSparseArrays.jl/dev/)
[![codecov](https://codecov.io/gh/jeremyiwk/MetalSparseArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jeremyiwk/MetalSparseArrays.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![ColPrac: Contributor's Guide](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Sparse arrays for the Apple Silicon GPU backend. Code written against `SparseArrays` on the CPU, or against `CUDA.CUSPARSE` on NVIDIA hardware, should run unchanged on Metal — same storage formats, same function names, same semantics — including the operator interface that iterative solvers such as `Krylov.jl` require.

Supported Julia versions: 1.10 (current LTS) and later. Device operations require Apple Silicon; the test suite runs anywhere.

Contributions follow the [ColPrac](https://github.com/SciML/ColPrac) contributor guide.
