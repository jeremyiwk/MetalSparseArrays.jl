using Documenter
using MetalSparseArrays

makedocs(
    sitename = "MetalSparseArrays.jl",
    modules = [MetalSparseArrays],
    format = Documenter.HTML(
        canonical = "https://jeremyiwk.github.io/MetalSparseArrays.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Roadmap" => "roadmap.md",
        "API reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/jeremyiwk/MetalSparseArrays.jl.git",
    devbranch = "main",
    push_preview = false,
)
