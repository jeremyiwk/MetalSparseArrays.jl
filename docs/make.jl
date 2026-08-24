using Documenter
using MetalSparseArrays

makedocs(
    sitename = "MetalSparseArrays.jl",
    modules = [MetalSparseArrays],
    format = Documenter.HTML(
        canonical = "https://jeremyiwk.github.io/MetalSparseArrays.jl",
        edit_link = "master",
    ),
    pages = [
        "Home" => "index.md",
        "Roadmap" => "roadmap.md",
        "API reference" => "api.md",
        "References" => "references.md",
        "Changelog" => "changelog.md",
    ],
)

deploydocs(
    repo = "github.com/jeremyiwk/MetalSparseArrays.jl.git",
    devbranch = "master",
    push_preview = false,
)
