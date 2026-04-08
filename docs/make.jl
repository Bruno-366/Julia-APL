using Documenter
using JuliaAPL
import JuliaAPL: ≡, ≢

makedocs(
    sitename = "JuliaAPL",
    modules  = [JuliaAPL],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    warnonly = [:missing_docs],
    pages    = [
        "Home"          => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo      = "github.com/Bruno-366/Julia-APL.git",
    branch    = "gh-pages",
    devbranch = "main",
)
