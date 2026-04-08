# JuliaAPL

```@meta
CurrentModule = JuliaAPL
```

A Julia implementation of APL (A Programming Language) dyadic and monadic
functions/operators.

Each exported APL symbol is a regular Julia function defined with multiple
dispatch:
- **One argument** → *monadic* (prefix) form.
- **Two arguments** → *dyadic* (infix / function-call) form.

APL operators (higher-order functions) are prefixed with `apl_`.

## Installation

```julia
using Pkg
Pkg.develop(path = ".")
```

## Usage

```julia
using JuliaAPL
import JuliaAPL: ≡, ≢    # resolve ambiguity with Base.≡ (===)
```

## Contents

```@contents
Pages = ["api.md"]
Depth = 2
```
