# Julia-APL

A Julia implementation of APL (A Programming Language) dyadic and monadic
functions/operators, leveraging Julia's support for unicode identifiers,
multiple dispatch, and anonymous functions.

## Overview

APL uses non-ASCII characters for most of its built-in functions.  Julia
allows unicode characters as variable and function names, making it possible
to implement APL's notation almost verbatim.  Both languages have excellent
support for array and matrix operations.

Each exported APL symbol is a regular Julia function:
- **One argument** → invokes the *monadic* (prefix) APL form.
- **Two arguments** → invokes the *dyadic* (infix/function-call) APL form.

APL *operators* (higher-order functions) are prefixed with `apl_`.

For a full description of every function and operator, see
**[docs/functions.md](docs/functions.md)**.

## Installation

```julia
using Pkg
Pkg.develop(path = ".")   # from the repository root
```

## Quick usage

```julia
using JuliaAPL
import JuliaAPL: ≡, ≢    # needed when Base.≡ (===) would otherwise conflict
```

## Exported symbols

### APL Functions

| Symbol | Monadic | Dyadic |
|--------|---------|--------|
| `⍳` | Index generator `[1..n]` | Index-of (returns `length+1` when not found) |
| `⍴` | Shape | Reshape with APL row-major cyclic fill |
| `⌹` | Matrix inverse | Matrix divide |
| `⍉` | Reverse axes (transpose for 2-D, no-op for 1-D) | Permute axes |
| `⍋` | Grade up (sort-permutation ascending) | — |
| `⍒` | Grade down (sort-permutation descending) | — |
| `⌽` | Reverse last axis | Rotate last axis (any number of dimensions) |
| `⊖` | Reverse first axis | Rotate first axis (any number of dimensions) |
| `∊` | Flatten in APL row-major order | Membership test (preserves shape) |
| `↑` | — | Take first/last `n` elements (zero-padded) |
| `↓` | — | Drop first/last `n` elements |
| `⊂` | Enclose (wrap in 1-element vector) | — |
| `⊃` | First element | Element at index `i` |
| `≡`* | Nesting depth | Structural match (`isequal`) |
| `≢`* | Tally (`size(A,1)`) | Structural not-match |
| `⍟` | Natural logarithm | Logarithm with base `b` |
| `⍲` | — | NAND (broadcasts over arrays) |
| `⍱` | — | NOR (broadcasts over arrays) |

> \* `≡` and `≢` also exist in `Base` — add `import JuliaAPL: ≡, ≢` after
> `using JuliaAPL` to resolve the ambiguity.

### APL Operators (higher-order functions)

| APL notation | Julia | Description |
|:-------------|:------|:------------|
| `f/A`   | `apl_reduce(f)(A)` | Left fold of `A` with `f` |
| `f\A`   | `apl_scan(f)(A)`   | Prefix scan of `A` with `f` |
| `f¨A`   | `apl_each(f)(A)`   | Apply `f` to each element of `A` |
| `A∘.fB` | `apl_outer(f)(A,B)` | Outer product |
| `Af.gB` | `apl_inner(f,g)(A,B)` | Generalised inner product |

### Infix syntax

Julia's parser recognises some APL symbols as binary operators:

```julia
[1,2,3,4] ∊ [2,4]    # [false,true,false,true]
1 ⊖ [1 2; 3 4; 5 6]  # rotate rows up by 1
3 ↑ [10,20,30,40]     # [10,20,30]
[1,2,3] ≡ [1,2,3]    # true
```

Symbols Julia does *not* accept as infix (`⍳`, `⍴`, `⌹`, `⍉`, `⌽`, `⍟`,
`⍲`, `⍱`) must be called with standard function-call syntax, e.g.
`⍳([1,2,3], 2)`.

## Running Tests

```julia
julia --project=. test/runtests.jl
```
