# Julia-APL

A Julia implementation of APL (A Programming Language) dyadic and monadic
functions/operators, leveraging Julia's support for unicode identifiers,
multiple dispatch, and anonymous functions.

## Overview

APL uses non-ASCII characters for most of its built-in functions.  Julia
allows unicode characters as variable and function names, making it possible
to implement APL's notation almost verbatim.  Both languages have excellent
support for array and matrix operations.

In this implementation:
- Each APL symbol is a Julia function defined with **multiple dispatch**.
  Calling it with one argument uses the **monadic** (prefix) form; calling it
  with two arguments uses the **dyadic** (infix) form.
- Julia allows many APL symbols to be used as **infix operators** (e.g.
  `A ∊ B`, `n ↑ v`, `A ≡ B`).  Where a symbol cannot be used as an infix
  operator it must be called in the usual Julia function-call syntax
  (e.g. `⍳(A, b)`, `⍴([2,3], data)`, `⌹(M, b)`).
- APL *operators* (higher-order functions) are implemented as ordinary Julia
  functions whose names are prefixed with `apl_`.

## Installation

```julia
using Pkg
Pkg.develop(path = ".")   # from the repository root
```

## Usage

```julia
using JuliaAPL
import JuliaAPL: ≡, ≢    # needed when Base.≡ (===) would otherwise conflict
```

---

## APL Functions

### ⍳  Iota — index generator / index-of

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⍳n` | `⍳(n)` |
| Dyadic | `A⍳B` | `⍳(A, B)` |

```julia
⍳(5)                        # [1, 2, 3, 4, 5]
⍳([10,20,30], 20)           # 2  (1-based index)
⍳([10,20,30], 99)           # 4  (not found → length+1)
⍳([10,20,30], [20,30])      # [2, 3]
```

---

### ⍴  Rho — shape / reshape

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⍴A` | `⍴(A)` |
| Dyadic | `s⍴A` | `⍴(s, A)` |

```julia
⍴([1 2 3; 4 5 6])     # [2, 3]
⍴([2,3], 1:6)         # 2×3 matrix  [1 2 3; 4 5 6]  (row-major fill)
⍴([2,3], 0)           # 2×3 matrix of zeros
⍴([2,4], [1,2,3])     # 2×4 matrix  [1 2 3 1; 2 3 1 2]  (cyclic fill)
```

---

### ⌹  Domino — matrix inverse / matrix divide

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⌹M` | `⌹(M)` |
| Dyadic | `A⌹B` | `⌹(A, B)` |

```julia
M = [1.0 2.0; 3.0 4.0]
⌹(M)           # matrix inverse
⌹(M, [1.0, 2.0])   # solve M*x = b  (least-squares when non-square)
```

---

### ⍉  Transpose — reverse axes / permute axes

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⍉A` | `⍉(A)` |
| Dyadic | `p⍉A` | `⍉(p, A)` |

```julia
⍉([1 2 3; 4 5 6])         # 3×2 transpose
⍉([2,1,3], rand(2,3,4))   # permute axes
```

---

### ⍋ / ⍒  Grade Up / Grade Down

```julia
v = [3, 1, 4, 1, 5]
⍋(v)   # sortperm ascending  → v[⍋(v)] is sorted ascending
⍒(v)   # sortperm descending → v[⍒(v)] is sorted descending
```

---

### ⌽  Rotate/Reverse — last axis

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⌽A` | `⌽(A)` |
| Dyadic | `n⌽A` | `⌽(n, A)` |

```julia
⌽([1,2,3,4,5])     # [5,4,3,2,1]  (reverse)
⌽(2, [1,2,3,4,5])  # [3,4,5,1,2]  (left-rotate by 2)
```

---

### ⊖  Rotate/Reverse — first axis

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⊖A` | `⊖(A)` |
| Dyadic | `n⊖A` | `n ⊖ A` (infix) |

```julia
⊖([1 2; 3 4; 5 6])   # [5 6; 3 4; 1 2]  (reverse rows)
1 ⊖ [1 2; 3 4; 5 6]  # [3 4; 5 6; 1 2]  (rotate rows up by 1)
```

---

### ∊  Epsilon — enlist / membership

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `∊A` | `∊(A)` |
| Dyadic | `A∊B` | `A ∊ B` (infix) |

```julia
∊([1 2; 3 4])          # [1,2,3,4]  (row-major flatten)
[1,2,3,4] ∊ [2,4]      # [false,true,false,true]
```

---

### ↑ / ↓  Take / Drop

```julia
3 ↑ [10,20,30,40,50]       # [10,20,30]  (take first 3)
(-2) ↑ [10,20,30,40,50]    # [40,50]     (take last 2; use parentheses for negative)
2 ↓ [10,20,30,40,50]       # [30,40,50]  (drop first 2)
(-2) ↓ [10,20,30,40,50]    # [10,20,30]  (drop last 2)
```

> **Note**: Because `↑` and `↓` are parsed as infix operators in Julia,
> write `(-n) ↑ v` rather than `-n ↑ v` to avoid unary-minus precedence
> issues.

---

### ⊂ / ⊃  Enclose / First

```julia
⊂([1,2,3])       # [[1,2,3]]  (wrap in a 1-element vector)
⊃([10,20,30])    # 10         (first element)
2 ⊃ [10,20,30]   # 20         (element at index 2)
```

---

### ≡ / ≢  Depth–Match / Tally–Not-Match

> **Important**: `≡` and `≢` also exist in `Base` (`===` and `!==`).
> To use the APL versions unambiguously, add
> `import JuliaAPL: ≡, ≢` after `using JuliaAPL`.

```julia
≡([1,2,3])              # 1   (depth: simple array)
≡([[1,2],[3,4]])         # 2   (depth: nested array)
[1,2,3] ≡ [1,2,3]       # true  (structural match)

≢([1,2,3,4])            # 4   (tally: length of first axis)
[1,2] ≢ [1,3]           # true  (not-match)
```

---

### ⍟  Log

```julia
⍟(exp(1))    # 1.0     (natural logarithm, monadic)
⍟(2, 8)      # 3.0     (log base 2 of 8, dyadic; function-call syntax)
```

---

### ⍲ / ⍱  NAND / NOR

```julia
⍲(true, false)             # true
⍱(false, false)            # true
⍲([true,false], [true,true])   # [false, true]  (broadcasts)
```

---

## APL Operators (Higher-order Functions)

| APL notation | Julia | Description |
|:-------------|:------|:------------|
| `f/A`  | `apl_reduce(f)(A)` | Reduce (fold) `A` left-to-right with `f` |
| `f\A`  | `apl_scan(f)(A)`   | Prefix scan of `A` with `f`             |
| `f¨A`  | `apl_each(f)(A)`   | Apply `f` to each element of `A`        |
| `A∘.fB` | `apl_outer(f)(A,B)` | Outer product: `result[i,j] = f(A[i], B[j])` |
| `Af.gB` | `apl_inner(f,g)(A,B)` | Generalised inner product          |

### apl_reduce

```julia
apl_reduce(+)([1,2,3,4])      # 10
apl_reduce(*)([1,2,3,4])      # 24
apl_reduce(max)([3,1,4,1,5])  # 5
```

### apl_scan

```julia
apl_scan(+)([1,2,3,4])    # [1, 3, 6, 10]  (running sum)
apl_scan(*)([1,2,3,4])    # [1, 2, 6, 24]  (running product)
apl_scan(max)([3,1,4,2])  # [3, 3, 4, 4]
```

### apl_each

```julia
apl_each(⍳)([3, 4, 5])
# [[1,2,3], [1,2,3,4], [1,2,3,4,5]]

apl_each(sum)([[1,2], [3,4,5]])   # [3, 12]
```

### apl_outer

```julia
apl_outer(*)(1:3, 1:3)
# 3×3 multiplication table:
# 1 2 3
# 2 4 6
# 3 6 9

apl_outer(==)(1:3, 1:3)   # 3×3 boolean identity matrix
```

### apl_inner

The classic matrix multiply is `apl_inner(+, *)`:

```julia
dot = apl_inner(+, *)
dot([1,2,3], [4,5,6])     # 32   (dot product)
dot([1 2; 3 4], [5 6; 7 8])  # standard matrix multiply

# Tropical (max-plus) algebra inner product:
apl_inner(max, +)([1,2], [3,4])   # max(1+3, 2+4) = 6
```

## Running Tests

```julia
julia --project=. test/runtests.jl
```
