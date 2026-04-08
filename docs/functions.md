# JuliaAPL — Function & Operator Reference

This document provides a detailed reference for every function and operator
exported by `JuliaAPL`.

---

## Using the package

```julia
using JuliaAPL
import JuliaAPL: ≡, ≢    # resolve ambiguity with Base.≡ (===) and Base.≢
```

---

## APL Functions

### ⍳  Iota — index generator / index-of

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⍳n` | `⍳(n)` |
| Dyadic | `A⍳B` | `⍳(A, B)` |

**Monadic** `⍳(n)`: generate the integer vector `[1, 2, …, n]`.

**Dyadic** `⍳(A, b)`: return the 1-based position of the first occurrence of
`b` in the APL ravel (row-major) of `A`, or `length(A) + 1` if not present.

**Dyadic** `⍳(A, B)` where `B` is an array: apply element-wise, preserving
`B`'s shape.

```julia
⍳(5)                        # [1, 2, 3, 4, 5]
⍳([10,20,30], 20)           # 2  (1-based index)
⍳([10,20,30], 99)           # 4  (not found → length+1)
⍳([10,20,30], [20,30])      # [2, 3]

# matrix left-arg — uses APL row-major ravel; result preserves B's shape
⍳([10 20; 30 40], [40 10; 99 30])   # [4 1; 5 3]
```

---

### ⍴  Rho — shape / reshape

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⍴A` | `⍴(A)` |
| Dyadic | `s⍴A` | `⍴(s, A)` |

**Monadic** `⍴(A)`: return the shape (size along each axis) as an integer
vector.  Returns `Int[]` for a scalar.

**Dyadic** `⍴(shape, A)`: reshape `A` into the given shape using APL's
row-major cyclic fill.  `shape` can be a vector of integers or a single
integer.

```julia
⍴([1 2 3; 4 5 6])     # [2, 3]
⍴(42)                  # Int[]
⍴([2,3], 1:6)         # 2×3 matrix  [1 2 3; 4 5 6]  (row-major fill)
⍴([2,3], 0)           # 2×3 zeros
⍴([2,4], [1,2,3])     # 2×4 cyclic fill  [1 2 3 1; 2 3 1 2]

# reshaping a 2-D source (row-major ravel of source):
⍴(6, [1 2 3; 4 5 6])       # [1,2,3,4,5,6]
⍴([3,2], [1 2 3; 4 5 6])   # [1 2; 3 4; 5 6]
```

---

### ⌹  Domino — matrix inverse / matrix divide

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⌹M` | `⌹(M)` |
| Dyadic | `A⌹B` | `⌹(A, B)` |

**Monadic** `⌹(M)`: compute the inverse of square matrix `M`.

**Dyadic** `⌹(A, B)`: solve the linear system `A*x = B` (least-squares for
non-square `A`), equivalent to Julia's `A \ B`.

```julia
M = [1.0 2.0; 3.0 4.0]
⌹(M)                    # matrix inverse
⌹(M, [1.0, 2.0])       # solve M*x = b
```

---

### ⍉  Transpose — reverse axes / permute axes

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⍉A` | `⍉(A)` |
| Dyadic | `p⍉A` | `⍉(p, A)` |

**Monadic** `⍉(A)`: reverse the axis order.  For a 2-D matrix this is the
ordinary transpose.  For a 1-D vector it is a no-op (returns a copy).

**Dyadic** `⍉(perm, A)`: permute the axes of `A` according to `perm`.

```julia
⍉([1 2 3; 4 5 6])          # 3×2 transpose
⍉([1, 2, 3])               # [1, 2, 3]  (1-D no-op)
⍉([2,1,3], rand(2,3,4))    # permute axes of a 3-D array
```

---

### ⍋ / ⍒  Grade Up / Grade Down

| Symbol | APL | Julia | Description |
|--------|-----|-------|-------------|
| `⍋` | `⍋A` | `⍋(A)` | Sort-permutation (ascending) |
| `⍒` | `⍒A` | `⍒(A)` | Sort-permutation (descending) |

```julia
v = [3, 1, 4, 1, 5]
⍋(v)   # [2,4,1,3,5]  → v[⍋(v)] is sorted ascending
⍒(v)   # [5,3,1,2,4]  → v[⍒(v)] is sorted descending
```

---

### ⌽  Rotate/Reverse — last axis

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⌽A` | `⌽(A)` |
| Dyadic | `n⌽A` | `⌽(n, A)` |

**Monadic** `⌽(A)`: reverse elements along the last axis.

**Dyadic** `⌽(n, A)`: rotate elements along the last axis by `n` positions
(positive = left-rotate, negative = right-rotate).  Works for arrays of any
number of dimensions.

```julia
⌽([1,2,3,4,5])            # [5,4,3,2,1]
⌽(2, [1,2,3,4,5])         # [3,4,5,1,2]
⌽([1 2 3; 4 5 6])         # [3 2 1; 6 5 4]  (reverse each row)
⌽(1, [1 2 3; 4 5 6])      # [2 3 1; 5 6 4]
```

---

### ⊖  Rotate/Reverse — first axis

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⊖A` | `⊖(A)` |
| Dyadic | `n⊖A` | `n ⊖ A` (infix) |

**Monadic** `⊖(A)`: reverse elements along the first axis.

**Dyadic** `n ⊖ A`: rotate elements along the first axis by `n` positions
(positive = up-rotate, negative = down-rotate).  Works for any number of
dimensions.

```julia
⊖([1 2; 3 4; 5 6])    # [5 6; 3 4; 1 2]  (reverse rows)
1 ⊖ [1 2; 3 4; 5 6]   # [3 4; 5 6; 1 2]  (rotate rows up by 1)
```

---

### ∊  Epsilon — enlist / membership

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `∊A` | `∊(A)` |
| Dyadic | `A∊B` | `A ∊ B` (infix) |

**Monadic** `∊(A)`: flatten `A` into a 1-D vector in APL row-major order.

**Dyadic** `A ∊ B`: boolean array with the same shape as `A`, `true` where
an element of `A` appears in `B`.

```julia
∊([1 2; 3 4])               # [1,2,3,4]  (row-major flatten)
[1,2,3,4] ∊ [2,4]           # [false,true,false,true]
[1 2; 3 4] ∊ [2,4]          # Bool[false true; false true]  (shape preserved)
```

---

### ↑ / ↓  Take / Drop

> **Note**: `↑` and `↓` are infix in Julia. Use `(-n) ↑ v` (with parentheses)
> to avoid unary-minus precedence issues.

```julia
3 ↑ [10,20,30,40,50]       # [10,20,30]       (take first 3)
(-2) ↑ [10,20,30,40,50]    # [40,50]          (take last 2)
6 ↑ [1,2,3]                # [1,2,3,0,0,0]    (zero-padded)

2 ↓ [10,20,30,40,50]       # [30,40,50]       (drop first 2)
(-2) ↓ [10,20,30,40,50]    # [10,20,30]       (drop last 2)
```

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
> After `using JuliaAPL`, add `import JuliaAPL: ≡, ≢` to avoid ambiguity.

| Form | APL | Julia |
|------|-----|-------|
| Monadic `≡` | `≡A` | `≡(A)` — nesting depth |
| Dyadic `≡` | `A≡B` | `A ≡ B` (infix) — `isequal` match |
| Monadic `≢` | `≢A` | `≢(A)` — tally (`size(A,1)`) |
| Dyadic `≢` | `A≢B` | `A ≢ B` (infix) — `!isequal` |

```julia
≡(42)                # 0  (scalar / any atom)
≡("hello")           # 0  (any non-array is depth 0)
≡([1,2,3])           # 1  (simple array)
≡([[1,2],[3,4]])      # 2  (nested array)
[1,2,3] ≡ [1,2,3]   # true

≢([1,2,3,4])         # 4
≢([1 2; 3 4; 5 6])   # 3  (rows)
[1,2] ≢ [1,3]        # true
```

---

### ⍟  Log

| Form | APL | Julia |
|------|-----|-------|
| Monadic | `⍟x` | `⍟(x)` — natural log |
| Dyadic | `b⍟x` | `⍟(b, x)` — log base `b` |

```julia
⍟(exp(1))    # 1.0
⍟(1.0)       # 0.0
⍟(2, 8)      # 3.0  (log₂ 8)
⍟(10, 100)   # 2.0  (log₁₀ 100)
```

---

### ⍲ / ⍱  NAND / NOR

Scalar and broadcasting forms:

```julia
⍲(true, false)                        # true   (NAND)
⍱(false, false)                       # true   (NOR)
⍲([true,false], [true,true])          # [false, true]
⍱([false,false], [false,true])        # [true, false]
```

---

## APL Operators (Higher-order Functions)

APL operators accept functions and return new functions.

| APL notation | Julia | Description |
|:-------------|:------|:------------|
| `f/A`   | `apl_reduce(f)(A)` | Left fold of `A` with `f` |
| `f\A`   | `apl_scan(f)(A)`   | Prefix scan of `A` with `f` |
| `f¨A`   | `apl_each(f)(A)`   | Apply `f` to each element of `A` |
| `A∘.fB` | `apl_outer(f)(A,B)` | Outer product: `result[i,j] = f(A[i],B[j])` |
| `Af.gB` | `apl_inner(f,g)(A,B)` | Generalised inner product |

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
dot([1,2,3], [4,5,6])        # 32  (dot product)
dot([1 2; 3 4], [5 6; 7 8])  # standard matrix multiply

# Tropical (max-plus) algebra:
apl_inner(max, +)([1,2], [3,4])   # max(1+3, 2+4) = 6
```
