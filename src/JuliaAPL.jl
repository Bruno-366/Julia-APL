"""
    JuliaAPL

A Julia implementation of APL (A Programming Language) dyadic and monadic
functions/operators, leveraging Julia's support for unicode identifiers,
multiple dispatch, and anonymous functions.

# APL Functions

Each APL symbol is a Julia function defined with multiple dispatch:
- Calling with **one argument** invokes the *monadic* (prefix) form.
- Calling with **two arguments** invokes the *dyadic* (infix/function-call) form.

Some APL symbols are also recognised by Julia's parser as binary operators,
allowing natural infix syntax (`A ∊ B`, `n ↑ v`, `A ≡ B`).  Others must
always be called as regular functions (`⍳(A,b)`, `⍴(s,A)`, `⌹(M,b)`, etc.).

## Exported functions
`⍳` `⍴` `⌹` `⍉` `⍋` `⍒` `⌽` `⊖` `∊` `↑` `↓` `⊂` `⊃` `≡` `≢` `⍟` `⍲` `⍱`

## Exported operators (higher-order functions)
`apl_reduce`  `apl_scan`  `apl_each`  `apl_outer`  `apl_inner`

## Note on `≡` and `≢`
`≡` (`===`) and `≢` exist in `Base`.  After `using JuliaAPL`, add
`import JuliaAPL: ≡, ≢` to use the APL versions without ambiguity.
"""
module JuliaAPL

using LinearAlgebra

export ⍳, ⍴, ⌹, ⍉, ⍋, ⍒, ⌽, ⊖, ∊, ↑, ↓, ⊂, ⊃, ≡, ≢, ⍟, ⍲, ⍱
export apl_reduce, apl_scan, apl_each, apl_outer, apl_inner

# ============================================================
# ⍳  Iota  —  index generator (monadic) / index-of (dyadic)
# ============================================================

"""
    ⍳(n) -> Vector{Int}
    ⍳(A, b) -> Int
    ⍳(A, B) -> Array

**Monadic** `⍳n`: generate the integer vector `[1, 2, …, n]`.

**Dyadic** `⍳(A, b)`: return the 1-based position of the first occurrence of
`b` in the APL ravel (row-major) of `A`, or `length(A) + 1` if not present.

**Dyadic** `⍳(A, B)`: apply element-wise to `B`, preserving `B`'s shape.

# Examples
```julia
⍳(5)                       # [1, 2, 3, 4, 5]
⍳([10,20,30], 20)          # 2  (1-based position)
⍳([10,20,30], 99)          # 4  (not found → length+1)
⍳([10,20,30], [20,30])     # [2, 3]
```
"""
⍳(n::Integer) = collect(1:n)
⍳(A::AbstractArray, b) =
    let rv = _apl_ravel(A)
        idx = findfirst(==(b), rv)
        isnothing(idx) ? length(rv) + 1 : idx
    end
⍳(A::AbstractArray, B::AbstractArray) = map(b -> ⍳(A, b), B)

# ============================================================
# ⍴  Rho  —  shape (monadic) / reshape (dyadic)
# ============================================================

"""
    ⍴(A) -> Vector{Int}
    ⍴(shape, A) -> Array

**Monadic** `⍴A`: return the shape (size along each dimension) of `A` as an
integer vector.  Returns `Int[]` for a scalar.

**Dyadic** `shape ⍴ A`: reshape `A` (or fill with `A` if scalar) into the
given shape.  `shape` may be a vector of integers or a single integer.

# Examples
```julia
⍴([1 2 3; 4 5 6])          # [2, 3]
⍴(42)                       # Int[]
[2,3] ⍴ (1:6)              # 2×3 matrix
[2,3] ⍴ 0                   # 2×3 matrix of zeros
3 ⍴ [1,2,3,4,5]             # [1,2,3]
```
"""
⍴(A::AbstractArray) = collect(size(A))
⍴(A::Number) = Int[]
⍴(shape::AbstractVector{<:Integer}, A::AbstractArray) = _apl_reshape(_apl_ravel(A), shape)
⍴(shape::AbstractVector{<:Integer}, A::Number) = fill(A, shape...)
⍴(n::Integer, A::AbstractArray) =
    collect(Iterators.take(Iterators.cycle(_apl_ravel(A)), n))
⍴(n::Integer, A::Number) = fill(A, n)

# Internal: ravel an array in APL's row-major (last-axis-varies-fastest) order.
_apl_ravel(A::AbstractVector) = collect(A)
_apl_ravel(A::AbstractArray) = vec(permutedims(A, reverse(1:ndims(A))))

# Internal: reshape data (cycling as needed) into the given shape using
# APL's row-major (last-axis-varies-fastest) fill order.
function _apl_reshape(data::AbstractVector, shape::AbstractVector{<:Integer})
    N = length(shape)
    isempty(data) && throw(ArgumentError("cannot reshape empty data"))
    N == 0 && return first(data)
    n    = prod(shape)
    flat = collect(Iterators.take(Iterators.cycle(data), n))
    N == 1 && return flat
    # Julia reshape is column-major; APL fill is row-major.
    # Trick: reshape into reversed-shape, then permute axes back.
    tmp = reshape(flat, reverse(shape)...)
    return permutedims(tmp, reverse(1:N))
end

# ============================================================
# ⌹  Domino  —  matrix inverse (monadic) / matrix divide (dyadic)
# ============================================================

"""
    ⌹(A) -> Matrix
    ⌹(A, B) -> Array

**Monadic** `⌹A`: compute the inverse of square matrix `A`.

**Dyadic** `A ⌹ B`: solve the linear system / compute the least-squares
solution, equivalent to Julia's `A \\ B`.

# Examples
```julia
M = [1.0 2.0; 3.0 4.0]
⌹(M)          # matrix inverse
M ⌹ [1.0, 2.0]  # solve M*x = [1,2]
```
"""
⌹(A::AbstractMatrix) = inv(A)
⌹(A::AbstractMatrix, B::AbstractVecOrMat) = A \ B

# ============================================================
# ⍉  Transpose  —  reverse axes (monadic) / permute axes (dyadic)
# ============================================================

"""
    ⍉(A) -> Array
    ⍉(perm, A) -> Array

**Monadic** `⍉A`: reverse the axis order.  For a 2-D matrix this is the
ordinary transpose.  For a 1-D vector reversing axis order is a no-op and
returns a copy.

**Dyadic** `⍉(perm, A)`: permute the axes of `A` according to `perm`.

# Examples
```julia
⍉([1 2 3; 4 5 6])              # 3×2 transpose
⍉([1, 2, 3])                   # [1, 2, 3]  (1-D no-op)
⍉([2,1,3], rand(2,3,4))        # permute axes of a 3-D array
```
"""
⍉(A::AbstractVector) = copy(A)   # reversing axis order of a 1-D array is a no-op
⍉(A::AbstractMatrix) = collect(transpose(A))
⍉(A::AbstractArray) = permutedims(A, reverse(1:ndims(A)))
⍉(perm::AbstractVector{<:Integer}, A::AbstractArray) = permutedims(A, perm)

# ============================================================
# ⍋  Grade Up  /  ⍒  Grade Down
# ============================================================

"""
    ⍋(A) -> Vector{Int}

Return the permutation vector that sorts `A` in **ascending** order (1-based).

# Examples
```julia
⍋([3,1,4,1,5])  # [2,4,1,3,5]
```
"""
⍋(A::AbstractVector) = sortperm(A)

"""
    ⍒(A) -> Vector{Int}

Return the permutation vector that sorts `A` in **descending** order (1-based).

# Examples
```julia
⍒([3,1,4,1,5])  # [5,3,1,2,4]
```
"""
⍒(A::AbstractVector) = sortperm(A, rev=true)

# ============================================================
# ⌽  Rotate/Reverse (last axis)
# ============================================================

"""
    ⌽(A) -> Array
    ⌽(n, A) -> Array

**Monadic** `⌽A`: reverse elements along the **last** axis.

**Dyadic** `n ⌽ A`: rotate elements along the last axis by `n` positions
(positive = left-rotate, negative = right-rotate).

# Examples
```julia
⌽([1,2,3,4])        # [4,3,2,1]
⌽([1 2 3; 4 5 6])   # [3 2 1; 6 5 4]
2 ⌽ [1,2,3,4,5]     # [3,4,5,1,2]
```
"""
⌽(A::AbstractVector) = reverse(A)
⌽(A::AbstractMatrix) = reverse(A, dims=2)
⌽(A::AbstractArray) = reverse(A, dims=ndims(A))
⌽(n::Integer, A::AbstractVector) = circshift(A, -n)
⌽(n::Integer, A::AbstractMatrix) = circshift(A, (0, -n))
⌽(n::Integer, A::AbstractArray) =
    circshift(A, ntuple(d -> d == ndims(A) ? -n : 0, ndims(A)))

# ============================================================
# ⊖  Rotate/Reverse (first axis)
# ============================================================

"""
    ⊖(A) -> Array
    ⊖(n, A) -> Array

**Monadic** `⊖A`: reverse elements along the **first** axis.

**Dyadic** `n ⊖ A`: rotate elements along the first axis by `n` positions
(positive = up-rotate, negative = down-rotate).

# Examples
```julia
⊖([1,2,3,4])        # [4,3,2,1]
⊖([1 2; 3 4; 5 6])  # [5 6; 3 4; 1 2]
1 ⊖ [1 2; 3 4; 5 6] # [3 4; 5 6; 1 2]
```
"""
⊖(A::AbstractArray) = reverse(A, dims=1)
⊖(n::Integer, A::AbstractVector) = circshift(A, -n)
⊖(n::Integer, A::AbstractMatrix) = circshift(A, (-n, 0))
⊖(n::Integer, A::AbstractArray) =
    circshift(A, ntuple(d -> d == 1 ? -n : 0, ndims(A)))

# ============================================================
# ∊  Epsilon  —  enlist/flatten (monadic) / membership (dyadic)
# ============================================================

"""
    ∊(A) -> Vector
    ∊(A, B) -> Array{Bool}
    A ∊ B  -> Array{Bool}

**Monadic** `∊A`: flatten `A` into a 1-D vector in APL row-major order.

**Dyadic** `A ∊ B`: return a boolean array with the same shape as `A`, with
`true` at each position where the corresponding element of `A` appears in `B`.

# Examples
```julia
∊([1 2; 3 4])               # [1,2,3,4]  (row-major flatten)
[1,2,3,4] ∊ [2,4]           # [false,true,false,true]
[1 2; 3 4] ∊ [2,4]          # Bool[false true; false true]  (preserves shape)
```
"""
∊(A::AbstractArray) = _apl_ravel(A)
∊(A::AbstractArray, B::AbstractArray) = in.(A, Ref(B))

# ============================================================
# ↑  Take  /  ↓  Drop
# ============================================================

"""
    ↑(n, A) -> Array

Take `n` elements from `A`:
- Positive `n`: take from the **beginning**.
- Negative `n`: take from the **end**.

If `|n|` exceeds the length, the result is padded with the zero of
`eltype(A)`.

# Examples
```julia
3 ↑ [1,2,3,4,5]   # [1,2,3]
-2 ↑ [1,2,3,4,5]  # [4,5]
6 ↑ [1,2,3]       # [1,2,3,0,0,0]
```
"""
function ↑(n::Integer, A::AbstractVector)
    len = length(A)
    if n >= 0
        n <= len ? A[1:n] : vcat(A, fill(zero(eltype(A)), n - len))
    else
        k = -n
        k <= len ? A[end-k+1:end] : vcat(fill(zero(eltype(A)), k - len), A)
    end
end

function ↑(n::Integer, A::AbstractMatrix)
    rows, cols = size(A)
    if n >= 0
        n <= rows ? A[1:n, :] :
            vcat(A, fill(zero(eltype(A)), n - rows, cols))
    else
        k = -n
        k <= rows ? A[end-k+1:end, :] :
            vcat(fill(zero(eltype(A)), k - rows, cols), A)
    end
end

"""
    ↓(n, A) -> Array

Drop `n` elements from `A`:
- Positive `n`: drop from the **beginning**.
- Negative `n`: drop from the **end**.

# Examples
```julia
2 ↓ [1,2,3,4,5]   # [3,4,5]
-2 ↓ [1,2,3,4,5]  # [1,2,3]
```
"""
function ↓(n::Integer, A::AbstractVector)
    len = length(A)
    if n >= 0
        n >= len ? eltype(A)[] : A[n+1:end]
    else
        k = -n
        k >= len ? eltype(A)[] : A[1:end-k]
    end
end

function ↓(n::Integer, A::AbstractMatrix)
    rows = size(A, 1)
    if n >= 0
        n >= rows ? A[1:0, :] : A[n+1:end, :]
    else
        k = -n
        k >= rows ? A[1:0, :] : A[1:end-k, :]
    end
end

# ============================================================
# ⊂  Enclose  /  ⊃  First (Disclose)
# ============================================================

"""
    ⊂(A) -> Vector{typeof(A)}

**Enclose**: wrap `A` in a one-element vector, creating a nested scalar.

# Examples
```julia
⊂([1,2,3])  # [[1,2,3]]
```
"""
⊂(A) = [A]

"""
    ⊃(A) -> element
    ⊃(i, A) -> element

**Monadic** `⊃A`: extract the first element of `A` (disclose).

**Dyadic** `i ⊃ A`: extract element at index `i` (1-based scalar index).

# Examples
```julia
⊃([10,20,30])   # 10
2 ⊃ [10,20,30]  # 20
```
"""
⊃(A::AbstractArray) = first(A)
⊃(i::Integer, A::AbstractArray) = A[i]

# ============================================================
# ≡  Depth / Match  /  ≢  Tally / Not-match
# ============================================================

"""
    ≡(A) -> Int
    ≡(A, B) -> Bool

**Monadic** `≡A`: return the nesting depth of `A`.  A simple (non-nested)
array has depth 1; a scalar or any non-array atom has depth 0.

**Dyadic** `A ≡ B`: test structural equality using `isequal` (same as Julia's
`isequal`; treats `NaN == NaN` and `missing == missing` as true).

# Examples
```julia
≡([1,2,3])                  # 1
≡([[1,2],[3,4]])             # 2
≡("hello")                  # 0  (any non-array is depth 0)
[1,2,3] ≡ [1,2,3]           # true
[1,2,3] ≡ [1,2,4]           # false
```
"""
≡(A::AbstractArray) = _depth(A)
≡(A) = 0
≡(A, B) = isequal(A, B)

_depth(::Any) = 0
_depth(A::AbstractArray) =
    isempty(A) ? 1 : 1 + maximum(_depth(a) for a in A)

"""
    ≢(A) -> Int
    ≢(A, B) -> Bool

**Monadic** `≢A`: tally — return the length of the first dimension of `A`
(equivalent to `size(A,1)`).  Returns 1 for a scalar.

**Dyadic** `A ≢ B`: test structural non-equality using `isequal` (negation of
`≡`).

# Examples
```julia
≢([1,2,3,4])          # 4
≢([1 2; 3 4; 5 6])    # 3
[1,2] ≢ [1,3]         # true
```
"""
≢(A::AbstractArray) = size(A, 1)
≢(A::Number) = 1
≢(A, B) = !isequal(A, B)

# ============================================================
# ⍟  Log  —  natural log (monadic) / log-base (dyadic)
# ============================================================

"""
    ⍟(x) -> Number
    ⍟(b, x) -> Number

**Monadic** `⍟x`: natural logarithm of `x`.

**Dyadic** `b ⍟ x`: logarithm of `x` with base `b`.

# Examples
```julia
⍟(exp(1))   # 1.0
2 ⍟ 8       # 3.0
```
"""
⍟(x) = log(x)
⍟(b, x) = log(b, x)

# ============================================================
# ⍲  NAND  /  ⍱  NOR
# ============================================================

"""
    ⍲(a, b) -> Bool

**NAND**: logical not-and.  `a ⍲ b` is equivalent to `!(a && b)`.
Broadcasts over arrays.

# Examples
```julia
true ⍲ true    # false
true ⍲ false   # true
[true,false] ⍲ [true,true]  # [false,true]
```
"""
⍲(a::Bool, b::Bool) = !(a && b)
⍲(a, b) = .!(a .& b)

"""
    ⍱(a, b) -> Bool

**NOR**: logical not-or.  `a ⍱ b` is equivalent to `!(a || b)`.
Broadcasts over arrays.

# Examples
```julia
false ⍱ false   # true
true ⍱ false    # false
[true,false] ⍱ [false,false]  # [false,true]
```
"""
⍱(a::Bool, b::Bool) = !(a || b)
⍱(a, b) = .!(a .| b)

# ============================================================
# APL Operators  (higher-order functions)
# ============================================================

"""
    apl_reduce(f) -> Function

**APL Reduce** (`f/`): return a function that reduces an array with `f` from
left to right.

# Examples
```julia
apl_reduce(+)([1,2,3,4])   # 10
apl_reduce(*)([1,2,3,4])   # 24
apl_reduce(max)([3,1,4,1,5]) # 5
```
"""
apl_reduce(f) = A -> reduce(f, A)

"""
    apl_scan(f) -> Function

**APL Scan** (`f\\`): return a function that computes the running (prefix)
fold of an array with `f`.

# Examples
```julia
apl_scan(+)([1,2,3,4])   # [1, 3, 6, 10]
apl_scan(*)([1,2,3,4])   # [1, 2, 6, 24]
apl_scan(max)([3,1,4,2]) # [3, 3, 4, 4]
```
"""
apl_scan(f) = A -> accumulate(f, A)

"""
    apl_each(f) -> Function

**APL Each** (`f¨`): return a function that applies `f` to every element of
an array, preserving the array structure.

# Examples
```julia
apl_each(⍳)([3,4,5])
# [[1,2,3], [1,2,3,4], [1,2,3,4,5]]

apl_each(sum)([[1,2],[3,4,5]])  # [3, 12]
```
"""
apl_each(f) = A -> map(f, A)

"""
    apl_outer(f) -> Function

**APL Outer product** (`∘.f`): return a two-argument function `(A,B) ->
Matrix` where element `[i,j]` is `f(A[i], B[j])`.

# Examples
```julia
apl_outer(*)(1:3, 1:4)
# 3×4 multiplication table

apl_outer(+)([1,2,3], [10,20])
# [11 21; 12 22; 13 23]

apl_outer(==)(1:4, 1:4)
# 4×4 identity-like boolean matrix
```
"""
apl_outer(f) = (A, B) -> [f(a, b) for a in A, b in B]

"""
    apl_inner(f, g) -> Function

**APL Inner product** (`f.g`): return a two-argument function for the
generalised inner product.  The classic matrix multiply corresponds to
`apl_inner(+, *)`.

For vectors `u` and `v`, `apl_inner(f,g)(u,v)` computes
`reduce(f, g.(u, v))`.

For matrices, it generalises matrix multiplication by replacing `+` with `f`
and `*` with `g`.

# Examples
```julia
apl_inner(+, *)([1,2,3], [4,5,6])         # dot product = 32
apl_inner(+, *)([1 2; 3 4], [1 0; 0 1])   # matrix multiply (result = [1 2; 3 4])
apl_inner(max, +)([1,2], [3,4])           # max(1+3, 2+4) = 6  (vectors must be same length)
```
"""
function apl_inner(f, g)
    function inner(u::AbstractVector, v::AbstractVector)
        reduce(f, g.(u, v))
    end
    function inner(A::AbstractMatrix, v::AbstractVector)
        [reduce(f, g.(A[i, :], v)) for i in axes(A, 1)]
    end
    function inner(u::AbstractVector, B::AbstractMatrix)
        [reduce(f, g.(u, B[:, j])) for j in axes(B, 2)]
    end
    function inner(A::AbstractMatrix, B::AbstractMatrix)
        [reduce(f, g.(A[i, :], B[:, j]))
         for i in axes(A, 1), j in axes(B, 2)]
    end
    return inner
end

end # module JuliaAPL
