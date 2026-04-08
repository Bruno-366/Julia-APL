using Test
using LinearAlgebra
using JuliaAPL
import JuliaAPL: ≡, ≢  # resolve ambiguity with Base.≡ (===) and Base.≢

@testset "JuliaAPL" begin

    # ─────────────────────────────────────────────────────────────────────────
    @testset "⍳  iota / index-of" begin
        @test ⍳(5)  == [1, 2, 3, 4, 5]
        @test ⍳(1)  == [1]
        @test ⍳(0)  == Int[]

        # dyadic: index of a scalar (function-call syntax, ⍳ is not an infix op in Julia)
        @test ⍳([10, 20, 30], 20) == 2
        @test ⍳([10, 20, 30], 10) == 1
        @test ⍳([10, 20, 30], 99) == 4   # not found → length + 1

        # dyadic: index of a vector
        @test ⍳([10, 20, 30], [20, 30]) == [2, 3]
        @test ⍳([10, 20, 30], [99, 10]) == [4, 1]

        # dyadic: matrix left-arg uses APL row-major ravel order; result preserves B's shape
        A = [10 20; 30 40]
        B = [40 10; 99 30]
        result = ⍳(A, B)
        @test result == [4 1; 5 3]
        @test size(result) == size(B)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "⍴  shape / reshape" begin
        # monadic
        @test ⍴([1, 2, 3])          == [3]
        @test ⍴([1 2 3; 4 5 6])     == [2, 3]
        @test ⍴(42)                  == Int[]

        # dyadic: reshape vector → matrix (function-call syntax, ⍴ is not infix in Julia)
        result = ⍴([2, 3], 1:6)
        @test result == [1 2 3; 4 5 6]
        @test size(result) == (2, 3)

        # dyadic: scalar fill
        mat = ⍴([2, 3], 0)
        @test mat == zeros(Int, 2, 3)

        # dyadic with single integer
        @test ⍴(3, [10, 20, 30, 40]) == [10, 20, 30]

        # dyadic: cycling (APL wraps around)
        @test ⍴([2, 4], [1, 2, 3]) == [1 2 3 1; 2 3 1 2]

        # dyadic: reshaping a 2-D source uses APL row-major ravel order
        src2d = [1 2 3; 4 5 6]
        @test ⍴(6, src2d)       == [1, 2, 3, 4, 5, 6]
        @test ⍴([3, 2], src2d)  == [1 2; 3 4; 5 6]
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "⌹  matrix inverse / matrix divide" begin
        M = [1.0 2.0; 3.0 4.0]

        # monadic: inverse
        inv_M = ⌹(M)
        @test inv_M ≈ inv(M)
        @test M * inv_M ≈ I

        # dyadic: matrix divide (function-call syntax, ⌹ is not infix in Julia)
        b = [1.0, 2.0]
        x = ⌹(M, b)
        @test M * x ≈ b
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "⍉  transpose / permute axes" begin
        A = [1 2 3; 4 5 6]       # 2×3

        # monadic: transpose
        @test ⍉(A) == [1 4; 2 5; 3 6]
        @test size(⍉(A)) == (3, 2)

        # monadic: vector transpose is a no-op (returns a copy)
        v = [1, 2, 3]
        @test ⍉(v) == v
        @test size(⍉(v)) == (3,)

        # dyadic: permute axes of 3-D array (function-call syntax, ⍉ is not infix in Julia)
        T = reshape(1:24, 2, 3, 4)
        @test size(⍉([2, 1, 3], T)) == (3, 2, 4)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "⍋ grade up / ⍒ grade down" begin
        A = [3, 1, 4, 1, 5, 9, 2, 6]

        perm_up = ⍋(A)
        @test A[perm_up] == sort(A)

        perm_dn = ⍒(A)
        @test A[perm_dn] == sort(A, rev=true)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "⌽  reverse / rotate (last axis)" begin
        v = [1, 2, 3, 4, 5]

        # monadic: reverse
        @test ⌽(v) == [5, 4, 3, 2, 1]

        # dyadic: left-rotate (function-call syntax, ⌽ is not infix in Julia)
        @test ⌽(2, v)  == [3, 4, 5, 1, 2]
        @test ⌽(-1, v) == [5, 1, 2, 3, 4]  # right-rotate by 1

        # matrix: reverse columns
        M = [1 2 3; 4 5 6]
        @test ⌽(M) == [3 2 1; 6 5 4]
        @test ⌽(1, M) == [2 3 1; 5 6 4]
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "⊖  reverse / rotate (first axis)" begin
        v = [1, 2, 3, 4, 5]

        # monadic: reverse
        @test ⊖(v) == [5, 4, 3, 2, 1]

        # dyadic: up-rotate rows of matrix (⊖ works as infix in Julia)
        M = [1 2; 3 4; 5 6]
        @test ⊖(M) == [5 6; 3 4; 1 2]
        @test (1 ⊖ M) == [3 4; 5 6; 1 2]
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "∊  enlist / membership" begin
        A = [1 2; 3 4]

        # monadic: flatten
        @test ∊(A) == [1, 2, 3, 4]

        # dyadic: membership (vector inputs)
        @test ([1, 2, 3, 4] ∊ [2, 4]) == [false, true, false, true]
        @test ([5, 6] ∊ [1, 2, 3])    == [false, false]

        # dyadic: membership preserves A's shape (matrix input)
        @test (A ∊ [2, 4]) == Bool[false true; false true]
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "↑  take" begin
        v = [10, 20, 30, 40, 50]

        @test (3 ↑ v)    == [10, 20, 30]
        @test ((-2) ↑ v) == [40, 50]       # use explicit parens for negative
        @test (0 ↑ v)    == Int[]

        # padding
        @test (7 ↑ [1, 2, 3]) == [1, 2, 3, 0, 0, 0, 0]

        # matrix rows
        M = [1 2; 3 4; 5 6]
        @test (2 ↑ M)    == [1 2; 3 4]
        @test ((-1) ↑ M) == reshape([5, 6], 1, 2)  # explicit parens for negative
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "↓  drop" begin
        v = [10, 20, 30, 40, 50]

        @test (2 ↓ v)    == [30, 40, 50]
        @test ((-2) ↓ v) == [10, 20, 30]  # explicit parens for negative
        @test (5 ↓ v)    == Int[]

        # matrix rows
        M = [1 2; 3 4; 5 6]
        @test (1 ↓ M)    == [3 4; 5 6]
        @test ((-2) ↓ M) == reshape([1, 2], 1, 2)  # explicit parens for negative
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "⊂  enclose / ⊃  first" begin
        v = [1, 2, 3]

        @test ⊂(v)  == [v]
        @test ⊃(⊂(v)) == v   # round-trip

        @test ⊃([10, 20, 30]) == 10
        @test (2 ⊃ [10, 20, 30]) == 20
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "≡  depth / match" begin
        @test ≡(42)            == 0
        @test ≡("hello")       == 0   # any non-array atom has depth 0
        @test ≡([1, 2, 3])     == 1
        @test ≡([[1, 2], [3]])  == 2

        @test ([1, 2, 3] ≡ [1, 2, 3]) == true
        @test ([1, 2, 3] ≡ [1, 2, 4]) == false
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "≢  tally / not-match" begin
        @test ≢([1, 2, 3, 4])       == 4
        @test ≢([1 2; 3 4; 5 6])    == 3
        @test ≢(42)                  == 1

        @test ([1, 2] ≢ [1, 3]) == true
        @test ([1, 2] ≢ [1, 2]) == false
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "⍟  log" begin
        @test ⍟(exp(1)) ≈ 1.0
        @test ⍟(1.0)    ≈ 0.0
        @test ⍟(2, 8)   ≈ 3.0   # function-call syntax, ⍟ is not infix in Julia
        @test ⍟(10, 100) ≈ 2.0
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "⍲  NAND / ⍱  NOR" begin
        @test ⍲(true,  true)  == false  # function-call syntax, ⍲ is not infix in Julia
        @test ⍲(true,  false) == true
        @test ⍲(false, true)  == true
        @test ⍲(false, false) == true

        @test ⍱(true,  true)  == false  # function-call syntax, ⍱ is not infix in Julia
        @test ⍱(true,  false) == false
        @test ⍱(false, true)  == false
        @test ⍱(false, false) == true

        # broadcast over arrays
        @test ⍲([true, false], [true, true])   == [false, true]
        @test ⍱([false, false], [false, true]) == [true, false]
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "apl_reduce" begin
        @test apl_reduce(+)([1, 2, 3, 4])  == 10
        @test apl_reduce(*)([1, 2, 3, 4])  == 24
        @test apl_reduce(max)([3, 1, 4, 1, 5]) == 5
        @test apl_reduce(min)([3, 1, 4, 1, 5]) == 1
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "apl_scan" begin
        @test apl_scan(+)([1, 2, 3, 4])   == [1, 3, 6, 10]
        @test apl_scan(*)([1, 2, 3, 4])   == [1, 2, 6, 24]
        @test apl_scan(max)([3, 1, 4, 1]) == [3, 3, 4, 4]
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "apl_each" begin
        result = apl_each(⍳)([3, 4])
        @test result[1] == [1, 2, 3]
        @test result[2] == [1, 2, 3, 4]

        @test apl_each(sum)([[1, 2], [3, 4, 5]]) == [3, 12]
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "apl_outer" begin
        # multiplication table
        table = apl_outer(*)(1:3, 1:3)
        @test table == [1 2 3; 2 4 6; 3 6 9]

        # addition
        @test apl_outer(+)([1, 2], [10, 20]) == [11 21; 12 22]

        # equality — produces identity-like boolean matrix
        eq = apl_outer(==)(1:3, 1:3)
        @test eq == [true false false; false true false; false false true]
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "apl_inner" begin
        dot = apl_inner(+, *)

        # dot product of two vectors
        @test dot([1, 2, 3], [4, 5, 6]) == 32

        # matrix multiply
        A = [1 2; 3 4]
        B = [5 6; 7 8]
        @test dot(A, B) == A * B

        # identity check
        I2 = [1 0; 0 1]
        @test dot(A, I2) == A

        # generalised: max.+ inner product (tropical algebra) — vectors must match length
        tropical = apl_inner(max, +)
        @test tropical([1, 2], [3, 4]) == max(1 + 3, 2 + 4)  # = 6
    end

end # @testset "JuliaAPL"

# Run property-based tests if Supposition.jl is available
if haskey(Base.loaded_modules, Base.PkgId(Base.UUID("5a0628fe-1370-4cba-a1f9-24f9bde67092"), "Supposition"))
    include("property_tests.jl")
else
    try
        @eval using Supposition
        include("property_tests.jl")
    catch
        @info "Supposition.jl not available — skipping property-based tests"
    end
end
