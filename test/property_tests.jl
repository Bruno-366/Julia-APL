# Property-based tests for JuliaAPL using Supposition.jl
#
# These tests verify algebraic laws that should hold for the APL primitives
# regardless of the specific input values.

using Test
using Supposition
using Supposition.Data: Vectors, Integers, Floats
using LinearAlgebra
using JuliaAPL
import JuliaAPL: ≡, ≢

const INTS  = Integers(-100, 100)
const POS   = Integers(1, 50)
const VECS  = Vectors(INTS; min_size=1, max_size=16)
const NVECS = Vectors(INTS; min_size=2, max_size=12)  # non-empty, used for ⌽/⊖

@testset "Property-based tests (Supposition.jl)" begin

    # ── ⍳  iota ────────────────────────────────────────────────────────────────
    @testset "⍳ monadic: length and bounds" begin
        @check function iota_length(n=POS)
            length(⍳(n)) == n
        end
        @check function iota_one_indexed(n=POS)
            ⍳(n)[1] == 1 && ⍳(n)[end] == n
        end
    end

    @testset "⍳ dyadic: found element returns valid index" begin
        @check function iota_found(v=VECS)
            # every element of v should be found in v (at valid position)
            all(⍳(v, x) in 1:length(v) for x in v)
        end
        @check function iota_not_found_returns_length_plus_one(v=VECS)
            sentinel = maximum(v) + 1  # guaranteed not in v
            ⍳(v, sentinel) == length(v) + 1
        end
    end

    # ── ⍴  rho ─────────────────────────────────────────────────────────────────
    @testset "⍴ monadic: identity round-trip" begin
        @check function rho_shape_of_vector(v=VECS)
            ⍴(v) == [length(v)]
        end
    end

    @testset "⍴ dyadic: reshape preserves element count" begin
        @check function rho_reshape_count(v=NVECS, n=Integers(1, 6))
            result = ⍴(n, v)
            length(result) == n
        end
    end

    # ── ∊  enlist (flatten) ────────────────────────────────────────────────────
    @testset "∊ monadic: length equals element count" begin
        @check function enlist_length(v=VECS)
            length(∊(v)) == length(v)
        end
    end

    # ── ⌽  reverse ─────────────────────────────────────────────────────────────
    @testset "⌽ involution: double-reverse = identity" begin
        @check function reverse_involution(v=NVECS)
            ⌽(⌽(v)) == v
        end
    end

    @testset "⌽ rotate by length = identity" begin
        @check function rotate_by_length(v=NVECS)
            ⌽(length(v), v) == v
        end
    end

    @testset "⌽ rotate and reverse are consistent" begin
        @check function rotate_consistency(v=NVECS)
            # rotating by 1 = shifting first element to the end
            n = length(v)
            ⌽(1, v) == vcat(v[2:end], [v[1]])
        end
    end

    # ── ⊖  reverse first axis (vectors behave same as ⌽) ─────────────────────
    @testset "⊖ involution on vectors" begin
        @check function reverse_first_involution(v=NVECS)
            ⊖(⊖(v)) == v
        end
    end

    # ── ⍋ / ⍒  grade ──────────────────────────────────────────────────────────
    @testset "⍋ grade-up: v[⍋(v)] is sorted ascending" begin
        @check function grade_up_sorts(v=NVECS)
            v[⍋(v)] == sort(v)
        end
    end

    @testset "⍒ grade-down: v[⍒(v)] is sorted descending" begin
        @check function grade_down_sorts(v=NVECS)
            v[⍒(v)] == sort(v; rev=true)
        end
    end

    # ── ↑ / ↓  take / drop ────────────────────────────────────────────────────
    @testset "↑ / ↓ complement: take-n ++ drop-n = original" begin
        @check function take_drop_complement(v=NVECS, n=Integers(0, 8))
            k = min(n, length(v))
            vcat((k ↑ v), (k ↓ v)) == v
        end
    end

    @testset "↑ take: length of result" begin
        @check function take_length(v=NVECS, n=Integers(0, 20))
            length(n ↑ v) == n
        end
    end

    # ── ≡ / ≢  match ──────────────────────────────────────────────────────────
    @testset "≡ self-match" begin
        @check function self_match(v=VECS)
            v ≡ v
        end
    end

    @testset "≢ is negation of ≡" begin
        @check function not_match_negation(v=VECS, w=VECS)
            (v ≢ w) == !(v ≡ w)
        end
    end

    # ── ⍟  log ─────────────────────────────────────────────────────────────────
    @testset "⍟ b^(⍟(b,x)) ≈ x" begin
        pos_floats = Floats(minimum=0.001, maximum=1000.0, infs=false, nans=false)
        bases      = Floats(minimum=1.001, maximum=100.0,  infs=false, nans=false)
        @check function log_base_roundtrip(b=bases, x=pos_floats)
            b ^ ⍟(b, x) ≈ x
        end
    end

    # ── apl_reduce ─────────────────────────────────────────────────────────────
    @testset "apl_reduce(+) == sum" begin
        @check function reduce_plus_is_sum(v=NVECS)
            apl_reduce(+)(v) == sum(v)
        end
    end

    @testset "apl_reduce(*) == prod" begin
        @check function reduce_times_is_prod(v=NVECS)
            apl_reduce(*)(v) == prod(v)
        end
    end

    # ── apl_scan ───────────────────────────────────────────────────────────────
    @testset "apl_scan(+) last element == sum" begin
        @check function scan_plus_last_is_sum(v=NVECS)
            last(apl_scan(+)(v)) == sum(v)
        end
    end

    @testset "apl_scan length = input length" begin
        @check function scan_length(v=NVECS)
            length(apl_scan(+)(v)) == length(v)
        end
    end

    # ── apl_outer ──────────────────────────────────────────────────────────────
    @testset "apl_outer(*) shape is (m, n)" begin
        @check function outer_shape(u=NVECS, v=NVECS)
            m, n = length(u), length(v)
            size(apl_outer(*)(u, v)) == (m, n)
        end
    end

end
