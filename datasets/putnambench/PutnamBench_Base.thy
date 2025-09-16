theory PutnamBench_Base
  imports
    Complex_Main

    (* Analysis umbrella: covers Finite_Cartesian_Product, Linear_Algebra,
       Lebesgue_Measure, Interval_Integral, Derivative, Determinants, etc. *)
    "HOL-Analysis.Analysis"

    (* Algebraic/combinatorial computation: Polynomial, Primes, FPS, etc. *)
    "HOL-Computational_Algebra.Computational_Algebra"

    (* Number theory (Congruences, Primes, …). *)
    "HOL-Number_Theory.Number_Theory"

    (* Library fragments that recur in PutnamBench. *)
    "HOL-Library.Cardinality"
    "HOL-Library.Extended_Real"
    "HOL-Library.Extended_Nonnegative_Real"
    "HOL-Library.Multiset"
    "HOL-Library.Disjoint_Sets"
    "HOL-Library.Interval"
    "HOL-Library.FuncSet"
    "HOL-Library.Countable_Set"
    "HOL-Library.Liminf_Limsup"
    "HOL-Library.Periodic_Fun"
    "HOL-Library.Sum_of_Squares"
    "HOL-Library.Code_Target_Numeral"

    (* Algebra bits seen in imports. *)
    "HOL-Algebra.Group"
    "HOL-Algebra.Ring"
    "HOL-Algebra.Multiplicative_Group"
    "HOL-Algebra.Complete_Lattice"

    (* Combinatorics used by several problems. *)
    "HOL-Combinatorics.Permutations"

    (* Probability umbrella for Probability_Measure / Independent_Family. *)
    "HOL-Probability.Probability"
begin

end
