# prover/prover.py
from collections import defaultdict
import time, json, re, os
from typing import List, Tuple, Optional, Dict, Any

from .utils import color, parse_subgoals, state_fingerprint, RunLogger, slugify_goal, write_theory_file
from .config import (
    BEAM_WIDTH, MAX_DEPTH, HINT_LEMMAS, FACTS_LIMIT,
    MINIMIZE_DEFAULT, MINIMIZE_TIMEOUT, VARIANTS_DEFAULT,
    VARIANT_TIMEOUT, VARIANT_TRIES
)
from .llm import propose_steps, propose_finishers
from .isabelle_api import build_theory, run_theory, finished_ok, last_print_state_block, use_calls_count

# Replaced 4 imports with a single tactics module
from .tactics import (
    mine_lemmas_from_state, mine_facts_prioritized,
    sledgehammer_finishers,
    precheck_quickcheck_refutes, precheck_nitpick_refutes,
    try_structured_variants, suggest_continuations,
)

from .minimize import minimize_proof
from .ranker import SklearnReranker

_result_cache: Dict[Tuple[Tuple[str, ...], str], Tuple[bool, Optional[int], str]] = {}

def try_step_raw(isabelle, session_id: str, steps: List[str], cand: str) -> Tuple[bool, Optional[int], str, float]:
    t0 = time.monotonic()
    thy = build_theory(steps + [cand], add_print_state=True, end_with="sorry")
    resps = run_theory(isabelle, session_id, thy)
    ok, _ = finished_ok(resps)
    hint = last_print_state_block(resps) if ok else ""
    n = parse_subgoals(hint) if ok else None
    return ok, n, hint, (time.monotonic() - t0) * 1000

def try_step_cached(isabelle, session_id: str, steps: List[str], cand: str) -> Tuple[bool, Optional[int], str, bool, float]:
    key = (tuple(steps), cand)
    if key in _result_cache:
        ok, n_sub, hint = _result_cache[key]
        return ok, n_sub, hint, True, 0.0
    ok, n_sub, hint, elapsed_ms = try_step_raw(isabelle, session_id, steps, cand)
    _result_cache[key] = (ok, n_sub, hint)
    return ok, n_sub, hint, False, elapsed_ms

def try_finish(isabelle, session_id: str, steps: List[str], fin: str) -> Tuple[bool, float]:
    t0 = time.monotonic()
    thy = build_theory(steps + [fin], add_print_state=False, end_with=None)
    ok, _ = finished_ok(run_theory(isabelle, session_id, thy))
    return ok, (time.monotonic() - t0) * 1000

def prove_goal(isabelle, session_id: str, goal: str, model_name_or_ensemble: str,
               beam_w: int, max_depth: int, hint_lemmas: int, budget_s: int,
               models: Optional[List[str]] = None, save_dir: Optional[str] = None,
               use_sledge: bool = False, sledge_timeout: int = 5, sledge_every: int = 2,
               trace: bool = False, use_color: bool = True,
               use_qc: bool = False, qc_timeout: int = 2, qc_every: int = 1,
               use_np: bool = False, np_timeout: int = 5, np_every: int = 2,
               facts_limit: int = 6,
               do_minimize: bool = True, minimize_timeout: int = 8,
               do_variants: bool = True, variant_timeout: int = 6, variant_tries: int = 24,
               macro_map: Optional[Dict[str, List[Tuple[str, int]]]] = None, 
               enable_reranker: bool = True) -> Dict[str, Any]:

    reranker = None
    if enable_reranker and os.environ.get("RERANKER_DISABLE", "0") not in ("1", "true", "True"):
        reranker = SklearnReranker()

    global _result_cache
    _result_cache = {}

    model_list = models if models else [model_name_or_ensemble]
    display_model = ",".join(model_list)
    logger = RunLogger(goal, display_model)

    start_t = time.monotonic()
    def time_left_s() -> float: return budget_s - (time.monotonic() - start_t)

    seed_steps = [f'lemma "{goal}"']
    beam: List[Tuple[int, List[str], str, Optional[int]]] = [(9999, seed_steps, "", None)]
    visited_by_depth = defaultdict(set)  # depth -> set(state_fingerprint)

    if trace:
        print(color(use_color, "bold", f"\n▶ Goal: {goal}"))
        print(color(use_color, "gray", f"Models: {display_model} | Beam={beam_w} | MaxDepth={max_depth} | Budget={budget_s}s"))

    for depth in range(max_depth):
        if time_left_s() <= 0:
            best = min(beam, key=lambda t: (t[0], len("\n".join(t[1]))))
            logger.finish(False, best[1], depth, use_calls_count())
            if save_dir:
                name = slugify_goal(goal) + ".thy.partial"
                from .isabelle_api import build_theory as _bt
                write_theory_file(os.path.join(save_dir, name), _bt(best[1] + ["sorry"], False, None))
            return {"goal": goal, "success": False, "steps": best[1], "depth": depth,
                    "use_calls": use_calls_count(), "elapsed_s": logger.elapsed_s, "model": display_model, "timeout": True}

        depth_reached = depth + 1
        new_beam: List[Tuple[int, List[str], str, Optional[int]]] = []

        # Facts from top-of-beam
        facts_for_depth: List[str] = []
        if beam and beam[0][2]:
            try:
                facts_for_depth = mine_facts_prioritized(isabelle, session_id, beam[0][2], limit=facts_limit)
                if trace and facts_for_depth:
                    print(color(use_color, "yellow", f"Mined facts: {facts_for_depth}"))
            except Exception as e:
                if trace: print(color(use_color, "yellow", f"Facts mining error ignored: {e}"))

        # Sledgehammer (top-of-beam only)
        sledge_sugs: List[str] = []
        if use_sledge and depth % max(1, sledge_every) == 0 and beam:
            try:
                sledge_sugs = sledgehammer_finishers(isabelle, session_id, beam[0][1], timeout_s=sledge_timeout, limit=5)
                if trace and sledge_sugs:
                    print(color(use_color, "yellow", f"Sledgehammer finishers: {sledge_sugs}"))
            except Exception as e:
                if trace: print(color(use_color, "yellow", f" Sledgehammer error ignored: {e}"))

        # ---- Try to finish ----
        for j, (_, steps, state_hint, _) in enumerate(beam):
            mined = mine_lemmas_from_state(isabelle, session_id, state_hint, max_lemmas=hint_lemmas) if state_hint else []
            finishers_llm = propose_finishers(model_list, goal, steps, state_hint, mined, hint_lemmas, facts=facts_for_depth, temp=0.2)
            finishers = (sledge_sugs + finishers_llm) if (j == 0 and sledge_sugs) else finishers_llm
            for fin in finishers:
                if time_left_s() <= 0: break
                ok, elapsed_ms = try_finish(isabelle, session_id, steps, fin)
                logger.log_attempt("finish", steps, fin, ok, 0 if ok else None, False, elapsed_ms, depth_reached)
                if trace:
                    tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
                    print(f"  finish {tag} {fin}  ({round(elapsed_ms)}ms)")
                if ok:
                    final = steps + [fin]
                    if do_minimize:
                        try:
                            final = minimize_proof(isabelle, session_id, final, timeout_s=MINIMIZE_TIMEOUT, trace=trace, use_color=use_color)
                        except Exception as e:
                            if trace: print(color(use_color, "yellow", f"Minimize error ignored: {e}"))
                    if do_variants:
                        try:
                            from .heuristics import suggest_common_lemmas
                            seed_facts = suggest_common_lemmas(state_hint)
                            variant = try_structured_variants(isabelle, session_id, goal, final,
                                                              facts_seed=seed_facts, timeout_s=VARIANT_TIMEOUT,
                                                              max_tries=VARIANT_TRIES, trace=trace, use_color=use_color)
                            if variant: final = variant
                        except Exception as e:
                            if trace: print(color(use_color, "yellow", f"Variants error ignored: {e}"))
                    logger.finish(True, final, depth_reached, use_calls_count())
                    if save_dir:
                        name = slugify_goal(goal) + ".thy"
                        from .isabelle_api import build_theory as _bt
                        write_theory_file(os.path.join(save_dir, name), _bt(final, False, None))
                    if trace: print(color(use_color, "green", "✔ PROVED"))
                    return {"goal": goal, "success": True, "steps": final, "depth": depth_reached,
                            "use_calls": use_calls_count(), "elapsed_s": logger.elapsed_s, "model": display_model}

        # ---- Expand ----
        for _, steps, state_hint, _ in beam:
            if time_left_s() <= 0: break
            cands = propose_steps(model_list, goal, steps, state_hint, facts=facts_for_depth, reranker=reranker, depth=depth, temp=0.2)
            if trace: print(color(use_color, "blue", f"Proposals: {cands}"))
            if not cands: continue
            best_local: List[Tuple[int, List[str], str, Optional[int]]] = []
            for c in cands:
                if time_left_s() <= 0: break
                pruned = False
                if use_qc and depth % max(1, 1) == 0:
                    try:
                        if precheck_quickcheck_refutes(isabelle, session_id, steps + [c], timeout_s=2):
                            pruned = True
                            if trace: print(color(use_color, "dim", f"  step  ·  {c}  [pruned by quickcheck]"))
                    except Exception as e:
                        if trace: print(color(use_color, "yellow", f"  quickcheck error ignored: {e}"))
                if not pruned and use_np and depth % max(1, 2) == 0:
                    try:
                        if precheck_nitpick_refutes(isabelle, session_id, steps + [c], timeout_s=5):
                            pruned = True
                            if trace: print(color(use_color, "dim", f"  step  ·  {c}  [pruned by nitpick]"))
                    except Exception as e:
                        if trace: print(color(use_color, "yellow", f"  nitpick error ignored: {e}"))
                if pruned:
                    logger.log_attempt("expand", steps, c, False, None, False, 0.0, depth_reached)
                    continue
                ok, n_sub, hint, cache_hit, elapsed_ms = try_step_cached(isabelle, session_id, steps, c)
                logger.log_attempt("expand", steps, c, ok, n_sub, cache_hit, elapsed_ms, depth_reached)
                if trace:
                    tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
                    extra = f" n_sub={n_sub}" if n_sub is not None else ""
                    cache = color(use_color, "dim", " [cache]") if cache_hit else ""
                    print(f"  step  {tag} {c}{extra} ({round(elapsed_ms)}ms){cache}")
                if not ok: continue
                score = n_sub if n_sub is not None else 9999
                fp = state_fingerprint(hint or "")
                if fp in visited_by_depth[depth_reached]:
                    continue
                visited_by_depth[depth_reached].add(fp)
                # immediate macro continuation
                if macro_map:
                    from .macros import suggest_continuations
                    for cont in suggest_continuations(c, macro_map, k=1):
                        ok2, n_sub2, hint2, cache_hit2, elapsed_ms2 = try_step_cached(isabelle, session_id, steps + [c], cont)
                        logger.log_attempt("expand_macro", steps + [c], cont, ok2, n_sub2, cache_hit2, elapsed_ms2, depth_reached)
                        if ok2:
                            fp2 = state_fingerprint(hint2 or "")
                            if fp2 not in visited_by_depth[depth_reached]:
                                visited_by_depth[depth_reached].add(fp2)
                                score2 = n_sub2 if n_sub2 is not None else 9999
                                best_local.append((score2, steps + [c, cont], hint2, n_sub2))
                        break
                best_local.append((score, steps + [c], hint, n_sub))
            best_local.sort(key=lambda t: (t[0], len("\n".join(t[1]))))
            new_beam.extend(best_local[:2])

        if not new_beam:
            best = min(beam, key=lambda t: (t[0], len("\n".join(t[1]))))
            logger.finish(False, best[1], depth_reached, use_calls_count())
            if save_dir:
                name = slugify_goal(goal) + ".thy.partial"
                from .isabelle_api import build_theory as _bt
                write_theory_file(os.path.join(save_dir, name), _bt(best[1] + ["sorry"], False, None))
            if trace: print(color(use_color, "red", "✖ Search exhausted at this depth"))
            return {"goal": goal, "success": False, "steps": best[1], "depth": depth_reached,
                    "use_calls": use_calls_count(), "elapsed_s": logger.elapsed_s, "model": display_model}

        new_beam.sort(key=lambda t: (t[0], len("\n".join(t[1]))))
        seen_fp, dedup_beam = set(), []
        for it in new_beam:
            fp = state_fingerprint(it[2] or "")
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            dedup_beam.append(it)
        beam = dedup_beam[:beam_w]

    best = min(beam, key=lambda t: (t[0], len("\n".join(t[1]))))
    logger.finish(False, best[1], max_depth, use_calls_count())
    return {"goal": goal, "success": False, "steps": best[1], "depth": max_depth,
            "use_calls": use_calls_count(), "elapsed_s": logger.elapsed_s, "model": display_model}
