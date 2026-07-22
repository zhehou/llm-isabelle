from collections import defaultdict, OrderedDict
import time, json, re, os, hashlib
from typing import List, Tuple, Optional, Dict, Any

from .utils import color, parse_subgoals, state_fingerprint, RunLogger, slugify_goal, write_theory_file
from .config import (
    BEAM_WIDTH, MAX_DEPTH, HINT_LEMMAS, FACTS_LIMIT,
    MINIMIZE_DEFAULT, MINIMIZE_TIMEOUT, VARIANTS_DEFAULT,
    VARIANT_TIMEOUT, VARIANT_TRIES,
)
# IMPORTANT: read premise/context from live config, not frozen names
from . import config as CFG
from .llm import propose_steps, propose_finishers
from .isabelle_api import (
    build_theory, run_theory, finished_ok, first_error_message, last_print_state_block,
    use_calls_count, last_call_timed_out,
)

from .tactics import (
    mine_lemmas_from_state, mine_facts_prioritized,
    sledgehammer_finishers,
    precheck_quickcheck_refutes, precheck_nitpick_refutes,
    try_structured_variants, suggest_continuations,
    variant_step_templates,
    fact_composition_finishers,
)

from .minimize import minimize_proof
from .features import flags_from_goal
from .ranker import Reranker
# new: premise retrieval & optional file-aware context
from .premises import PremisesIndex, build_score_map, cand_features, selection_features
from .heuristics import extract_candidate_facts
from .context import ContextWindow
from .strategies import (
    definition_guided_finishers,
    definition_guided_obligations,
    definition_guided_steps,
    generic_structural_finishers,
    generic_structural_steps,
)

_result_cache: Dict[Tuple[Tuple[str, ...], str], Tuple[bool, Optional[int], str]] = {}

_GLOBAL_CACHE_MAX = int(os.getenv("GLOBAL_STEP_CACHE_MAX", "8192"))
_global_result_cache: "OrderedDict[tuple, tuple]" = OrderedDict()

_GOAL_LINE = re.compile(r'lemma\s+"(.*)"')

def _global_cache_key(steps: List[str], cand: str) -> tuple:
    lemma_line = next((line for line in steps if _GOAL_LINE.search(line)), "")
    m = _GOAL_LINE.search(lemma_line)
    goal_s = m.group(1) if m else lemma_line
    prefix = "\n".join(steps[:-1]) if len(steps) > 1 else ""
    g = hashlib.sha1(goal_s.encode("utf-8")).hexdigest()[:12]
    p = hashlib.sha1(prefix.encode("utf-8")).hexdigest()[:12]
    return (g, p, cand)

def _global_cache_get(k: tuple):
    v = _global_result_cache.get(k)
    if v is not None:
        _global_result_cache.move_to_end(k)
    return v

def _global_cache_put(k: tuple, v: tuple):
    _global_result_cache[k] = v
    _global_result_cache.move_to_end(k)
    if len(_global_result_cache) > _GLOBAL_CACHE_MAX:
        _global_result_cache.popitem(last=False)


def try_step_raw(
    isabelle,
    session_id: str,
    steps: List[str],
    cand: str,
    timeout_s: Optional[int] = None,
) -> Tuple[bool, Optional[int], str, float]:
    t0 = time.monotonic()
    thy = build_theory(steps + [cand], add_print_state=True, end_with="sorry")
    resps = run_theory(isabelle, session_id, thy, timeout_s=timeout_s)
    ok, _ = finished_ok(resps)
    hint = last_print_state_block(resps) if ok else ""
    n = parse_subgoals(hint) if ok else None
    return ok, n, hint, (time.monotonic() - t0) * 1000

def try_step_cached(
    isabelle,
    session_id: str,
    steps: List[str],
    cand: str,
    timeout_s: Optional[int] = None,
) -> Tuple[bool, Optional[int], str, bool, float]:
    gkey = _global_cache_key(steps, cand)
    gval = _global_cache_get(gkey)
    if gval is not None:
        ok, n_sub, hint = gval
        return ok, n_sub, hint, True, 0.0

    key = (tuple(steps), cand)
    if key in _result_cache:
        ok, n_sub, hint = _result_cache[key]
        _global_cache_put(gkey, (ok, n_sub, hint))
        return ok, n_sub, hint, True, 0.0

    ok, n_sub, hint, elapsed_ms = try_step_raw(isabelle, session_id, steps, cand, timeout_s=timeout_s)
    # Do not cache wall-clock timeouts: a later retry with more budget may succeed.
    if not last_call_timed_out():
        _result_cache[key] = (ok, n_sub, hint)
        _global_cache_put(gkey, (ok, n_sub, hint))
    return ok, n_sub, hint, False, elapsed_ms


def try_finish(
    isabelle,
    session_id: str,
    steps: List[str],
    fin: str,
    timeout_s: Optional[int] = None,
) -> Tuple[bool, float]:
    t0 = time.monotonic()
    thy = build_theory(steps + [fin], add_print_state=False, end_with=None)
    ok, _ = finished_ok(run_theory(isabelle, session_id, thy, timeout_s=timeout_s))
    return ok, (time.monotonic() - t0) * 1000


def try_finish_detailed(
    isabelle,
    session_id: str,
    steps: List[str],
    fin: str,
    timeout_s: Optional[int] = None,
) -> Tuple[bool, float, str]:
    """Check a finisher and retain the first Isabelle error for tracing."""

    t0 = time.monotonic()
    thy = build_theory(steps + [fin], add_print_state=False, end_with=None)
    responses = run_theory(isabelle, session_id, thy, timeout_s=timeout_s)
    ok, _ = finished_ok(responses)
    diagnostic = "" if ok else first_error_message(responses)
    return ok, (time.monotonic() - t0) * 1000, diagnostic


def _trace_diagnostic(text: str, *, limit: int = 600) -> str:
    compact = " | ".join(
        line.strip() for line in str(text or "").splitlines() if line.strip()
    )
    if len(compact) > limit:
        return compact[: max(0, limit - 1)].rstrip() + "…"
    return compact


def prove_goal(isabelle, session_id: str, goal: str, model_name_or_ensemble: str,
               beam_w: int, max_depth: int, hint_lemmas: int,
               timeout: Optional[int] = None,  # timeout is the single source of truth
               models: Optional[List[str]] = None, save_dir: Optional[str] = None,
               use_sledge: bool = False, sledge_timeout: int = 5, sledge_every: int = 2,
               trace: bool = False, use_color: bool = True,
               use_qc: bool = False, qc_timeout: int = 2, qc_every: int = 1,
               use_np: bool = False, np_timeout: int = 5, np_every: int = 2,
               facts_limit: int = 6,
               do_minimize: bool = True, minimize_timeout: int = 8,
               do_variants: bool = True, variant_timeout: int = 6, variant_tries: int = 24,
               macro_map: Optional[Dict[str, List[Tuple[str, int]]]] = None,
               enable_reranker: bool = True,
               *,
               initial_state_hint: Optional[str] = None,
               theory_preamble: str = "",
               deterministic_only: bool = False,
               ) -> Dict[str, Any]:

    budget = timeout if timeout is not None else 10

    global _result_cache
    _result_cache = {}

    # Resolve models robustly (planner may pass model=None)
    if models:
        model_list = [str(m) for m in models if m]
    else:
        default_model = model_name_or_ensemble or os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")
        model_list = [str(default_model)]
    if not model_list:
        model_list = [os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")]
    display_model = ",".join(model_list)
    logger = RunLogger(goal, display_model)

    start_t = time.monotonic()
    def time_left_s() -> float: return budget - (time.monotonic() - start_t)

    # Keep local datatypes, definitions and simp equations in every standalone
    # search theory.  Without this prefix a benchmark-defined constant is
    # parsed as a fresh free variable and the prover searches the wrong logic.
    preamble_steps = theory_preamble.strip().splitlines() if theory_preamble.strip() else []
    seed_steps = preamble_steps + [f'lemma "{goal}"']
    if trace:
        print(color(use_color, "bold", f"\n▶ Goal: {goal}"))
        print(color(use_color, "gray", f"Models: {display_model} | Beam={beam_w} | MaxDepth={max_depth} | Timeout={budget}s"))

    # Isabelle automation is cheaper, faster and more reliable than asking a
    # model to rediscover a one-line proof.  Run this before constructing the
    # reranker, context index or premise encoder, and before any model call.
    # Tier 0 contains only cheap, high-yield direct closure.  Previously six
    # generic methods ran before structural search; on a 35-second fill budget
    # their theory startup costs left roughly one second for induction.
    automation_finishers = ("by simp", "by auto")
    attempted_finishers = set()
    for automation_index, fin in enumerate(automation_finishers):
        if time_left_s() <= 0.5:
            break
        # A use_theories call commonly spends a little over two seconds in
        # client/server and theory setup before the method itself runs.  The
        # old two-second cap therefore classified successful methods as
        # failures.  Share the remaining portfolio budget, but retain enough
        # headroom for a cold Isabelle invocation.
        attempts_left = max(1, len(automation_finishers) - automation_index)
        fair_share = time_left_s() / attempts_left
        per_automation_timeout = max(3.0, min(6.0, fair_share * 0.65))
        per_automation_timeout = max(1, int(min(time_left_s(), per_automation_timeout) + 0.999))
        try:
            attempted_finishers.add((tuple(seed_steps), fin))
            ok, elapsed_ms = try_finish(
                isabelle,
                session_id,
                seed_steps,
                fin,
                timeout_s=per_automation_timeout,
            )
        except Exception:
            ok, elapsed_ms = False, 0.0
        if trace:
            tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
            print(f"  finish {tag} [automation] {fin}  ({round(elapsed_ms)}ms)")
        if ok:
            final = seed_steps + [fin]
            logger.finish(True, final, 1, use_calls_count())
            if save_dir:
                name = slugify_goal(goal) + ".thy"
                write_theory_file(
                    os.path.join(save_dir, name),
                    build_theory(final, False, None),
                )
            if trace:
                print(color(use_color, "green", "✔ PROVED"))
            return {
                "goal": goal,
                "success": True,
                "steps": final,
                "depth": 1,
                "use_calls": use_calls_count(),
                "elapsed_s": logger.elapsed_s,
                "model": display_model,
                "origin": "automation",
            }

    standard_finishers = generic_structural_finishers(
        goal, initial_state_hint or "", limit=5
    )
    composition_finishers = fact_composition_finishers(goal, limit=5)

    # Tier 1: synthesize complete strategies from the defining equations.
    # These are intentionally tested before broad automation, Sledgehammer,
    # retrieval, or an LLM call.  A candidate couples the structural decision
    # with branch closure, e.g. induction on the recursive argument with the
    # state/accumulator generalized followed by simp_all.
    guided_finishers = definition_guided_finishers(
        goal, theory_preamble, initial_state_hint or "", limit=6
    )
    for index, fin in enumerate(guided_finishers):
        if time_left_s() <= max(3.0, budget * 0.25):
            break
        attempted_finishers.add((tuple(seed_steps), fin))
        try:
            ok, elapsed_ms = try_finish(
                isabelle,
                session_id,
                seed_steps,
                fin,
                timeout_s=max(3, int(min(8.0, time_left_s()))),
            )
        except Exception:
            ok, elapsed_ms = False, 0.0
        if trace:
            tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
            print(f"  finish {tag} [definition] {fin}  ({round(elapsed_ms)}ms)")
        if ok:
            final = seed_steps + [fin]
            logger.finish(True, final, 1, use_calls_count())
            if save_dir:
                name = slugify_goal(goal) + ".thy"
                write_theory_file(os.path.join(save_dir, name), build_theory(final, False, None))
            if trace:
                print(color(use_color, "green", "✔ PROVED"))
            return {
                "goal": goal,
                "success": True,
                "steps": final,
                "depth": 1,
                "use_calls": use_calls_count(),
                "elapsed_s": logger.elapsed_s,
                "model": display_model,
                "origin": "definition",
                "strategy_rank": index,
            }

    # Imported HOL functions have no equations in ``theory_preamble``.  Use
    # their standard recursive argument positions to test complete induction
    # macros early.  This is the library counterpart of definition-directed
    # search and prevents generic finishers or model latency from starving the
    # only structural move that can solve the goal.
    for index, fin in enumerate(standard_finishers):
        if time_left_s() <= max(3.0, budget * 0.22):
            break
        attempted_finishers.add((tuple(seed_steps), fin))
        try:
            ok, elapsed_ms = try_finish(
                isabelle,
                session_id,
                seed_steps,
                fin,
                timeout_s=max(4, int(min(9.0, time_left_s()))),
            )
        except Exception:
            ok, elapsed_ms = False, 0.0
        if trace:
            tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
            print(f"  finish {tag} [library-structure] {fin}  ({round(elapsed_ms)}ms)")
        if ok:
            final = seed_steps + [fin]
            logger.finish(True, final, 1, use_calls_count())
            if save_dir:
                name = slugify_goal(goal) + ".thy"
                write_theory_file(os.path.join(save_dir, name), build_theory(final, False, None))
            if trace:
                print(color(use_color, "green", "✔ PROVED"))
            return {
                "goal": goal, "success": True, "steps": final, "depth": 1,
                "use_calls": use_calls_count(), "elapsed_s": logger.elapsed_s,
                "model": display_model, "origin": "library-structure",
                "strategy_rank": index,
            }

    # Compose assumptions with a small, goal-directed rule set after atomic
    # structural strategies but before obligations, retrieval or generation.
    # Typical examples are eliminating ``inj`` and chaining two order facts.
    for fin in composition_finishers:
        if time_left_s() <= max(3.0, budget * 0.22):
            break
        attempted_finishers.add((tuple(seed_steps), fin))
        try:
            ok, elapsed_ms = try_finish(
                isabelle,
                session_id,
                seed_steps,
                fin,
                timeout_s=max(3, int(min(7.0, time_left_s()))),
            )
        except Exception:
            ok, elapsed_ms = False, 0.0
        if trace:
            tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
            print(f"  finish {tag} [fact-composition] {fin}  ({round(elapsed_ms)}ms)")
        if ok:
            final = seed_steps + [fin]
            logger.finish(True, final, 1, use_calls_count())
            if trace:
                print(color(use_color, "green", "✔ PROVED"))
            return {
                "goal": goal, "success": True, "steps": final, "depth": 1,
                "use_calls": use_calls_count(), "elapsed_s": logger.elapsed_s,
                "model": display_model, "origin": "fact-composition",
            }

    # Tier 1b: verified auxiliary-obligation search.  Each conjecture is first
    # proved in its own theorem transaction.  Only a checked theorem is added
    # to the context of the main goal, giving us a small proof DAG instead of
    # one opaque helper script.  This distinction matters for invariant
    # synthesis: a useful but false conjecture is rejected before it can waste
    # the main induction budget.
    obligations = definition_guided_obligations(
        goal, theory_preamble, initial_state_hint or "", limit=3
    )
    # A schema can serve several consumer inductions.  Cache proof checks by
    # theorem body/method so ranking variants never spend another Isabelle
    # transaction on an obligation already accepted or rejected in this run.
    obligation_proof_cache: Dict[Tuple[str, str], Tuple[bool, float, str]] = {}
    for index, obligation in enumerate(obligations):
        if time_left_s() <= max(5.0, budget * 0.25):
            break
        base_name = obligation.name_hint
        aux_name = base_name
        suffix = 1
        occupied = set(re.findall(r"\b(?:lemma|theorem|have)\s+([A-Za-z_][A-Za-z0-9_']*)", theory_preamble or ""))
        while aux_name in occupied:
            suffix += 1
            aux_name = f"{base_name}_{suffix}"

        checked_proof: Optional[str] = None
        aux_seed = preamble_steps + [
            f'lemma {aux_name}: "{obligation.top_level_statement}"'
        ]
        if trace:
            print(
                f'  obligation candidate [{obligation.kind}] {aux_name}: '
                f'"{obligation.top_level_statement}"'
            )
        for proof_method in obligation.proof_methods[:2]:
            if time_left_s() <= max(4.0, budget * 0.20):
                break
            cache_key = (obligation.top_level_statement, proof_method)
            cached = cache_key in obligation_proof_cache
            if cached:
                aux_ok, aux_ms, diagnostic = obligation_proof_cache[cache_key]
            else:
                try:
                    aux_ok, aux_ms, diagnostic = try_finish_detailed(
                        isabelle,
                        session_id,
                        aux_seed,
                        proof_method,
                        timeout_s=max(4, int(min(10.0, time_left_s()))),
                    )
                except Exception as exc:
                    aux_ok, aux_ms = False, 0.0
                    diagnostic = f"{type(exc).__name__}: {exc}"
                obligation_proof_cache[cache_key] = (
                    aux_ok, aux_ms, diagnostic
                )
            if trace:
                tag = color(use_color, "green", "✓") if aux_ok else color(use_color, "red", "×")
                cache_note = " [cached]" if cached else ""
                print(
                    f"  obligation {tag} [{obligation.kind}] {proof_method} "
                    f"({round(aux_ms)}ms){cache_note}"
                )
                rendered_diagnostic = _trace_diagnostic(diagnostic)
                if not aux_ok and rendered_diagnostic:
                    print(f"    Isabelle: {rendered_diagnostic}")
            if aux_ok:
                checked_proof = proof_method
                break
        if checked_proof is None:
            continue

        combined_seed = [
            *preamble_steps,
            f'lemma {aux_name}: "{obligation.top_level_statement}"',
            checked_proof,
            f'lemma "{goal}"',
        ]
        for main_template in obligation.main_methods[:3]:
            if time_left_s() <= max(3.0, budget * 0.12):
                break
            main_method = main_template.format(aux=aux_name)
            try:
                ok, elapsed_ms = try_finish(
                    isabelle,
                    session_id,
                    combined_seed,
                    main_method,
                    timeout_s=max(4, int(min(12.0, time_left_s()))),
                )
            except Exception:
                ok, elapsed_ms = False, 0.0
            if trace:
                tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
                print(
                    f"  finish {tag} [obligation:{obligation.kind}] "
                    f"{main_method} ({round(elapsed_ms)}ms)"
                )
            if not ok:
                continue
            final = combined_seed + [main_method]
            logger.finish(True, final, 1, use_calls_count())
            if save_dir:
                name = slugify_goal(goal) + ".thy"
                write_theory_file(os.path.join(save_dir, name), build_theory(final, False, None))
            if trace:
                print(color(use_color, "green", "✔ PROVED"))
            return {
                "goal": goal, "success": True, "steps": final, "depth": 1,
                "use_calls": use_calls_count(), "elapsed_s": logger.elapsed_s,
                "model": display_model, "origin": "definition-obligation",
                "obligation_kind": obligation.kind, "strategy_rank": index,
            }

    # Give the deterministic planner pre-pass its Sledgehammer opportunity
    # before broader one-line methods consume the remaining short budget.
    if deterministic_only and use_sledge and time_left_s() > 3.0:
        suggestions: List[str] = []
        try:
            suggestions = sledgehammer_finishers(
                isabelle,
                session_id,
                seed_steps,
                timeout_s=max(1, int(min(float(sledge_timeout), time_left_s() - 2.0))),
                limit=4,
            )
        except Exception as exc:
            if trace:
                print(color(use_color, "yellow", f"  deterministic Sledgehammer failed: {exc}"))
        for fin in suggestions:
            if time_left_s() <= 1.0:
                break
            try:
                ok, elapsed_ms = try_finish(
                    isabelle,
                    session_id,
                    seed_steps,
                    fin,
                    timeout_s=max(1, int(time_left_s())),
                )
            except Exception:
                ok, elapsed_ms = False, 0.0
            if trace:
                tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
                print(f"  finish {tag} [sledge] {fin}  ({round(elapsed_ms)}ms)")
            if ok:
                final = seed_steps + [fin]
                logger.finish(True, final, 1, use_calls_count())
                return {
                    "goal": goal, "success": True, "steps": final,
                    "depth": 1, "use_calls": use_calls_count(),
                    "elapsed_s": logger.elapsed_s, "model": display_model,
                    "origin": "sledge",
                }

    # Tier 2: broader generic methods, but retain a hard reserve for actual
    # state expansion.  This prevents a row of two-second failures from
    # starving the only candidates capable of changing the proof state.
    broad_finishers = ("by simp_all", "by clarsimp", "by fastforce", "by blast")
    expansion_reserve = max(6.0, min(14.0, budget * 0.30))
    for fin in broad_finishers:
        if time_left_s() <= expansion_reserve:
            break
        attempted_finishers.add((tuple(seed_steps), fin))
        try:
            ok, elapsed_ms = try_finish(
                isabelle,
                session_id,
                seed_steps,
                fin,
                timeout_s=max(3, int(min(6.0, time_left_s() - expansion_reserve + 1.0))),
            )
        except Exception:
            ok, elapsed_ms = False, 0.0
        if trace:
            tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
            print(f"  finish {tag} [automation] {fin}  ({round(elapsed_ms)}ms)")
        if ok:
            final = seed_steps + [fin]
            logger.finish(True, final, 1, use_calls_count())
            if trace:
                print(color(use_color, "green", "✔ PROVED"))
            return {
                "goal": goal, "success": True, "steps": final, "depth": 1,
                "use_calls": use_calls_count(), "elapsed_s": logger.elapsed_s,
                "model": display_model, "origin": "automation",
            }

    # ``auto`` planner mode uses this bounded deterministic portfolio as a
    # genuine pre-pass.  Sledgehammer is allowed because it is an Isabelle
    # search tool, but no reranker, premise encoder or language model is
    # constructed.  A miss therefore returns quickly to outline generation.
    if deterministic_only:
        logger.finish(False, seed_steps, 0, use_calls_count())
        return {
            "goal": goal, "success": False, "steps": seed_steps, "depth": 0,
            "use_calls": use_calls_count(), "elapsed_s": logger.elapsed_s,
            "model": display_model, "timeout": time_left_s() <= 0.0,
            "origin": "deterministic-prepass",
        }

    reranker = None
    if enable_reranker and os.environ.get("RERANKER_OFF", "0") not in ("1", "true", "True"):
        reranker = Reranker()

    # ----- optional: build a lightweight context window + retrieval index -----
    ctx_win: Optional[ContextWindow] = None
    prem_idx: Optional[PremisesIndex] = None
    if CFG.PROVER_CONTEXT_ENABLE and CFG.PROVER_CONTEXT_FILES:
        try:
            ctx_win = ContextWindow(window_size=CFG.PROVER_CONTEXT_WINDOW)
            for _pth in CFG.PROVER_CONTEXT_FILES:
                try:
                    ctx_win.ingest_theory(_pth)
                except Exception:
                    pass
        except Exception:
            ctx_win = None
    if CFG.PREMISES_ENABLE:
        try:
            prem_idx = PremisesIndex()
            # NEW: try to load trained premise models (bi-encoder + cross-encoder)
            try:
                prem_idx.try_load_encoder_from_env()
            except Exception:
                pass
            try:
                prem_idx.try_load_reranker_from_env()
            except Exception:
                pass
            if ctx_win is not None:
                prem_idx.add_many(ctx_win.seed_pairs())
            prem_idx.finalize()
        except Exception:
            prem_idx = None

    # Seed the first beam with an optional initial state hint (used by planner after print_state)
    seed_hint = (initial_state_hint or "").strip()
    guided_steps = definition_guided_steps(
        goal, theory_preamble, seed_hint, limit=max(2, min(6, beam_w * 2))
    )
    guided_steps.extend(
        generic_structural_steps(
            goal, seed_hint, limit=max(2, min(4, beam_w * 2))
        )
    )
    guided_steps = list(dict.fromkeys(guided_steps))
    beam: List[Tuple[int, List[str], str, Optional[int]]] = [(9999, seed_steps, seed_hint, None)]
    visited_by_depth = defaultdict(set)

    last_best_n: Optional[int] = None
    stagnant_depths = 0

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

        # Complete branches created by a structural step before doing premise
        # retrieval, Sledgehammer, or model inference.  This is the multi-step
        # counterpart of the Tier-1 atomic strategy portfolio above.
        if depth > 0:
            for _, branch_steps, _, _ in beam:
                for fin in ("by simp_all", "by auto"):
                    if time_left_s() <= 2.0:
                        break
                    if (tuple(branch_steps), fin) in attempted_finishers:
                        continue
                    attempted_finishers.add((tuple(branch_steps), fin))
                    try:
                        ok, elapsed_ms = try_finish(
                            isabelle, session_id, branch_steps, fin,
                            timeout_s=max(3, int(min(7.0, time_left_s()))),
                        )
                    except Exception:
                        ok, elapsed_ms = False, 0.0
                    if trace:
                        tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
                        print(f"  finish {tag} [structural] {fin}  ({round(elapsed_ms)}ms)")
                    if ok:
                        final = branch_steps + [fin]
                        logger.finish(True, final, depth_reached, use_calls_count())
                        if trace:
                            print(color(use_color, "green", "✔ PROVED"))
                        return {
                            "goal": goal, "success": True, "steps": final,
                            "depth": depth_reached, "use_calls": use_calls_count(),
                            "elapsed_s": logger.elapsed_s, "model": display_model,
                            "origin": "definition-step",
                        }

        valid_ns = [n for _, _, _, n in beam if isinstance(n, int)]
        best_n_now = min(valid_ns) if valid_ns else None
        if best_n_now is not None:
            if last_best_n is None or best_n_now < last_best_n:
                stagnant_depths = 0
            else:
                stagnant_depths += 1
            last_best_n = best_n_now
        variant_burst = do_variants and (stagnant_depths >= 2)
        steps_temp  = min(0.9, 0.5 + 0.10 * stagnant_depths)
        finish_temp = min(0.6, 0.2 + 0.05 * stagnant_depths)

        facts_for_depth: List[str] = []
        if beam and beam[0][2]:
            try:
                facts_for_depth = mine_facts_prioritized(isabelle, session_id, beam[0][2], limit=facts_limit)
                if trace and facts_for_depth:
                    print(color(use_color, "yellow", f"Mined facts: {facts_for_depth}"))
            except Exception as e:
                if trace: print(color(use_color, "yellow", f"Facts mining error ignored: {e}"))

        # new: premise retrieval (global or seeded by context window)
        # We keep it conservative: use it as *additional hints* for the LLM/ranker
        # and do not explode candidates.
        premise_scores: Dict[str, Tuple[float, float]] = {}
        picks_ids_compact: List[str] = []  # <-- for training supervision
        pool_feats: Dict[str, float] = {
            "premise_cosine_top1": 0.0,
            "premise_cosine_topk_mean": 0.0,
            "premise_rerank_top1": 0.0,
            "premise_rerank_topk_mean": 0.0,
            "n_premises": 0.0,
        }        
        if CFG.PREMISES_ENABLE and prem_idx is not None:
            try:
                state_proxy = beam[0][2] or ""
                query_text = f"{goal}  {state_proxy}".strip()
                picks = prem_idx.select(
                    query_text,
                    k_select=max(64, min(1024, CFG.PREMISES_K_SELECT)),
                    k_rerank=max(16, min(256, CFG.PREMISES_K_RERANK)),
                    boost_ids=(ctx_win.facts_for(CFG.PROVER_CONTEXT_FILES[0], 10**12)
                               if (ctx_win and CFG.PROVER_CONTEXT_FILES) else None),
                )
                # real pool metrics from actual retrieval picks
                pool_feats = selection_features(picks)                
                premise_scores = build_score_map(picks)
                picks_ids_compact = [fid for fid, _s, _r in picks[:128]]
                # Convert fact_ids like 'File.thy:lemma_name:i' → lemma_name
                sel_names = []
                for fid, _s, _r in picks[:max(FACTS_LIMIT, facts_limit)]:
                    parts = fid.split(":")
                    sel_names.append(parts[1] if len(parts) >= 2 else fid)
                if sel_names:
                    # prepend retrieved names, then mined ones; keep unique and cap to limit
                    seen = set()
                    merged: List[str] = []
                    for nm in (sel_names + facts_for_depth):
                        if nm not in seen:
                            seen.add(nm)
                            merged.append(nm)
                    facts_for_depth = merged[:facts_limit]
                    if trace:
                        print(color(use_color, "yellow", f"Premises (retrieved): {sel_names[:min(8,len(sel_names))]}"))
            except Exception as e:
                if trace: print(color(use_color, "yellow", f"Premises retrieval error ignored: {e}"))                

        sledge_sugs: List[str] = []
        if use_sledge and depth % max(1, sledge_every) == 0 and beam and not (depth == 0 and guided_steps):
            try:
                sledge_budget = max(1, int(min(float(sledge_timeout), time_left_s())))
                sledge_sugs = sledgehammer_finishers(
                    isabelle, session_id, beam[0][1],
                    timeout_s=sledge_budget, limit=5,
                )
                if trace:
                    if sledge_sugs:
                        print(color(use_color, "yellow", f"Sledgehammer finishers: {sledge_sugs}"))
                    else:
                        print(color(use_color, "yellow", "Sledgehammer yielded no finishers at this depth"))
            except Exception as e:
                if trace: print(color(use_color, "yellow", f" Sledgehammer error ignored: {e}"))
        elif trace and use_sledge:
            why = []
            if not beam: why.append("no beam")
            if sledge_every > 1 and (depth % sledge_every) != 0: why.append(f"depth {depth} not multiple of sledge_every={sledge_every}")
            if not why: why.append("unknown condition")
            print(color(use_color, "yellow", f"Sledgehammer skipped ({', '.join(why)})"))                

        # ---- Try to finish ----
        for j, (_, steps, state_hint, _) in enumerate(beam):
            mined = mine_lemmas_from_state(isabelle, session_id, state_hint, max_lemmas=hint_lemmas) if state_hint else []
            # allow 'done' only when there are zero subgoals
            _n_now = parse_subgoals(state_hint) if state_hint else None
            _allow_done = bool(_n_now == 0)            
            finishers_llm = [] if (depth == 0 and guided_steps) else propose_finishers(
                model_list, goal, steps, state_hint, mined, hint_lemmas,
                facts=facts_for_depth, temp=finish_temp, reranker=reranker, allow_done=_allow_done,
                premise_scores=premise_scores, premise_pool=pool_feats,
                timeout_s=max(1, int(min(15.0, time_left_s() * 0.35))),
            )

            # Build finishers with explicit origin tags so we can print who proposed what.
            finishers_with_origin: List[Tuple[str, str]] = []
            # sledge proposals get priority (only for the first beam entry as before)
            if j == 0 and sledge_sugs:
                for s in sledge_sugs:
                    if s not in [f for f, _ in finishers_with_origin]:
                        finishers_with_origin.append((s, "sledge"))
            # LLm finishers (only add if not already present)
            for f in finishers_llm:
                if (tuple(steps), f) not in attempted_finishers and f not in [ff for ff, _ in finishers_with_origin]:
                    finishers_with_origin.append((f, "llm"))

            if trace and finishers_with_origin:
                pretty = ", ".join([f"{o}:{fin}" for fin, o in finishers_with_origin])
                print(color(use_color, "blue", f"Finishers (origin): [{pretty}]"))

            for fin, origin in finishers_with_origin:
                if time_left_s() <= 0: break
                attempted_finishers.add((tuple(steps), fin))
                # Enforce wall-clock budget at the Isabelle boundary to avoid false positives
                # where Isabelle keeps running long after our global timeout.
                per_call_timeout = max(1, int(min(time_left_s(), float(budget))))
                ok, elapsed_ms = try_finish(isabelle, session_id, steps, fin, timeout_s=per_call_timeout)
                # If sledge/finisher solved before retrieval ran, lazily compute pool metrics now.
                if CFG.PREMISES_ENABLE and prem_idx is not None and (not pool_feats or float(pool_feats.get("n_premises", 0.0)) == 0.0):
                    try:
                        state_proxy = beam[0][2] or ""
                        query = f"{goal}  {state_proxy}".strip()
                        picks_lazy = prem_idx.select(
                            query,
                            k_select=max(64, min(1024, CFG.PREMISES_K_SELECT)),
                            k_rerank=max(16, min(256, CFG.PREMISES_K_RERANK)),
                            boost_ids=(ctx_win.facts_for(CFG.PROVER_CONTEXT_FILES[0], 10**9)
                                       if (ctx_win and CFG.PROVER_CONTEXT_FILES) else None),
                        )
                        pool_feats = selection_features(picks_lazy)
                        premise_scores = build_score_map(picks_lazy)
                        picks_ids_compact = [fid for fid, _s, _r in picks_lazy[:128]]
                    except Exception:
                        pass
                # per-candidate metrics (real if names match; else zeros)
                fin_facts = extract_candidate_facts(fin)
                cf = cand_features(fin_facts, premise_scores) if premise_scores else {
                    "cand_cos_mean": 0.0, "cand_cos_max": 0.0, "cand_rerank_mean": 0.0, "cand_hit_topk": 0.0, "cand_n_facts": 0.0
                }
                logger.log_attempt("finish", steps, fin, ok, 0 if ok else None, False, elapsed_ms, depth_reached,
                    extra={
                        "origin": origin,
                        # supervision for premise training
                        "retrieval_picks": picks_ids_compact,
                        "cand_facts": fin_facts,                        
                        # Pool metrics (real numbers if PREMISES_ENABLE)
                        "premise_cosine_top1": float(pool_feats.get("premise_cosine_top1", 0.0)),
                        "premise_cosine_topk_mean": float(pool_feats.get("premise_cosine_topk_mean", 0.0)),
                        "premise_rerank_top1": float(pool_feats.get("premise_rerank_top1", 0.0)),
                        "premise_rerank_topk_mean": float(pool_feats.get("premise_rerank_topk_mean", 0.0)),
                        "n_premises": float(pool_feats.get("n_premises", 0.0)),
                        # Per-candidate (real if facts referenced & matched)
                        "cand_cos_mean": float(cf.get("cand_cos_mean", 0.0)),
                        "cand_cos_max": float(cf.get("cand_cos_max", 0.0)),
                        "cand_rerank_mean": float(cf.get("cand_rerank_mean", 0.0)),
                        "cand_hit_topk": float(cf.get("cand_hit_topk", 0.0)),
                        "cand_n_facts": float(cf.get("cand_n_facts", 0.0)),
                    })
                if trace:
                    tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
                    print(f"  finish {tag} [{origin}] {fin}  ({round(elapsed_ms)}ms)")
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
        for _, steps, state_hint, prev_n in beam:
            if time_left_s() <= 0: break
            # Typed induction/case proposals are cheap and often constitute the
            # essential first move, so offer them at the root as well as after
            # search stagnates.
            extra_cands = []
            if depth == 0:
                extra_cands.extend(guided_steps)
            if do_variants and (depth == 0 or variant_burst):
                extra_cands.extend(variant_step_templates(state_hint))
            # At the root, execute Isabelle-derived typed structural moves
            # before another model call.  This guarantees that induction/case
            # search receives a real budget even with a high-latency backend.
            llm_cands = [] if (depth == 0 and extra_cands) else propose_steps(
                model_list, goal, steps, state_hint,
                facts=facts_for_depth, reranker=reranker,
                depth=depth, temp=steps_temp,
                premise_scores=premise_scores,
                premise_pool=pool_feats,
                timeout_s=max(1, int(time_left_s())),
            )

            # Build list of (cand, origin) where origin in {"variant","llm"}
            cands_with_origin: List[Tuple[str, str]] = []
            for c in extra_cands:
                cands_with_origin.append((c, "variant"))
            for c in llm_cands:
                cands_with_origin.append((c, "llm"))

            # Deduplicate preserving first-seen origin (so variant wins if first)
            seen_c, dedup_pairs = set(), []
            for c, origin in cands_with_origin:
                if c not in seen_c:
                    seen_c.add(c)
                    dedup_pairs.append((c, origin))
            if trace:
                pretty = ", ".join([f"{o}:{c}" for c, o in dedup_pairs])
                print(color(use_color, "blue", f"Proposals (origin): [{pretty}]"))
            if not dedup_pairs: continue

            state_fp_before = state_fingerprint(state_hint or "")
            pid_seed = f"{logger.run_id}|d={depth_reached}|{state_fp_before}|{len(steps)}|{steps[-2:] if steps else ''}"
            proposal_id = hashlib.sha1(pid_seed.encode('utf-8')).hexdigest()[:12]
            best_local: List[Tuple[int, List[str], str, Optional[int]]] = []
            _flags = flags_from_goal(goal, state_hint or "")
            _should_boolean_precheck = bool(_flags.get("has_q", 0) or _flags.get("is_bool", 0))
            for k, (c, origin) in enumerate(dedup_pairs):
                if time_left_s() <= 0: break
                pruned = False
                if use_qc and _should_boolean_precheck and depth % max(1, qc_every) == 0:
                    try:
                        if precheck_quickcheck_refutes(isabelle, session_id, steps + [c], timeout_s=qc_timeout):
                            pruned = True
                            if trace: print(color(use_color, "dim", f"  step  ·  [{origin}] {c}  [pruned by quickcheck]"))
                    except Exception as e:
                        if trace: print(color(use_color, "yellow", f"  quickcheck error ignored: {e}"))
                if not pruned and use_np and _should_boolean_precheck and depth % max(1, np_every) == 0:
                    try:
                        if precheck_nitpick_refutes(isabelle, session_id, steps + [c], timeout_s=np_timeout):
                            pruned = True
                            if trace: print(color(use_color, "dim", f"  step  ·  [{origin}] {c}  [pruned by nitpick]"))
                    except Exception as e:
                        if trace: print(color(use_color, "yellow", f"  nitpick error ignored: {e}"))
                if pruned:
                    logger.log_attempt(
                        "expand", steps, c, False, None, False, 0.0, depth_reached,
                        subgoals_before=prev_n,
                    extra={
                           "proposal_id": proposal_id, "proposal_k": int(k),
                           "state_fp_before": state_fp_before,
                           "origin": origin,
                           # supervision for premise training
                           "retrieval_picks": picks_ids_compact,
                           "cand_facts": extract_candidate_facts(c),                           
                           # real pool metrics from retrieval picks
                           "premise_cosine_top1": float(pool_feats.get("premise_cosine_top1", 0.0)),
                           "premise_cosine_topk_mean": float(pool_feats.get("premise_cosine_topk_mean", 0.0)),
                           "premise_rerank_top1": float(pool_feats.get("premise_rerank_top1", 0.0)),
                           "premise_rerank_topk_mean": float(pool_feats.get("premise_rerank_topk_mean", 0.0)),
                           "n_premises": float(pool_feats.get("n_premises", 0.0)),
                           # NEW per-candidate metrics
                           **(lambda cf: {
                               "cand_cos_mean": cf.get("cand_cos_mean", 0.0),
                               "cand_cos_max": cf.get("cand_cos_max", 0.0),
                               "cand_rerank_mean": cf.get("cand_rerank_mean", 0.0),
                               "cand_hit_topk": cf.get("cand_hit_topk", 0.0),
                               "cand_n_facts": cf.get("cand_n_facts", 0.0),
                           })(cand_features(extract_candidate_facts(c), premise_scores))
                    }
                    )
                    continue
                # A structural state transition deserves a real verifier
                # window, but no single failed candidate may consume the
                # remaining global budget.
                candidate_cap = float(max(4, variant_timeout + 1)) if origin == "variant" else 10.0
                per_call_timeout = max(2, int(min(time_left_s(), candidate_cap)))
                ok, n_sub, hint, cache_hit, elapsed_ms = try_step_cached(
                    isabelle, session_id, steps, c, timeout_s=per_call_timeout
                )
                logger.log_attempt(
                    "expand", steps, c, ok, n_sub, cache_hit, elapsed_ms, depth_reached,
                    subgoals_before=prev_n,
                    extra={
                        "proposal_id": proposal_id,
                        "proposal_k": int(k),
                        "state_fp_before": state_fp_before,
                        "state_fp_after": state_fingerprint(hint or "") if ok else None,
                        "origin": origin,
                        # supervision for premise training
                        "retrieval_picks": picks_ids_compact,
                        "cand_facts": extract_candidate_facts(c),                        
                        # real pool metrics from retrieval picks
                        "premise_cosine_top1": float(pool_feats.get("premise_cosine_top1", 0.0)),
                        "premise_cosine_topk_mean": float(pool_feats.get("premise_cosine_topk_mean", 0.0)),
                        "premise_rerank_top1": float(pool_feats.get("premise_rerank_top1", 0.0)),
                        "premise_rerank_topk_mean": float(pool_feats.get("premise_rerank_topk_mean", 0.0)),
                        "n_premises": float(pool_feats.get("n_premises", 0.0)),
                        # NEW per-candidate metrics
                        **(lambda cf: {
                            "cand_cos_mean": cf.get("cand_cos_mean", 0.0),
                            "cand_cos_max": cf.get("cand_cos_max", 0.0),
                            "cand_rerank_mean": cf.get("cand_rerank_mean", 0.0),
                            "cand_hit_topk": cf.get("cand_hit_topk", 0.0),
                            "cand_n_facts": cf.get("cand_n_facts", 0.0),
                        })(cand_features(extract_candidate_facts(c), premise_scores)),
                    }
                )
                if trace:
                    tag = color(use_color, "green", "✓") if ok else color(use_color, "red", "×")
                    extra = f" n_sub={n_sub}" if n_sub is not None else ""
                    cache = color(use_color, "dim", " [cache]") if cache_hit else ""
                    print(f"  step  {tag} [{origin}] {c}{extra} ({round(elapsed_ms)}ms){cache}")
                if not ok: continue
                score = n_sub if n_sub is not None else 9999
                best_local.append((score, steps + [c], hint, n_sub))

            # Merge best_local into new_beam (keep top-beam_w by score, tie-break by shorter script)
            if best_local:
                best_local.sort(key=lambda t: (t[0], len("\n".join(t[1]))))
                for tpl in best_local[:max(1, beam_w)]:
                    new_beam.append(tpl)

        if not new_beam:
            # Stuck — return current best partial
            best = min(beam, key=lambda t: (t[0], len("\n".join(t[1]))))
            logger.finish(False, best[1], depth_reached, use_calls_count())
            if save_dir:
                name = slugify_goal(goal) + ".thy.partial"
                from .isabelle_api import build_theory as _bt
                write_theory_file(os.path.join(save_dir, name), _bt(best[1] + ["sorry"], False, None))
            return {"goal": goal, "success": False, "steps": best[1], "depth": depth_reached,
                    "use_calls": use_calls_count(), "elapsed_s": logger.elapsed_s, "model": display_model}

        # Keep only distinct prefixes per depth (avoid revisiting same state)
        next_beam: List[Tuple[int, List[str], str, Optional[int]]] = []
        seen_fp = set()
        for n_sub, steps, hint, _ in sorted(new_beam, key=lambda t: (t[0], len("\n".join(t[1]))))[:max(1, beam_w)]:
            fp = state_fingerprint(hint or "")
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            next_beam.append((n_sub if n_sub is not None else 9999, steps, hint or "", n_sub))
        beam = next_beam

    # Ran out of depth; return best partial
    best = min(beam, key=lambda t: (t[0], len("\n".join(t[1]))))
    logger.finish(False, best[1], MAX_DEPTH, use_calls_count())
    if save_dir:
        name = slugify_goal(goal) + ".thy.partial"
        from .isabelle_api import build_theory as _bt
        write_theory_file(os.path.join(save_dir, name), _bt(best[1] + ["sorry"], False, None))
    return {"goal": goal, "success": False, "steps": best[1], "depth": MAX_DEPTH,
            "use_calls": use_calls_count(), "elapsed_s": logger.elapsed_s, "model": display_model}
