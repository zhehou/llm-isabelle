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
    build_theory, run_theory, finished_ok, last_print_state_block,
    use_calls_count, last_call_timed_out,
)

from .tactics import (
    mine_lemmas_from_state, mine_facts_prioritized,
    sledgehammer_finishers,
    precheck_quickcheck_refutes, precheck_nitpick_refutes,
    try_structured_variants, suggest_continuations,
    variant_step_templates,
)

from .minimize import minimize_proof
from .features import flags_from_goal
from .ranker import Reranker
# new: premise retrieval & optional file-aware context
from .premises import PremisesIndex, build_score_map, cand_features, selection_features
from .heuristics import extract_candidate_facts
from .context import ContextWindow

_result_cache: Dict[Tuple[Tuple[str, ...], str], Tuple[bool, Optional[int], str]] = {}

_GLOBAL_CACHE_MAX = int(os.getenv("GLOBAL_STEP_CACHE_MAX", "8192"))
_global_result_cache: "OrderedDict[tuple, tuple]" = OrderedDict()

_GOAL_LINE = re.compile(r'lemma\s+"(.*)"')

def _global_cache_key(steps: List[str], cand: str) -> tuple:
    lemma_line = steps[0] if steps else ""
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
               initial_state_hint: Optional[str] = None  # <-- NEW (optional, backward-compatible)
               ) -> Dict[str, Any]:

    budget = timeout if timeout is not None else 10

    reranker = None
    if enable_reranker and os.environ.get("RERANKER_OFF", "0") not in ("1", "true", "True"):
        reranker = Reranker()

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

    start_t = time.monotonic()
    def time_left_s() -> float: return budget - (time.monotonic() - start_t)

    seed_steps = [f'lemma "{goal}"']
    # Seed the first beam with an optional initial state hint (used by planner after print_state)
    seed_hint = (initial_state_hint or "").strip()
    beam: List[Tuple[int, List[str], str, Optional[int]]] = [(9999, seed_steps, seed_hint, None)]
    visited_by_depth = defaultdict(set)

    last_best_n: Optional[int] = None
    stagnant_depths = 0

    if trace:
        print(color(use_color, "bold", f"\n▶ Goal: {goal}"))
        print(color(use_color, "gray", f"Models: {display_model} | Beam={beam_w} | MaxDepth={max_depth} | Timeout={budget}s"))

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
        if use_sledge and depth % max(1, sledge_every) == 0 and beam:
            try:
                sledge_sugs = sledgehammer_finishers(isabelle, session_id, beam[0][1], timeout_s=sledge_timeout, limit=5)
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
            finishers_llm = propose_finishers(
                model_list, goal, steps, state_hint, mined, hint_lemmas,
                facts=facts_for_depth, temp=finish_temp, reranker=reranker, allow_done=_allow_done,
                premise_scores=premise_scores, premise_pool=pool_feats
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
                if f not in [ff for ff, _ in finishers_with_origin]:
                    finishers_with_origin.append((f, "llm"))

            if trace and finishers_with_origin:
                pretty = ", ".join([f"{o}:{fin}" for fin, o in finishers_with_origin])
                print(color(use_color, "blue", f"Finishers (origin): [{pretty}]"))

            for fin, origin in finishers_with_origin:
                if time_left_s() <= 0: break
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
            extra_cands = variant_step_templates(state_hint) if variant_burst else []
            llm_cands = propose_steps(model_list, goal, steps, state_hint,
                                      facts=facts_for_depth, reranker=reranker,
                                      depth=depth, temp=steps_temp,
                                      premise_scores=premise_scores,
                                      premise_pool=pool_feats)

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
                per_call_timeout = max(1, int(min(time_left_s(), float(budget))))
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
