# prover/llm.py
import requests
from typing import List
from .config import OLLAMA_HOST, TEMP, TOP_P, TIMEOUT_S, NUM_CANDIDATES, OLLAMA_NUM_PREDICT
from .prompts import SYSTEM_STEPS, SYSTEM_FINISH, USER_TEMPLATE, parse_ollama_lines

def _ollama_generate(system_prompt: str, user_prompt: str, model: str, *, temperature: float | None = None) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}",
        "temperature": float(temperature if temperature is not None else TEMP),
        "top_p": float(TOP_P),
        "num_predict": int(OLLAMA_NUM_PREDICT),
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=int(TIMEOUT_S))
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        return f"__ERROR__ {e}"

def merge_candidates(list_of_lists: List[List[str]], limit: int) -> List[str]:
    merged, seen = [], set()
    idx = 0
    while len(merged) < limit and any(idx < len(lst) for lst in list_of_lists):
        for lst in list_of_lists:
            if idx < len(lst):
                cand = lst[idx]
                if cand not in seen:
                    seen.add(cand); merged.append(cand)
                    if len(merged) >= limit: break
        idx += 1
    return merged

def propose_steps(models: List[str], goal: str, steps_so_far: List[str], state_hint: str = "",
                  facts: List[str] | None = None, reranker=None, depth:int=0, *, temp: float | None = None) -> List[str]:
    user = USER_TEMPLATE.format(
        goal=goal,
        steps="\n".join(steps_so_far) or "(none)",
        state_hint=(state_hint.strip() or "(none)"),
        facts=("\n".join(facts) if facts else "(none)")
    )
    all_lists: List[List[str]] = []
    for m in models:
        raw = _ollama_generate(SYSTEM_STEPS, user, m, temperature=temp)
        cands = parse_ollama_lines(raw, ["apply ", "apply("], NUM_CANDIDATES)
        all_lists.append(cands)
    from .heuristics import augment_with_facts_for_steps, rank_candidates
    merged = merge_candidates(all_lists, NUM_CANDIDATES) or \
             ["apply simp", "apply auto", "apply clarsimp", "apply (simp only: foo_def)", "apply (simp add: foo_def)", "apply (induction xs)", "apply (cases xs)"]
    # If no facts provided, synthesize a couple likely *_def guesses from goal words.
    if not facts:
        import re
        words = re.findall(r"[A-Za-z][A-Za-z0-9_']{2,}", (goal + " " + state_hint))
        stems = [w for w in words if not w[0].isupper()]  # prefer constants/functions
        stems = list(dict.fromkeys(stems))[:3]
        for s in stems:
            merged.insert(0, f"apply (simp add: {s}_def)")
            merged.insert(0, f"apply (simp only: {s}_def)")    
            merged.insert(0, f"apply (auto simp add: {s}_def)")  
    if facts:
        merged = augment_with_facts_for_steps(merged, facts)
    return rank_candidates(merged, goal, state_hint, facts, reranker=reranker, depth=depth)

def propose_finishers(models: List[str], goal: str, steps_so_far: List[str], state_hint: str,
                      mined_lemmas: List[str], hint_lemmas_limit: int,
                      facts: List[str] | None = None, *, temp: float | None = None, reranker=None) -> List[str]:
    user = USER_TEMPLATE.format(
        goal=goal,
        steps="\n".join(steps_so_far) or "(none)",
        state_hint=(state_hint.strip() or "(none)"),
        facts=("\n".join(facts) if facts else "(none)")
    )
    base_lists: List[List[str]] = []
    for m in models:
        raw = _ollama_generate(SYSTEM_FINISH, user, m, temperature=temp)
        base = parse_ollama_lines(raw, ["done", "by "], max(3, min(NUM_CANDIDATES, 8)))
        base_lists.append(base)
    base_merged = merge_candidates(base_lists, max(3, min(NUM_CANDIDATES, 8))) or \
                  ["done", "by simp", "by auto", "by clarsimp", "by (simp only: foo_def)", "by (simp add: foo_def)", "by arith", "by presburger", "by fastforce", "by blast", "by meson", "by (metis)"]
    from .heuristics import suggest_common_lemmas, mk_finisher_variants, augment_with_facts_for_finishers
    static_hints = suggest_common_lemmas(state_hint)
    dyn_hints = mined_lemmas[:max(0, hint_lemmas_limit)]
    lemma_finishers = mk_finisher_variants(static_hints + dyn_hints)
    with_facts = augment_with_facts_for_finishers(base_merged, facts or [], cap=8)
    combined, seen = [], set()
    for x in with_facts + lemma_finishers + base_merged:
        if x not in seen:
            seen.add(x); combined.append(x)
        if len(combined) >= 8: break
    order = ["done", "by simp", "by clarsimp", "by auto", "by fastforce", "by blast", "by arith", "by presburger", "by (simp", "by (auto", "by (metis", "by (meson"]
    ranked = sorted(combined, key=lambda cmd: next((i for i,k in enumerate(order) if cmd.startswith(k)), len(order)))
    if reranker and getattr(reranker, "available", lambda: False)():
        from .heuristics import live_features_for
        def rscore(cmd: str) -> float:
            try: return float(reranker.score(live_features_for(cmd, goal, state_hint, depth=0)))
            except Exception: return 0.5
        # sort by model score desc, keep heuristic order as stable tiebreaker
        ranked = sorted(ranked, key=lambda c: (-rscore(c), ranked.index(c)))
    return ranked