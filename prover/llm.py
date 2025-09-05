# prover/llm.py
from __future__ import annotations

import os
import json
import shutil
import subprocess
from typing import List, Optional

import requests

from .config import (
    OLLAMA_HOST, TEMP, TOP_P, TIMEOUT_S, NUM_CANDIDATES, OLLAMA_NUM_PREDICT
)
from .prompts import SYSTEM_STEPS, SYSTEM_FINISH, USER_TEMPLATE, parse_ollama_lines

# -------- debug logging toggle --------
_VERBOSE = os.getenv("LLM_DEBUG", "").strip().lower() not in ("", "0", "false", "no")

def _log(msg: str) -> None:
    if _VERBOSE:
        print(f"[llm] {msg}", flush=True)

def detect_backend_for_model(model: str) -> str:
    """
    Return a human-readable description of which backend would be used for `model`.
    This function does not perform any network calls.
    """
    if model.startswith("gemini:"):
        return f"gemini-cli model={model.split(':',1)[1]}"
    if model.startswith("hf:"):
        repo = model.split(":", 1)[1]
        force_local = os.getenv("HF_MODE", "").strip().lower() == "local"
        has_token = bool(os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        if has_token and not force_local:
            return f"hf-inference-api repo={repo}"
        return f"hf-local repo={repo}"
    if model.startswith("ollama:"):
        return f"ollama model={model.split(':',1)[1]}"
    return f"ollama (unprefixed) model={model}"


# ---------------------------
# Backend: common helpers
# ---------------------------

def _join_prompts(system_prompt: str, user_prompt: str) -> str:
    """Unify prompt formatting so all backends see the same text."""
    return f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}"


# ---------------------------
# Backend: Ollama
# ---------------------------

def _ollama_generate(system_prompt: str, user_prompt: str, model: str, *,
                     temperature: float | None = None) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": _join_prompts(system_prompt, user_prompt),
        "temperature": float(temperature if temperature is not None else TEMP),
        "top_p": float(TOP_P),
        "num_predict": int(OLLAMA_NUM_PREDICT),
        "stream": False,
    }
    try:
        _log(f"call: ollama url={url} model={model}")
        resp = requests.post(url, json=payload, timeout=int(TIMEOUT_S))
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        err = f"__ERROR__ {e}"
        _log(f"error: ollama model={model}: {e}")
        return err


# ---------------------------
# Backend: Gemini CLI
# ---------------------------

def _gemini_cli_generate(system_prompt: str, user_prompt: str, model: str) -> str:
    """
    Uses the standalone `gemini` CLI.

    Env knobs:
      - GEMINI_CLI_BIN (default: "gemini")
    """
    bin_path = os.getenv("GEMINI_CLI_BIN", "gemini")
    if shutil.which(bin_path) is None:
        msg = "gemini CLI not found; set GEMINI_CLI_BIN or install https://github.com/google-gemini/gemini-cli"
        _log(f"error: {msg}")
        return f"__ERROR__ {msg}"

    prompt = _join_prompts(system_prompt, user_prompt)

    try:
        _log(f"call: gemini-cli bin={bin_path} model={model}")
        proc = subprocess.run(
            [bin_path, "-m", model, "-p", prompt],
            input=b"",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(TIMEOUT_S),
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).decode("utf-8", "ignore").strip()
            _log(f"error: gemini-cli rc={proc.returncode} {msg}")
            return f"__ERROR__ gemini CLI failed ({proc.returncode}): {msg}"
        return proc.stdout.decode("utf-8", "ignore")
    except Exception as e:
        _log(f"error: gemini-cli: {e}")
        return f"__ERROR__ {e}"


# ---------------------------
# Backend: Hugging Face
#   - Prefer Inference API if token available
#   - Fallback to local transformers if installed and HF_MODE=local
# ---------------------------

def _hf_inference_api_generate(system_prompt: str, user_prompt: str, repo_id: str,
                               *, temperature: float | None = None) -> str:
    """
    HF Inference API call. Requires HF_API_TOKEN or HUGGINGFACEHUB_API_TOKEN.
    Env:
      - HF_API_TOKEN / HUGGINGFACEHUB_API_TOKEN
      - HF_API_BASE (default https://api-inference.huggingface.co/models)
    """
    token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    base = os.getenv("HF_API_BASE", "https://api-inference.huggingface.co/models").rstrip("/")
    url = f"{base}/{repo_id}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {
        "inputs": _join_prompts(system_prompt, user_prompt),
        "parameters": {
            "max_new_tokens": int(OLLAMA_NUM_PREDICT),
            "temperature": float(temperature if temperature is not None else TEMP),
            "top_p": float(TOP_P),
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    try:
        _log(f"call: hf-inference-api repo={repo_id} url={url} token={'set' if token else 'missing'}")
        resp = requests.post(url, headers=headers, json=payload, timeout=int(TIMEOUT_S))
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        _log(f"error: hf-inference-api repo={repo_id}: {e}")
        return f"__ERROR__ {e}"


def _hf_local_generate(system_prompt: str, user_prompt: str, repo_id: str,
                       *, temperature: float | None = None) -> str:
    """
    Local transformers generation. Only used if HF_MODE=local.
    """
    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        msg = "transformers not installed; pip install transformers accelerate"
        _log(f"error: hf-local repo={repo_id}: {msg}")
        return f"__ERROR__ {msg}"

    try:
        _log(f"call: hf-local repo={repo_id} (pipeline load)")
        pipe = pipeline(
            "text-generation",
            model=repo_id,
            device_map="auto",
        )
        out = pipe(
            _join_prompts(system_prompt, user_prompt),
            max_new_tokens=int(OLLAMA_NUM_PREDICT),
            do_sample=True,
            temperature=float(temperature if temperature is not None else TEMP),
            top_p=float(TOP_P),
        )
        if isinstance(out, list) and out:
            return str(out[0].get("generated_text", ""))
        return str(out)
    except Exception as e:
        _log(f"error: hf-local repo={repo_id}: {e}")
        return f"__ERROR__ {e}"


# ---------------------------
# Router
# ---------------------------

def _generate_for_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: Optional[float] = None,
) -> str:
    """
    Route based on prefix:
      - "gemini:<model>" -> Gemini CLI
      - "hf:<repo_id>"   -> Hugging Face (API if token present, else local if HF_MODE=local)
      - "ollama:<name>"  -> Ollama
      - (no prefix)      -> Ollama (back-compat)
    """
    _log(f"route: {detect_backend_for_model(model)}")
    if model.startswith("gemini:"):
        return _gemini_cli_generate(system_prompt, user_prompt, model.split(":", 1)[1])

    if model.startswith("hf:"):
        repo = model.split(":", 1)[1]
        force_local = os.getenv("HF_MODE", "").strip().lower() == "local"
        has_token = bool(os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        if has_token and not force_local:
            return _hf_inference_api_generate(system_prompt, user_prompt, repo, temperature=temperature)
        return _hf_local_generate(system_prompt, user_prompt, repo, temperature=temperature)

    if model.startswith("ollama:"):
        return _ollama_generate(system_prompt, user_prompt, model.split(":", 1)[1], temperature=temperature)

    # Back-compat: treat unprefixed names as Ollama models (e.g., "qwen3-coder:30b")
    return _ollama_generate(system_prompt, user_prompt, model, temperature=temperature)


# ---------------------------
# Candidate merging (unchanged)
# ---------------------------

def merge_candidates(list_of_lists: List[List[str]], limit: int) -> List[str]:
    merged, seen = [], set()
    idx = 0
    while len(merged) < limit and any(idx < len(lst) for lst in list_of_lists):
        for lst in list_of_lists:
            if idx < len(lst):
                cand = lst[idx]
                if cand not in seen:
                    seen.add(cand)
                    merged.append(cand)
                    if len(merged) >= limit:
                        break
        idx += 1
    return merged


# ---------------------------
# Public APIs used by prover.py
# ---------------------------

def propose_steps(
    models: List[str],
    goal: str,
    steps_so_far: List[str],
    state_hint: str = "",
    facts: List[str] | None = None,
    reranker=None,
    depth: int = 0,
    *,
    temp: float | None = None,
) -> List[str]:
    user = USER_TEMPLATE.format(
        goal=goal,
        steps="\n".join(steps_so_far) or "(none)",
        state_hint=(state_hint.strip() or "(none)"),
        facts=("\n".join(facts) if facts else "(none)")
    )
    all_lists: List[List[str]] = []
    for m in models:
        raw = _generate_for_model(m, SYSTEM_STEPS, user, temperature=temp)
        if _VERBOSE and isinstance(raw, str) and raw.startswith("__ERROR__"):
            _log(f"warn: LLM call failed for model={m}; using heuristic fallbacks. detail={raw}")
        cands = parse_ollama_lines(raw, ["apply ", "apply("], NUM_CANDIDATES)
        all_lists.append(cands)

    from .heuristics import augment_with_facts_for_steps, rank_candidates
    merged = merge_candidates(all_lists, NUM_CANDIDATES) or [
        "apply simp",
        "apply auto",
        "apply clarsimp",
        "apply (simp only: foo_def)",
        "apply (simp add: foo_def)",
        "apply (induction xs)",
        "apply (cases xs)",
    ]

    # If no facts provided, synthesize a couple likely *_def guesses from goal words.
    if not facts:
        import re
        words = re.findall(r"[A-Za-z][A-Za-z0-9_']{2,}", (goal + " " + state_hint))
        stems = [w for w in words if not w[0].isupper()]
        stems = list(dict.fromkeys(stems))[:3]
        for s in stems:
            merged.insert(0, f"apply (auto simp add: {s}_def)")
            merged.insert(0, f"apply (simp only: {s}_def)")
            merged.insert(0, f"apply (simp add: {s}_def)")

    if facts:
        merged = augment_with_facts_for_steps(merged, facts)
    return rank_candidates(merged, goal, state_hint, facts, reranker=reranker, depth=depth)


def propose_finishers(
    models: List[str],
    goal: str,
    steps_so_far: List[str],
    state_hint: str,
    mined_lemmas: List[str],
    hint_lemmas_limit: int,
    facts: List[str] | None = None,
    *,
    temp: float | None = None,
    reranker=None,
) -> List[str]:
    user = USER_TEMPLATE.format(
        goal=goal,
        steps="\n".join(steps_so_far) or "(none)",
        state_hint=(state_hint.strip() or "(none)"),
        facts=("\n".join(facts) if facts else "(none)")
    )
    base_lists: List[List[str]] = []
    for m in models:
        raw = _generate_for_model(m, SYSTEM_FINISH, user, temperature=temp)
        if _VERBOSE and isinstance(raw, str) and raw.startswith("__ERROR__"):
            _log(f"warn: LLM call failed for model={m}; using heuristic fallbacks. detail={raw}")
        base = parse_ollama_lines(raw, ["done", "by "], max(3, min(NUM_CANDIDATES, 8)))
        base_lists.append(base)

    base_merged = merge_candidates(base_lists, max(3, min(NUM_CANDIDATES, 8))) or [
        "done", "by simp", "by auto", "by clarsimp",
        "by (simp only: foo_def)", "by (simp add: foo_def)",
        "by arith", "by presburger", "by fastforce", "by blast", "by meson", "by (metis)"
    ]

    from .heuristics import suggest_common_lemmas, mk_finisher_variants, augment_with_facts_for_finishers, live_features_for
    static_hints = suggest_common_lemmas(state_hint)
    dyn_hints = mined_lemmas[:max(0, hint_lemmas_limit)]
    lemma_finishers = mk_finisher_variants(static_hints + dyn_hints)
    with_facts = augment_with_facts_for_finishers(base_merged, facts or [], cap=8)

    combined, seen = [], set()
    for x in with_facts + lemma_finishers + base_merged:
        if x not in seen:
            seen.add(x)
            combined.append(x)
        if len(combined) >= 8:
            break

    # Heuristic tiering (short/simple first)
    order = [
        "done", "by simp", "by clarsimp", "by auto",
        "by fastforce", "by blast", "by arith", "by presburger",
        "by (simp", "by (auto", "by (metis", "by (meson",
    ]
    ranked = sorted(combined, key=lambda cmd: next((i for i, k in enumerate(order) if cmd.startswith(k)), len(order)))

    # Optional learned reranking
    if reranker and getattr(reranker, "available", lambda: False)():
        def rscore(cmd: str) -> float:
            try:
                return float(reranker.score(live_features_for(cmd, goal, state_hint, depth=0)))
            except Exception:
                return 0.5
        ranked = sorted(ranked, key=lambda c: (-rscore(c), ranked.index(c)))

    return ranked
