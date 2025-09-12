from __future__ import annotations
import argparse, json, math, re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Iterable, Optional

ID_RE   = re.compile(r"\b([A-Za-z_][A-Za-z0-9_']*)\b", re.UNICODE)
DEF_RE  = re.compile(r"\b([A-Za-z_][A-Za-z0-9_']*)_def\b", re.UNICODE)
SIMPAD_RE   = re.compile(r"\bsimp\s+add:\s+([^\n\)]+)", re.UNICODE)
UNFOLD_RE   = re.compile(r"\bunfolding\s+([^\n\)]+)", re.UNICODE)
CASES_RULE  = re.compile(r"\bcases\s+rule:\s*([A-Za-z0-9_\.]+)", re.UNICODE)
INDUCT_VAR  = re.compile(r"\binduction\s+([A-Za-z_][A-Za-z0-9_']*)", re.UNICODE)

def _tokenize_goal(g: str) -> List[str]:
    toks = ID_RE.findall(g)
    special = []
    if "@" in g: special.append("@")
    if "⟹" in g: special.append("implies")
    return list(dict.fromkeys(toks + special))

def _detect_pattern(outline: str) -> str:
    if "proof (induction" in outline:
        return "induction"
    m = CASES_RULE.search(outline)
    if m:
        return f"cases_rule:{m.group(1)}"
    if "proof (cases" in outline:
        return "cases"
    return "plain"

def _extract_hints(outline: str, defs_in_block: Iterable[str]) -> List[str]:
    hints: List[str] = []
    for m in SIMPAD_RE.finditer(outline):
        parts = m.group(1).strip().split()
        hints += [p.strip() for p in parts if p.strip()]
    for m in UNFOLD_RE.finditer(outline):
        parts = m.group(1).strip().split()
        hints += [("unfolding " + p.strip()) for p in parts if p.strip()]
    for m in CASES_RULE.finditer(outline):
        hints.append(f"cases rule: {m.group(1)}")
    for m in INDUCT_VAR.finditer(outline):
        hints.append(f"induction {m.group(1)}")
    # defs present in the block are generally good hints to unfold
    for d in defs_in_block:
        hints.append(f"{d}_def")
    # de-dup preserve order
    return list(dict.fromkeys(hints))

def aggregate(in_jsonl: str,
              out_priors: str,
              out_hintlex: str,
              min_count: int = 3,
              topk: int = 8) -> None:
    """
    Reads rich JSONL produced by mine_afp_corpus_rich and outputs:
      - priors: {"rules":[{"if_any_tokens":[...], "prefer_patterns":[...], "weight":0.3}, ...]}
      - hintlex: {"token":[["hint",count], ...], ...}
    """
    # token -> pattern -> count
    pat_counts: Dict[str, Counter] = defaultdict(Counter)
    # token -> hint -> count
    hint_counts: Dict[str, Counter] = defaultdict(Counter)

    n = 0
    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            goal = rec.get("goal") or ""
            outline = rec.get("outline") or ""
            defs_in_block = rec.get("defs_in_block") or []
            if not goal or not outline:  # keep robust
                continue
            tokens = _tokenize_goal(goal)
            if not tokens:
                continue
            pattern = _detect_pattern(outline)
            hints = _extract_hints(outline, defs_in_block)

            for t in tokens:
                pat_counts[t][pattern] += 1
                for h in hints:
                    hint_counts[t][h] += 1
            n += 1

    # ---- build priors rules
    rules: List[Dict[str, Any]] = []
    for tok, cnts in pat_counts.items():
        total = sum(cnts.values())
        if total < min_count:
            continue
        # pick top 1-2 preferred patterns
        prefs = [p for p, c in cnts.most_common(2)]
        if not prefs:
            continue
        # weight: mild (0.3) scaled by sharpness of distribution
        if len(cnts) == 1:
            weight = 0.45
        else:
            topc = cnts[prefs[0]]
            weight = 0.3 + 0.15 * (topc / max(1, total))
        rules.append({
            "if_any_tokens": [tok],
            "prefer_patterns": prefs,
            "weight": round(float(weight), 3)
        })

    with open(out_priors, "w", encoding="utf-8") as f:
        json.dump({"rules": rules}, f, ensure_ascii=False, indent=2)

    # ---- build hint lexicon
    hintlex: Dict[str, List[Tuple[str, int]]] = {}
    for tok, cnts in hint_counts.items():
        # keep only hints seen >= min_count, top-k
        filt = [(h, c) for h, c in cnts.most_common() if c >= min_count]
        if not filt:
            continue
        hintlex[tok] = filt[:topk]

    with open(out_hintlex, "w", encoding="utf-8") as f:
        json.dump(hintlex, f, ensure_ascii=False, indent=2)

    print(f"Aggregated {n} records → {out_priors}, {out_hintlex}")

def main():
    ap = argparse.ArgumentParser(description="Aggregate priors and hint lexicon from rich AFP JSONL.")
    ap.add_argument("--input", required=True, help="Path to rich JSONL (from mine_afp_corpus_rich)")
    ap.add_argument("--priors", required=True, help="Output JSON for planner --priors")
    ap.add_argument("--hintlex", required=True, help="Output JSON for planner --hintlex")
    ap.add_argument("--min-count", type=int, default=3, help="Minimum frequency to keep patterns/hints")
    ap.add_argument("--topk", type=int, default=8, help="Top hints per token to keep")
    args = ap.parse_args()
    aggregate(args.input, args.priors, args.hintlex, min_count=args.min_count, topk=args.topk)

if __name__ == "__main__":
    main()
