#!/usr/bin/env python3
import argparse, json, random, re, sys
from pathlib import Path

def load_dataset_stream(path: str):
    try:
        from datasets import load_dataset
    except Exception:
        print("Please `pip install -U datasets` to stream full_dataset.json", file=sys.stderr)
        raise
    # The file is standard JSON with an array or JSONL. datasets can handle both.
    return load_dataset("json", data_files=path, split="train", streaming=True)

LEMMA_NAME_RE = re.compile(r'^\s*lemma\s+([^\s:"]+)', re.IGNORECASE)

def extract_lemma_name_from_statement(statement: str) -> str | None:
    if not statement:
        return None
    first = statement.splitlines()[0]
    m = LEMMA_NAME_RE.match(first)
    return m.group(1) if m else None

def clean_goal_from_state(state: str, fallback_statement: str | None = None) -> str:
    if not state:
        return fallback_statement or ""
    # Try to keep the lines under "goal (X subgoal)" header
    lines = state.splitlines()
    out = []
    take = False
    for ln in lines:
        if "goal (" in ln and "subgoal" in ln:
            take = True
            continue
        if take:
            out.append(ln)
    if out:
        text = "\n".join(out).strip()
        return text or state
    # Else fall back to full state or the statement
    return state or (fallback_statement or "")

TOKEN_RE = re.compile(r"[A-Za-z0-9_.'<>]+")

def tokens_in_step(step: str):
    if not step or len(step) < 4:  # micro-opt, output identical
        return []
    return TOKEN_RE.findall(step)

def build_name_index(input_path: str, max_rows: int | None = None):
    # Same final result as original: names are de-duplicated preserving first occurrence order,
    # and name2stmt maps first seen statement.
    ds = load_dataset_stream(input_path)
    name2stmt = {}
    names = []
    seen_names = set()  # moved de-dup inline (keeps original order)
    count = 0
    for ex in ds:
        nm = ex.get("premise_name")
        st = ex.get("premise_statement")
        if nm:
            nm = str(nm)
            if nm not in name2stmt and st:
                name2stmt[nm] = str(st).strip()
            if nm not in seen_names:
                seen_names.add(nm)
                names.append(nm)
        count += 1
        if max_rows and count >= max_rows:
            break
    return names, name2stmt

def convert(input_path: str, out_path: str, k_pool: int = 64,
            max_pos: int = 4, max_rows: int | None = None,
            clean_goal: bool = True, seed: int = 7):
    random.seed(seed)
    # Pass 1: build global name index
    print("[magnus] Building name index...")
    names, name2stmt = build_name_index(input_path)
    names_set = set(names)
    print(f"[magnus] Unique names: {len(names)}; mapped statements: {len(name2stmt)}")

    # Pass 2: stream and write attempts
    ds = load_dataset_stream(input_path)
    out_dir = Path(out_path).parent; out_dir.mkdir(parents=True, exist_ok=True)
    out = open(out_path, "w", encoding="utf-8")
    rows = 0

    dumps = json.dumps  # local bind (tiny speedup)

    for i, ex in enumerate(ds):
        if max_rows and rows >= max_rows:
            break

        # goal text
        state = ex.get("state") or ""
        statement = ex.get("statement") or ""
        goal = clean_goal_from_state(state, fallback_statement=statement) if clean_goal else (state or statement or "")
        goal = goal.strip()

        # positives
        pos = set()
        main_name = ex.get("premise_name")
        if main_name:
            pos.add(str(main_name))
        # mine extra positives mentioned in 'step' if they are known lemma names
        step = ex.get("step") or ""
        if step:
            for tok in tokens_in_step(step):
                if tok in names_set:
                    pos.add(tok)

        # keep at most max_pos (deterministic order: put the explicit premise_name first when present)
        pos_list = []
        main = str(main_name) if main_name else None
        if main and main in pos:
            pos.remove(main); pos_list.append(main)
        # fill the rest in sorted order for determinism
        if pos:
            for t in sorted(pos):
                if len(pos_list) >= max_pos:
                    break
                pos_list.append(t)
        if not pos_list:
            # skip rows without at least one positive
            continue

        # retrieval pool: positive(s) + random negatives
        # Build population with O(1) membership via set; behavior unchanged.
        pos_set = set(pos_list)
        population = [n for n in names if n not in pos_set]
        pool = list(pos_list)
        if population:
            take = max(0, k_pool - len(pool))
            if take > 0:
                pool.extend(random.sample(population, k=min(take, len(population))))
        # ensure deterministic order: positives first, then sorted negatives
        if len(pool) > len(pos_list):
            negs = sorted([n for n in pool if n not in pos_set])
            pool = pos_list + negs

        # facts_map: embed text for every name in pool (fallback to the name)
        facts_map = {nm: name2stmt.get(nm, nm) for nm in pool}

        # Compose attempts row
        rec = {
            "type": "finish",
            "origin": "magnus",
            "ok": True,
            "goal": goal,
            "cand_facts": pos_list,           # single or multi positives
            "retrieval_picks": pool,          # includes positives
            "facts_map": facts_map,           # name -> statement text
            # (optional) hint about the enclosing lemma
            "lemma": extract_lemma_name_from_statement(statement) or None
        }
        out.write(dumps(rec, ensure_ascii=False) + "\n")
        rows += 1

        if rows % 10000 == 0:
            print(f"[magnus] Wrote {rows} rows...")

    out.close()
    print(f"[magnus] Done. Wrote {rows} rows to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Convert Magnus full_dataset.json to self-contained attempts JSONL (with facts_map).")
    ap.add_argument("--input", required=True, help="Path to full_dataset.json")
    ap.add_argument("--out", default="logs/attempts.magnus.fulltext.jsonl", help="Output attempts JSONL")
    ap.add_argument("--k-pool", type=int, default=64, help="Total pool size (positives + negatives)")
    ap.add_argument("--max-pos", type=int, default=4, help="Max number of positive facts to include from step/premise_name")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit rows (for quick runs)")
    ap.add_argument("--clean-goal", action="store_true", help="Strip boilerplate and keep only subgoal lines when present")
    ap.add_argument("--no-clean-goal", dest="clean_goal", action="store_false")
    ap.add_argument("--random-seed", type=int, default=7)
    ap.set_defaults(clean_goal=True)
    args = ap.parse_args()

    convert(args.input, args.out, k_pool=args.k_pool, max_pos=args.max_pos,
            max_rows=args.max_rows, clean_goal=args.clean_goal, seed=args.random_seed)

if __name__ == "__main__":
    main()
