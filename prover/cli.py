# prover/cli.py
import argparse
import os
import re
import requests
import sys

from . import config as CFG

from .config import (
    MODEL,
    BEAM_WIDTH,
    MAX_DEPTH,
    HINT_LEMMAS,
    FACTS_LIMIT,
    MINIMIZE_TIMEOUT,
    MINIMIZE_MAX_FACT_TRIES,
    VARIANT_TIMEOUT,
    VARIANT_TRIES,
    MINIMIZE_DEFAULT,
    VARIANTS_DEFAULT,
)
from .prover import prove_goal
from .isabelle_api import start_isabelle_server, get_isabelle_client
from .tactics import mine_two_step_macros

class _StoreSetFlag(argparse.Action):
    """Like 'store', but also marks that the user explicitly provided this flag."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"__set_{self.dest}", True)

# Avoid importing asyncio on Windows (perf + compatibility)
if sys.platform != "win32":
    import asyncio

# Pre-compile once (used by read_goals)
_LEMMA_RE = None
def _lemma_regex():
    global _LEMMA_RE
    if _LEMMA_RE is None:
        import re
        _LEMMA_RE = re.compile(r'lemma\s+"(.+)"', re.IGNORECASE)
    return _LEMMA_RE


def read_goals(path: str) -> list[str]:
    """Read one goal per line. Accepts raw formulas or 'lemma "..."' lines; ignores blanks and #comments."""
    goals: list[str] = []
    rx = _lemma_regex()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("lemma "):
                m = rx.search(line)
                if m:
                    goals.append(m.group(1))
                else:
                    # Fallback: strip 'lemma' prefix and outer quotes if present
                    payload = line[len("lemma ") :].strip()
                    goals.append(payload.strip('"'))
            else:
                goals.append(line.strip('"'))
    return goals


def _setup_loop():
    """Create an event loop + child watcher (POSIX only)."""
    if sys.platform == "win32":
        return None, None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    watcher = None
    try:
        watcher = asyncio.SafeChildWatcher()  # type: ignore[attr-defined]
        asyncio.get_event_loop_policy().set_child_watcher(watcher)
        watcher.attach_loop(loop)
    except Exception:
        watcher = None  # Some platforms may not support child watcher
    return loop, watcher


def _teardown_loop(loop, watcher):
    if sys.platform == "win32":
        return
    try:
        if watcher is not None:
            # Best-effort: detach to avoid warnings on some Python builds
            try:
                watcher.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        if loop is not None:
            loop.close()
    except Exception:
        pass


def _prove_one(isabelle, session_id: str, goal: str, CFG, args, macro_map=None):
    """Thin wrapper to avoid duplicating the long prove_goal(...) call."""
    return prove_goal(
        isabelle,
        session_id,
        goal,
        CFG.MODEL,
        CFG.BEAM_WIDTH,
        CFG.MAX_DEPTH,
        CFG.HINT_LEMMAS,
        args.timeout,
        models=[m.strip() for m in args.models.split(",")] if args.models else None,
        save_dir=args.save_proofs,
        use_sledge=args.sledge,
        sledge_timeout=args.sledge_timeout,
        sledge_every=args.sledge_every,
        trace=args.trace,
        use_color=(not args.no_color),
        use_qc=args.quickcheck,
        qc_timeout=args.quickcheck_timeout,
        qc_every=args.quickcheck_every,
        use_np=args.nitpick,
        np_timeout=args.nitpick_timeout,
        np_every=args.nitpick_every,
        facts_limit=CFG.FACTS_LIMIT,
        do_minimize=(MINIMIZE_DEFAULT and (not args.no_minimize)),
        minimize_timeout=CFG.MINIMIZE_TIMEOUT,
        do_variants=((VARIANTS_DEFAULT or args.variants) and (not args.no_variants)),
        variant_timeout=CFG.VARIANT_TIMEOUT,
        variant_tries=CFG.VARIANT_TRIES,
        macro_map=macro_map,
        enable_reranker=(not args.no_reranker),
    )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--goal", type=str)
    parser.add_argument("--goals-file", type=str)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--beam", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--hint-lemmas", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--save-proofs", type=str, default=None)
    parser.add_argument("--sledge", action="store_true")
    parser.add_argument("--sledge-timeout", type=int, default=5)
    parser.add_argument("--sledge-every", type=int, default=2)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--quickcheck", action="store_true")
    parser.add_argument("--quickcheck-timeout", type=int, default=2)
    parser.add_argument("--quickcheck-every", type=int, default=1)
    parser.add_argument("--nitpick", action="store_true")
    parser.add_argument("--nitpick-timeout", type=int, default=5)
    parser.add_argument("--nitpick-every", type=int, default=2)
    parser.add_argument("--facts-limit", type=int, default=None)
    parser.add_argument("--no-minimize", action="store_true")
    parser.add_argument("--minimize-timeout", type=int, default=None)
    parser.add_argument("--minimize-max-fact-tries", type=int, default=None)
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--no-variants", action="store_true")
    parser.add_argument("--variant-timeout", type=int, default=None)
    parser.add_argument("--variant-tries", type=int, default=None)
    parser.add_argument("--no-reranker", action="store_true", help="Disable ML reranker (even if a model file exists).")
    parser.add_argument("--premises", action="store_true", help="Enable premises retrieval.")
    parser.add_argument("--no-premises", action="store_true", help="Disable premises retrieval.")
    parser.add_argument("--premises-k-select", type=int, default=CFG.PREMISES_K_SELECT,
                        action=_StoreSetFlag, help="Top-K for fast SELECT stage.")
    parser.add_argument("--premises-k-rerank", type=int, default=CFG.PREMISES_K_RERANK,
                        action=_StoreSetFlag, help="Top-K for RE-RANK stage.")
    parser.add_argument("--context", action="store_true", help="Enable file-aware context window.")
    parser.add_argument("--no-context", action="store_true", help="Disable file-aware context window.")
    parser.add_argument("--context-files", type=str,
                        default=(" ".join(CFG.PROVER_CONTEXT_FILES) if CFG.PROVER_CONTEXT_FILES else ""),
                        action=_StoreSetFlag,
                        help="Space/comma-separated .thy files to seed context.")
    parser.add_argument("--context-window", type=int, default=CFG.PROVER_CONTEXT_WINDOW,
                        action=_StoreSetFlag, help="Context window size (implementation-defined).")
    args = parser.parse_args()

    # Optional: check Ollama (fast, can be skipped by env)
    if os.environ.get("OLLAMA_SKIP_CHECK") != "1":
        try:
            requests.get(
                os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434") + "/api/tags",
                timeout=1.0,  # keep startup snappy
            )
        except Exception as e:
            print(f"Warning: could not reach Ollama. Error: {e}")

    # Runtime overrides via module globals (backward compatible)
    if args.model is not None:
        CFG.MODEL = args.model
    if args.beam is not None:
        CFG.BEAM_WIDTH = args.beam
    if args.max_depth is not None:
        CFG.MAX_DEPTH = args.max_depth
    if args.hint_lemmas is not None:
        CFG.HINT_LEMMAS = args.hint_lemmas
    if args.facts_limit is not None:
        CFG.FACTS_LIMIT = args.facts_limit
    if args.minimize_timeout is not None:
        CFG.MINIMIZE_TIMEOUT = args.minimize_timeout
    if args.minimize_max_fact_tries is not None:
        CFG.MINIMIZE_MAX_FACT_TRIES = args.minimize_max_fact_tries
    if args.variant_timeout is not None:
        CFG.VARIANT_TIMEOUT = args.variant_timeout
    if args.variant_tries is not None:
        CFG.VARIANT_TRIES = args.variant_tries
    if args.premises:
        CFG.PREMISES_ENABLE = True
    if args.no_premises:
        CFG.PREMISES_ENABLE = False
    if getattr(args, "__set_premises_k_select", False):
        CFG.PREMISES_K_SELECT = int(args.premises_k_select)
    if getattr(args, "__set_premises_k_rerank", False):
        CFG.PREMISES_K_RERANK = int(args.premises_k_rerank)
    if args.context:
        CFG.PROVER_CONTEXT_ENABLE = True
    if args.no_context:
        CFG.PROVER_CONTEXT_ENABLE = False
    if getattr(args, "__set_context_window", False):
        CFG.PROVER_CONTEXT_WINDOW = int(args.context_window)
    if getattr(args, "__set_context_files", False) and args.context_files:
        parts = [p for p in re.split(r"[,\s]+", args.context_files.strip()) if p]
        CFG.PROVER_CONTEXT_FILES = [os.path.expanduser(p) for p in parts]

    loop, watcher = _setup_loop()
    server_info = proc = isabelle = session_id = None

    try:
        server_info, proc = start_isabelle_server(name="isabelle", log_file="server.log")
        print(server_info.strip())
        isabelle = get_isabelle_client(server_info)
        session_id = isabelle.session_start(session="HOL")
        print("session_id:", session_id)

        # Mine macros from existing logs (fast; skips if file missing)
        macro_map = mine_two_step_macros()

        # Single goal
        if args.goal and not args.goals_file:
            res = _prove_one(isabelle, session_id, args.goal, CFG, args, macro_map=macro_map)
            flag = "TIMEOUT" if res.get("timeout") else ("SUCCESS" if res["success"] else "FAILED")
            print(f"\n{flag} | depth: {res['depth']}")
            steps = res.get("steps") or []
            if steps:
                print("\n".join(steps))
            return

        # Batch goals
        if args.goals_file:
            goals = read_goals(args.goals_file)
            n = len(goals)
            print(f"Running batch on {n} goals…")
            ok = 0
            for i, g in enumerate(goals, 1):
                print(f"\n[{i}/{n}] {g}")
                res = _prove_one(isabelle, session_id, g, CFG, args, macro_map=macro_map)
                flag = "TIMEOUT" if res.get("timeout") else ("SUCCESS" if res["success"] else "FAILED")
                if res.get("success"):
                    ok += 1
                print(f"  -> {flag} | depth: {res['depth']}")
            print(f"\nBatch done. Success: {ok}/{n} ({ok * 100.0 / max(n,1):.1f}%).")
            return

        # Default quick demo
        default_goal = "rev (rev xs) = xs"
        res = _prove_one(isabelle, session_id, default_goal, CFG, args)
        flag = "TIMEOUT" if res.get("timeout") else ("SUCCESS" if res["success"] else "FAILED")
        print(f"\n{flag} | depth: {res['depth']}")
        steps = res.get("steps") or []
        if steps:
            print("\n".join(steps))

    finally:
        # Strict shutdown order to avoid "Event loop is closed"
        try:
            if isabelle is not None:
                isabelle.shutdown()
        except Exception:
            pass
        try:
            if proc is not None:
                proc.terminate()
                proc.wait(timeout=3)
        except Exception:
            pass
        _teardown_loop(loop, watcher)


if __name__ == "__main__":
    main()
