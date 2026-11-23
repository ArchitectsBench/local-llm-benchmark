"""Eval runner — benchmark Foundry Local models on the shared prompt set.

Usage (inside the venv from the runbook):
    python run_eval.py --model phi-3.5-mini
    python run_eval.py --model phi-4-mini-reasoning
    python run_eval.py --list-models

Each run writes results_<model>_<timestamp>.json with per-prompt outputs,
latency, and tokens/sec. Run once per candidate model; judge_eval.py then
scores all result files against the baseline.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

HERE = Path(__file__).parent


def get_manager():
    FoundryLocalManager.initialize(Configuration(app_name="foundry-local-pot"))
    return FoundryLocalManager.instance


def run(alias: str):
    mgr = get_manager()
    eps = [e.name for e in mgr.discover_eps()]
    model = mgr.catalog.get_model(alias)
    if model is None:
        opts = [m.alias for m in mgr.catalog.list_models()]
        raise SystemExit(f"'{alias}' not in catalog. Available: {opts}")

    if not model.is_cached:
        print(f"Downloading {model.id} ...")
        model.download(progress_callback=lambda p: print(f"\r  {p:.0f}%", end="", flush=True))
        print()
    t0 = time.time()
    model.load()
    load_s = time.time() - t0
    chat = model.get_chat_client()
    print(f"Loaded {model.id} in {load_s:.1f}s | EPs: {eps}")

    prompts = json.loads((HERE / "eval_prompts.json").read_text())
    results = {
        "model_alias": alias,
        "model_id": model.id,
        "execution_providers": eps,
        "load_seconds": round(load_s, 1),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "items": [],
    }

    for cat in prompts["categories"]:
        for p in cat["prompts"]:
            print(f"[{cat['id']}/{p['id']}] ...", end="", flush=True)
            t1 = time.time()
            first_token = None
            chunks = []
            for chunk in chat.complete_streaming_chat(
                [{"role": "user", "content": p["prompt"]}]
            ):
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    if first_token is None:
                        first_token = time.time() - t1
                    chunks.append(delta)
            total_s = time.time() - t1
            text = "".join(chunks)
            # chunk count approximates generated tokens closely enough for comparison
            tps = len(chunks) / total_s if total_s > 0 else 0
            results["items"].append({
                "category": cat["id"],
                "prompt_id": p["id"],
                "prompt": p["prompt"],
                "expected": p["expected"],
                "output": text,
                "first_token_seconds": round(first_token or 0, 2),
                "total_seconds": round(total_s, 2),
                "approx_tokens_per_second": round(tps, 1),
            })
            print(f" {total_s:.1f}s, ~{tps:.0f} tok/s")

    out = HERE / "results" / f"results_{alias.replace('.', '_')}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out.name} ({len(results['items'])} prompts)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phi-3.5-mini")
    ap.add_argument("--list-models", action="store_true")
    a = ap.parse_args()
    if a.list_models:
        print([m.alias for m in get_manager().catalog.list_models()])
    else:
        run(a.model)
