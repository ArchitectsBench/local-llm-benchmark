"""Smoke test — Foundry Local SDK v1.2.x (GA API).

Verifies: SDK install, execution-provider (GPU) discovery, model download/cache,
load, and a real non-empty completion.
"""
import sys
import time

from foundry_local_sdk import Configuration, FoundryLocalManager

ALIAS = "phi-3.5-mini"

t0 = time.time()
FoundryLocalManager.initialize(Configuration(app_name="foundry-local-pot"))
mgr = FoundryLocalManager.instance

# GPU / execution-provider check (nvidia-smi equivalent, from inside the SDK)
eps = mgr.discover_eps()
print("Execution providers:", [(e.name, getattr(e, "status", "")) for e in eps])

aliases = [m.alias for m in mgr.catalog.list_models()]
print("Catalog aliases (first 10):", aliases[:10])

model = mgr.catalog.get_model(ALIAS)
if model is None:
    sys.exit(f"'{ALIAS}' not in catalog for this hardware — pick one of: {aliases[:10]}")

if not model.is_cached:
    print(f"Downloading {model.id} (~2 GB first run)...")
    model.download(progress_callback=lambda p: print(f"\r  {p:.0f}%", end="", flush=True))
    print()
else:
    print(f"{model.id} already cached.")

model.load()
print(f"Loaded {model.id} in {time.time() - t0:.1f}s total")

chat = model.get_chat_client()
t1 = time.time()
resp = chat.complete_chat(
    [{"role": "user", "content": "Say 'local inference works' in five words."}]
)
out = resp.choices[0].message.content
print(f"Completion ({time.time() - t1:.1f}s): {out}")

assert out and out.strip(), (
    "EMPTY completion — GPU path unavailable (VM without GPU passthrough, "
    "or unsupported hardware). Run on physical hardware."
)
print("SMOKE TEST PASSED")
