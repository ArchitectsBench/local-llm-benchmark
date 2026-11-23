# Runbook (macOS): Foundry Local + Phi-3.5-mini on a MacBook Pro

**Goal:** working streaming chat at `http://localhost:8080` backed by a locally
cached model, running on the Apple Silicon GPU (Metal). No Azure subscription,
no cloud account; fully offline after the initial model download.

---

## 1. Prerequisites

| Item | Requirement | Check command |
|---|---|---|
| Mac | Apple Silicon (M1/M2/M3/M4) MacBook Pro | `sysctl -n machdep.cpu.brand_string` |
| macOS | Recent macOS (Sonoma or later recommended) | `sw_vers` |
| Python | 3.10 – 3.12, 64-bit arm64 | `python3 --version` and `python3 -c "import platform; print(platform.machine())"` → must print `arm64` |
| Disk | ~5 GB free | `df -h ~` |

> **Intel Macs:** no Metal GPU path for this stack — inference falls back to CPU.
> It will still work for a demo but is not representative for benchmarking.

> **arm64 gotcha:** if `platform.machine()` prints `x86_64`, your Python is running
> under Rosetta (common with older Anaconda installs). Use a native arm64 Python
> (e.g. from python.org or `brew install python@3.12`) or the GPU path won't engage.

## 2. Environment setup

On macOS you install the **cross-platform** package. (`foundry-local-sdk-winml` is
Windows-only — don't install it here.)

```bash
git clone https://github.com/ArchitectsBench/local-llm-benchmark.git && cd local-llm-benchmark

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install foundry-local-sdk
```

**Beware the impostor package:** `foundry-local` on PyPI (no `-sdk`) is an
unrelated third-party package. Microsoft's package is `foundry-local-sdk`.

## 3. GPU infrastructure check (before and during inference)

There's no `nvidia-smi` on a Mac. Three-step equivalent:

**a) Confirm the GPU exists and Metal is supported:**
```bash
system_profiler SPDisplaysDataType
```
Look for `Chipset Model: Apple M...`, the GPU core count, and
`Metal Support: Metal 3` (or later).

**b) Watch live GPU utilization while the model runs:**
Open **Activity Monitor → Window menu → GPU History** (⌘4). Keep it visible,
then run the smoke test in the terminal. You should see a clear utilization
spike during generation. Flat line during generation = CPU fallback (see §1
arm64 gotcha).

**c) Terminal alternative (numbers you can record):**
```bash
sudo powermetrics --samplers gpu_power -i 1000
```
Shows GPU active residency (%) and power draw per second. Run it in a second
terminal during generation; expect residency to jump well above idle. Ctrl+C
to stop.

Record idle vs during-inference residency — that's your evidence the Metal
path is engaged, and it belongs in the benchmark notes.

## 4. Model download, cache, and smoke test

```bash
python smoke_test.py
```

First run downloads Phi-3.5-mini (~2 GB) and caches it; subsequent runs load
from cache. Expected output: catalog aliases, load time, and a non-empty
one-line completion followed by `SMOKE TEST PASSED`.

- The SDK auto-selects the Metal/WebGPU execution provider on Apple Silicon —
  no configuration needed. The smoke test prints discovered execution
  providers (`mgr.discover_eps()`) — confirm a GPU-backed EP is listed.
- **SDK v1.2.x API note:** the module is `foundry_local_sdk` (not
  `foundry_local`), and usage is in-process:
  `FoundryLocalManager.initialize(Configuration(app_name=...))` →
  `FoundryLocalManager.instance.catalog.get_model(alias)` →
  `model.download()` / `model.load()` → `model.get_chat_client()`.
  Older samples using `from foundry_local import FoundryLocalManager` and an
  OpenAI client pointed at `mgr.endpoint` are pre-GA and will not import.
- **Model not found:** the catalog serves hardware-optimized variants per
  device; use one of the printed aliases if `phi-3.5-mini` isn't offered for
  your machine.
- Cache location: `~/.foundry` — persists across venvs; delete to reclaim disk.

## 5. Run the chat interface

```bash
python chat_app.py     # then open http://localhost:8080
```

Staged loading screen (runtime → catalog → download/cache → load → ready),
then streaming chat.

## 6. Reproduction checklist (acceptance test)

- [ ] `platform.machine()` prints `arm64` inside the venv
- [ ] Fresh venv; `pip install foundry-local-sdk` succeeds
- [ ] `python smoke_test.py` prints a non-empty completion, `SMOKE TEST PASSED`
- [ ] GPU History (or powermetrics) shows a utilization spike during generation
- [ ] Second run loads from cache with no download
- [ ] `python chat_app.py` → streaming responses at `http://localhost:8080`
- [ ] Wi-Fi off, restart `chat_app.py`, chat still works (zero cloud dependency)

Record: Mac model, chip (e.g. M3 Pro), GPU cores, load time, and GPU residency
during generation.

## 7. Fast triage

| Symptom | Cause | Fix |
|---|---|---|
| Slow tokens, flat GPU History | Rosetta/x86_64 Python → CPU fallback | Native arm64 Python, rebuild venv |
| `ModuleNotFoundError: foundry_local` | Pre-GA import name in old sample code | Import from `foundry_local_sdk` (v1.2.x API) |
| `ModuleNotFoundError: foundry_local_sdk` | venv not active / package missing | `source .venv/bin/activate`; `pip install foundry-local-sdk` |
| Model alias not found | Catalog variant differs per hardware | Use an alias from the printed catalog list |
| Slow first run | Initial ~2 GB download | Expected; cached afterwards |
| pip/download timeouts | Corporate proxy | `HTTPS_PROXY` env / `pip --proxy` |

## 8. Sources

- Get started with Foundry Local — learn.microsoft.com/en-us/azure/foundry-local/get-started
- SDK reference (Metal via WebGPU on Apple Silicon) — learn.microsoft.com/en-us/azure/foundry-local/reference/reference-sdk-current
- Package — pypi.org/project/foundry-local-sdk/
