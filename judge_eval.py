"""Judge — score local-model eval results using your Azure AI Foundry deployment
as both the quality baseline and the grader (LLM-as-judge).

Setup (uses your Azure pay-as-you-go subscription):
    export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
    export AZURE_OPENAI_API_KEY="<key>"          # or use Entra ID via azure-identity
    export AZURE_OPENAI_DEPLOYMENT="gpt-4o"      # your deployed model name
    pip install openai

Usage:
    python judge_eval.py results/results_phi-3_5-mini_*.json [results/results_phi-4-mini-reasoning_*.json ...]

For each result file it: (1) generates the baseline answer from the Foundry
deployment for every prompt, (2) asks the judge to score the LOCAL answer
1-5 on correctness, constraint adherence, and groundedness given the prompt,
the expected-behavior note, and the baseline answer, (3) writes a scored
copy and prints a per-category summary table.
"""
import glob
import json
import os
import statistics
import sys
from pathlib import Path

from openai import AzureOpenAI

DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-10-21",
)

JUDGE_SYSTEM = (
    "You are a strict evaluator. Score the CANDIDATE answer 1-5 (5 = fully correct, "
    "meets every constraint, no hallucination; 1 = wrong or ignores constraints). "
    "Judge against the PROMPT, the EXPECTED note, and the BASELINE answer. "
    "Respond ONLY with JSON: {\"score\": <1-5>, \"verdict\": \"pass|partial|fail\", "
    "\"reason\": \"<one sentence>\"}"
)


def baseline_answer(prompt: str) -> str:
    r = client.chat.completions.create(
        model=DEPLOYMENT, messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content or ""


def judge(prompt, expected, baseline, candidate) -> dict:
    user = (
        f"PROMPT:\n{prompt}\n\nEXPECTED:\n{expected}\n\n"
        f"BASELINE:\n{baseline}\n\nCANDIDATE:\n{candidate}"
    )
    r = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": user}],
    )
    txt = (r.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"score": 0, "verdict": "error", "reason": txt[:200]}


def main(paths):
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    if not files:
        sys.exit("No result files matched.")

    baselines: dict[str, str] = {}
    for f in files:
        data = json.loads(Path(f).read_text())
        print(f"\n=== {data['model_alias']} ({f}) ===")
        for item in data["items"]:
            pid = item["prompt_id"]
            if pid not in baselines:
                baselines[pid] = baseline_answer(item["prompt"])
            item["judge"] = judge(item["prompt"], item["expected"],
                                  baselines[pid], item["output"])
            print(f"  {item['category']}/{pid}: {item['judge']['verdict']} "
                  f"({item['judge']['score']}/5) - {item['judge']['reason']}")

        out = Path(f).with_name(Path(f).stem + "_scored.json")
        out.write_text(json.dumps(data, indent=2))

        by_cat: dict[str, list[int]] = {}
        for item in data["items"]:
            by_cat.setdefault(item["category"], []).append(item["judge"]["score"])
        print(f"\n  Category averages for {data['model_alias']}:")
        for cat, scores in by_cat.items():
            print(f"    {cat:15s} {statistics.mean(scores):.1f}/5  (n={len(scores)})")
        overall = statistics.mean(i["judge"]["score"] for i in data["items"])
        lat = statistics.mean(i["total_seconds"] for i in data["items"])
        print(f"    {'OVERALL':15s} {overall:.1f}/5   avg latency {lat:.1f}s")
        print(f"  Scored file: {out.name}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results_*.json"])
