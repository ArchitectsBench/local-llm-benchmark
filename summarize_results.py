"""Summarize benchmark runs — rebuild the README tables from results/*.json.

Usage:
    python summarize_results.py            # print both tables as Markdown
    python summarize_results.py --check    # fail if README.md has drifted

Standard library only. When a model has several result files, the most
recent run (by its run_at timestamp) is used.

--check exits 1 if any generated table row is missing from README.md, so the
published numbers cannot drift from the raw result files without it showing.
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).parent

# Display names used in the README. Unknown aliases and categories fall back
# to their raw id, so a new model or category still appears in the output.
MODEL_LABELS = {
    "phi-3.5-mini": "Phi-3.5 mini",
    "phi-4-mini-reasoning": "Phi-4 mini reasoning",
    "qwen2.5-7b": "Qwen 2.5 7B",
}
CATEGORY_LABELS = {
    "reasoning": "Multi-step reasoning",
    "structured": "Structured output",
    "constraints": "Constraint following",
    "code": "Code",
    "grounded": "Grounded answering",
    "ambiguity": "Ambiguity handling",
    "summarization": "Summarization",
}


def load_latest_runs(results_dir: Path) -> list[dict]:
    """Return one run per model alias (the most recent), in first-seen order."""
    latest: dict[str, dict] = {}
    for path in sorted(results_dir.glob("results_*.json")):
        run = json.loads(path.read_text())
        alias = run["model_alias"]
        if alias not in latest or run["run_at"] > latest[alias]["run_at"]:
            latest[alias] = run
    return list(latest.values())


def summary_rows(runs: list[dict]) -> list[str]:
    rows = [
        "| Model | Load time | First token (avg) | Response time (avg) "
        "| Slowest prompt | Tokens per second (avg) |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        items = run["items"]
        name = MODEL_LABELS.get(run["model_alias"], run["model_alias"])
        rows.append(
            f"| {name} "
            f"| {run['load_seconds']:.1f} s "
            f"| {mean(i['first_token_seconds'] for i in items):.2f} s "
            f"| {mean(i['total_seconds'] for i in items):.1f} s "
            f"| {max(i['total_seconds'] for i in items):.1f} s "
            f"| {mean(i['approx_tokens_per_second'] for i in items):.1f} |"
        )
    return rows


def category_rows(runs: list[dict]) -> list[str]:
    names = [MODEL_LABELS.get(r["model_alias"], r["model_alias"]) for r in runs]
    rows = [
        "| Category | Prompts | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(runs),
    ]
    # Category order follows the first run, which follows eval_prompts.json.
    categories = list(dict.fromkeys(i["category"] for i in runs[0]["items"]))
    for cat in categories:
        count = sum(1 for i in runs[0]["items"] if i["category"] == cat)
        cells = []
        for run in runs:
            times = [i["total_seconds"] for i in run["items"] if i["category"] == cat]
            cells.append(f"{mean(times):.1f}" if times else "n/a")
        label = CATEGORY_LABELS.get(cat, cat)
        rows.append(f"| {label} | {count} | " + " | ".join(cells) + " |")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--check", action="store_true",
                    help="verify that README.md contains every generated row")
    args = ap.parse_args()

    runs = load_latest_runs(args.results_dir)
    if not runs:
        print(f"No result files found in {args.results_dir}", file=sys.stderr)
        return 2

    summary, by_category = summary_rows(runs), category_rows(runs)

    if not args.check:
        print("\n".join(summary))
        print("\nAverage response time by category, in seconds:\n")
        print("\n".join(by_category))
        return 0

    readme_lines = {line.strip() for line in (HERE / "README.md").read_text().splitlines()}
    missing = [row for row in summary + by_category if row not in readme_lines]
    if missing:
        print("README.md does not match the result files. Missing rows:", file=sys.stderr)
        for row in missing:
            print(f"  {row}", file=sys.stderr)
        return 1
    print(f"README.md matches {len(runs)} result files "
          f"({len(summary) + len(by_category)} table rows checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
