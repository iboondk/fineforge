"""Render the benchmark markdown from a results JSON, or a TBD placeholder.

    python -m fineforge_train.bench.report [--results PATH] [--out results/benchmark.md]

With no results file (or the file is missing) every metric cell reads ``TBD``, so the
report never shows a number that was not measured (see PLAN.md section 6).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

METRICS = ("exact_match", "token_f1", "rouge_l")
TARGET_TEXT = "target ≈ 26%"
PLACEHOLDER_HEADLINE = "Headline: relative improvement = TBD (target ≈ 26%; not yet measured)"


def relative_improvement(base: float, tuned: float) -> str:
    """(tuned - base) / base as a signed percentage with one decimal, or 'n/a' if base is 0."""
    if base == 0:
        return "n/a"
    return f"{(tuned - base) / base * 100:+.1f}%"


def render(results: dict | None) -> str:
    """Return the benchmark markdown. ``results`` is a results_record dict or None."""
    lines = ["# FineForge benchmark", ""]
    lines.append("| Metric | Base | Tuned | Relative change |")
    lines.append("|---|---|---|---|")

    if results is None:
        for metric in METRICS:
            lines.append(f"| {metric} | TBD | TBD | TBD |")
        lines.append("")
        lines.append(PLACEHOLDER_HEADLINE)
        return "\n".join(lines) + "\n"

    base = results["base"]
    tuned = results["tuned"]
    for metric in METRICS:
        b = float(base[metric])
        t = float(tuned[metric])
        lines.append(f"| {metric} | {b:.4f} | {t:.4f} | {relative_improvement(b, t)} |")
    lines.append("")
    headline = relative_improvement(float(base["token_f1"]), float(tuned["token_f1"]))
    lines.append(
        f"Headline: relative improvement (token_f1) = {headline} "
        f"(measured; {TARGET_TEXT})"
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- model_id: {results['model_id']}")
    lines.append(f"- adapter_path: {results['adapter_path']}")
    lines.append(f"- eval_sha256: {results['eval_sha256']}")
    lines.append(f"- n: {results['n']}")
    lines.append(f"- seed: {results['seed']}")
    lines.append(f"- timestamp: {results['timestamp_utc']}")
    return "\n".join(lines) + "\n"


def _load_results(path: str | None) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fineforge_train.bench.report",
        description="Render results/benchmark.md from a results JSON, or a TBD placeholder.",
    )
    parser.add_argument("--results", default=None, help="results JSON from bench.run (optional)")
    parser.add_argument("--out", default="results/benchmark.md", help="markdown output path")
    args = parser.parse_args(argv)

    try:
        results = _load_results(args.results)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(render(results))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
