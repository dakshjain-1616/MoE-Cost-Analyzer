#!/usr/bin/env python3
"""LLM-driven cost-benefit analyzer: dense vs MoE model comparison.

Usage:
    python analyze.py benchmark.json
    python analyze.py benchmark.json --sla-latency-ms 2000 --sla-cost-per-1k 0.01
    python analyze.py benchmark.json --output results.csv --dry-run
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.runner import run_benchmark
from src.analyzer import build_decision_matrix, recommend, save_csv
from src.reporter import print_matrix, print_recommendation, console
from src.pricing import KNOWN_MODEL_IDS

DENSE_MODEL = "google/gemma-4-31b-it"
MOE_MODEL = "google/gemma-4-26b-a4b-it"


def load_benchmark(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with p.open() as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {path}: {e}", file=sys.stderr)
            sys.exit(1)

    tasks = data.get("tasks", [])
    if not tasks:
        print("No tasks to evaluate.")
        sys.exit(0)

    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cost-benefit analyzer: dense vs MoE model on your benchmark suite."
    )
    parser.add_argument("benchmark", help="Path to benchmark JSON file")
    parser.add_argument("--dense-model", default=DENSE_MODEL,
                        help=f"Dense model ID (default: {DENSE_MODEL})")
    parser.add_argument("--moe-model", default=MOE_MODEL,
                        help=f"MoE model ID (default: {MOE_MODEL})")
    parser.add_argument("--sla-latency-ms", type=float, default=2000.0,
                        help="Max acceptable average latency in ms (default: 2000)")
    parser.add_argument("--sla-cost-per-1k", type=float, default=0.01,
                        help="Max acceptable cost per 1K queries in USD (default: 0.01)")
    parser.add_argument("--output", default="results.csv",
                        help="Output CSV path (default: results.csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip real API calls; use simulated latency/token data")
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    # Validate model IDs early
    for model_id in (args.dense_model, args.moe_model):
        if model_id not in KNOWN_MODEL_IDS:
            print(f"Error: Invalid model ID '{model_id}'", file=sys.stderr)
            sys.exit(1)

    tasks = load_benchmark(args.benchmark)

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Error: OPENROUTER_API_KEY not set. Use --dry-run or add it to .env", file=sys.stderr)
        sys.exit(1)

    console.print(
        f"\n[bold cyan]Running benchmark[/bold cyan] — {len(tasks)} tasks × 2 models"
        f"{' [dim](dry-run)[/dim]' if args.dry_run else ''}"
    )
    console.print(f"  Dense : [cyan]{args.dense_model}[/cyan]")
    console.print(f"  MoE   : [cyan]{args.moe_model}[/cyan]")

    dense_results, moe_results = await run_benchmark(
        api_key=api_key,
        tasks=tasks,
        dense_model=args.dense_model,
        moe_model=args.moe_model,
        dry_run=args.dry_run,
    )

    dense_stats, moe_stats, matrix = build_decision_matrix(dense_results, moe_results)
    rec = recommend(dense_stats, moe_stats, args.sla_latency_ms, args.sla_cost_per_1k)

    print_matrix(matrix, args.dense_model, args.moe_model)
    print_recommendation(rec, dense_stats, moe_stats)

    save_csv(dense_results, moe_results, args.output)
    console.print(f"[dim]Per-query results saved to:[/dim] {args.output}\n")


if __name__ == "__main__":
    asyncio.run(main())
