"""CLI entry point for CombBandits experiments."""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("combbandits")


def main():
    parser = argparse.ArgumentParser(
        description="CombBandits: LLM-CUCB-AT experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run experiments from config")
    run_parser.add_argument("config", type=str, help="Path to experiment YAML config")
    run_parser.add_argument("--output-dir", type=str, default="results")
    run_parser.add_argument("--workers", type=int, default=None, help="Max parallel workers")
    run_parser.add_argument("--task-id", type=int, default=None, help="SLURM array task ID (run single task)")
    run_parser.add_argument("--task-range", type=str, default=None, help="Task range 'start:end' for SLURM")

    # --- export-tasks ---
    export_parser = subparsers.add_parser("export-tasks", help="Export task grid as CSV for SLURM")
    export_parser.add_argument("config", type=str)
    export_parser.add_argument("--output", type=str, default=None)

    # --- plot ---
    plot_parser = subparsers.add_parser("plot", help="Generate plots from results")
    plot_parser.add_argument("results", type=str, help="Path to results JSON")
    plot_parser.add_argument("--output-dir", type=str, default="figures")
    plot_parser.add_argument("--corruption", type=str, default=None)
    plot_parser.add_argument("--epsilon", type=float, default=None)

    # --- metrics ---
    metrics_parser = subparsers.add_parser("metrics", help="Compute summary metrics")
    metrics_parser.add_argument("results", type=str, help="Path to results JSON")

    args = parser.parse_args()

    if args.command == "run":
        from .engine.runner import ExperimentRunner
        runner = ExperimentRunner(args.config, args.output_dir)

        task_indices = None
        if args.task_id is not None:
            task_indices = [args.task_id]
        elif args.task_range:
            start, end = map(int, args.task_range.split(":"))
            task_indices = list(range(start, end))

        runner.run(max_workers=args.workers, task_indices=task_indices)

    elif args.command == "export-tasks":
        from .engine.runner import ExperimentRunner
        runner = ExperimentRunner(args.config)
        path = runner.export_task_list(args.output)
        print(f"Task list written to {path}")

    elif args.command == "plot":
        from .analysis.plots import plot_regret_curves, plot_corruption_comparison
        plot_regret_curves(args.results, f"{args.output_dir}/regret_curves.pdf",
                          filter_corruption=args.corruption, filter_epsilon=args.epsilon)
        plot_corruption_comparison(args.results, f"{args.output_dir}/corruption_comparison.pdf")
        print(f"Plots saved to {args.output_dir}/")

    elif args.command == "metrics":
        from .analysis.metrics import load_results, compute_metrics
        results = load_results(args.results)
        summary = compute_metrics(results)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
