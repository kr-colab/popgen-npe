#!/usr/bin/env python3
"""
Run MSMC2 baseline on existing tree sequences.

This script reads existing simulated tree sequences and runs MSMC2
to provide a baseline for comparison with neural network approaches.

All outputs go to baselines/results/. Existing comparison-results.pkl
files are never modified.

Usage:
    python run_baselines.py [options]
"""

import argparse
import pickle
import tskit
from pathlib import Path

from msmc2_utils import run_msmc2_pipeline, MSMC2_DEFAULT


# Scenarios and their tree sequence paths
BASE_DIR = Path(__file__).parent.parent / "ne_comparison_6_scenarios" / "log_time_with_LD"
SCENARIOS = {
    "medium": BASE_DIR / "medium" / "simulated_data_Dependent" / "simulated.trees",
    "large": BASE_DIR / "large" / "simulated_data_Dependent" / "simulated.trees",
    "decline": BASE_DIR / "decline" / "simulated_data_Dependent" / "simulated.trees",
    "expansion": BASE_DIR / "expansion" / "simulated_data_Dependent" / "simulated.trees",
    "bottleneck": BASE_DIR / "bottleneck" / "simulated_data_Dependent" / "simulated.trees",
    "zigzag": BASE_DIR / "zigzag" / "simulated_data_Dependent" / "simulated.trees",
}

MUTATION_RATE = 1e-8


def run_all_baselines(scenarios=None, output_dir=None, msmc2_path=MSMC2_DEFAULT,
                      seed=42):
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results to preserve other scenarios when running a subset
    output_file = output_dir / "baseline_results.pkl"
    if output_file.exists():
        with open(output_file, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Processing scenario: {scenario}")
        print(f"{'='*60}")

        ts_path = SCENARIOS[scenario]
        if not ts_path.exists():
            print(f"  WARNING: Tree sequence not found: {ts_path}")
            continue

        ts = tskit.load(ts_path)
        print(f"  Loaded tree sequence: {ts.num_samples} samples, {ts.num_sites} sites")
        print(f"  Sequence length: {ts.sequence_length:,.0f} bp")

        msmc2_dir = output_dir / scenario / "msmc2"

        print(f"\n  Running MSMC2 (seed={seed})...")
        try:
            msmc2_results = run_msmc2_pipeline(
                ts,
                msmc2_dir,
                mutation_rate=MUTATION_RATE,
                n_diploids=10,
                seed=seed, 
                msmc2_path=msmc2_path,
                n_threads=20,
                iterations=20,
                fixed_recombination=True,
                rho_over_mu=1,  # mu=r=1e-8
            )
            results[scenario] = {'msmc2': msmc2_results}
            print(f"  MSMC2 complete: {len(msmc2_results['times'])} time points")
        except Exception as e:
            print(f"  MSMC2 failed: {e}")
            results[scenario] = {'msmc2': None}

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run MSMC2 baseline")
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS.keys()), default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--msmc2-path", type=str, default=MSMC2_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_baselines(
        scenarios=args.scenarios,
        output_dir=args.output_dir,
        msmc2_path=args.msmc2_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
