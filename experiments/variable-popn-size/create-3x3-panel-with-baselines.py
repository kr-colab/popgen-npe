#!/usr/bin/env python3
"""
Create 3x3 panel plot with neural network posteriors and MSMC2 baseline.

This script reads:
1. Existing comparison-results.pkl files (neural network posteriors)
2. New baseline_results.pkl file (MSMC2 estimates)

And creates a unified panel plot showing all methods.
"""

import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from pathlib import Path


def plot_step_function(times, sizes, ax, label=None, color=None, alpha=1.0,
                       linewidth=2, linestyle="-"):
    """Plot population size as step function backward in time."""
    assert len(times) == len(sizes) + 1, \
        f"times (len={len(times)}) must be one longer than sizes (len={len(sizes)})"
    for i in range(len(sizes)):
        ax.hlines(sizes[i], times[i], times[i+1],
                  colors=color, linewidth=linewidth, alpha=alpha,
                  linestyle=linestyle, label=label if i == 0 else None)


def create_ribbon_plot_with_baselines(ax, posterior_samples, true_pop_sizes, change_times,
                                       msmc2_data=None, color='blue',
                                       show_legend=True, ylim=(1e2, 1e5)):
    """Create a ribbon plot for a single model with MSMC2 overlay."""
    # Extract population sizes from posterior samples (log10 scale to linear)
    posterior_pop_sizes = 10 ** posterior_samples[:, :-1]

    # Times for step plots
    times_with_one = np.concatenate([[1], np.array(change_times), [100000]])

    # Compute quantiles for each epoch
    lower_95 = np.percentile(posterior_pop_sizes, 2.5, axis=0)
    upper_95 = np.percentile(posterior_pop_sizes, 97.5, axis=0)
    lower_80 = np.percentile(posterior_pop_sizes, 10, axis=0)
    upper_80 = np.percentile(posterior_pop_sizes, 90, axis=0)
    lower_50 = np.percentile(posterior_pop_sizes, 25, axis=0)
    upper_50 = np.percentile(posterior_pop_sizes, 75, axis=0)

    def step_ribbon(times, lower, upper):
        step_times = np.empty(2 * (len(times) - 1))
        step_lower = np.empty_like(step_times)
        step_upper = np.empty_like(step_times)
        for j in range(len(times) - 1):
            step_times[2*j] = times[j]
            step_times[2*j+1] = times[j+1]
            step_lower[2*j] = lower[j]
            step_lower[2*j+1] = lower[j]
            step_upper[2*j] = upper[j]
            step_upper[2*j+1] = upper[j]
        return step_times, step_lower, step_upper

    # Plot ribbons
    st, sl, su = step_ribbon(times_with_one, lower_95, upper_95)
    ax.fill_between(st, sl, su, color=color, alpha=0.15,
                    label="95% CI" if show_legend else None)

    st, sl, su = step_ribbon(times_with_one, lower_80, upper_80)
    ax.fill_between(st, sl, su, color=color, alpha=0.25,
                    label="80% CI" if show_legend else None)

    st, sl, su = step_ribbon(times_with_one, lower_50, upper_50)
    ax.fill_between(st, sl, su, color=color, alpha=0.35,
                    label="50% CI" if show_legend else None)

    # Overlay true population size
    plot_step_function(times_with_one, true_pop_sizes, ax,
                       label="True" if show_legend else None, color="black", linewidth=3)

    # Add MSMC2 line
    if msmc2_data is not None and msmc2_data.get('times') is not None:
        ax.plot(msmc2_data['times'], msmc2_data['ne'],
                '--', color='saddlebrown', linewidth=2.5,
                label='MSMC2' if show_legend else None)

    # Styling
    ax.set_xlim(100, max(change_times) * 1.2)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Create 3x3 panel plot with neural network and MSMC2 baseline"
    )

    parser.add_argument(
        "--comparison-dirs", nargs="+", required=True,
        help="Directories containing comparison-results.pkl files"
    )
    parser.add_argument(
        "--baseline-results", required=True,
        help="Path to baseline_results.pkl file"
    )
    parser.add_argument(
        "--scenario-labels", nargs="+", required=True,
        help="Labels for each scenario"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output filename for panel plot"
    )
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=[15, 12],
        help="Figure size (width height)"
    )

    args = parser.parse_args()

    # Load baseline results
    with open(args.baseline_results, "rb") as f:
        baseline_results = pickle.load(f)

    # Extract scenario names from comparison dirs
    scenario_names = []
    for comp_dir in args.comparison_dirs:
        parts = Path(comp_dir).parts
        for part in parts:
            if part in ['abrupt_change', 'linear_decline', 'linear_growth']:
                scenario_names.append(part)
                break
        else:
            scenario_names.append(Path(comp_dir).name)

    # Load results from each comparison directory
    all_results = []
    model_labels = None

    for comp_dir in args.comparison_dirs:
        with open(os.path.join(comp_dir, "comparison-results.pkl"), "rb") as f:
            results = pickle.load(f)
        all_results.append(results)
        if model_labels is None:
            model_labels = [m["label"] for m in results["models"]]

    # Create panel plot
    n_scenarios = len(all_results)
    n_models = len(model_labels)

    fig, axes = plt.subplots(n_scenarios, n_models,
                             figsize=args.figsize,
                             constrained_layout=True, sharex=True, sharey=True)

    if n_scenarios == 1:
        axes = axes.reshape(1, -1)
    elif n_models == 1:
        axes = axes.reshape(-1, 1)

    colors = plt.cm.Accent(np.linspace(0, 1, n_models))

    for scenario_idx, (results, scenario_name) in enumerate(zip(all_results, scenario_names)):
        true_params = results["true_params"]
        change_times = results["change_times"]
        true_pop_sizes = 10 ** np.array(true_params[:-1])

        msmc2_data = baseline_results.get(scenario_name, {}).get('msmc2', None)

        for model_idx, model_data in enumerate(results["models"]):
            ax = axes[scenario_idx, model_idx]

            create_ribbon_plot_with_baselines(
                ax,
                model_data["posterior_samples"],
                true_pop_sizes,
                change_times,
                msmc2_data=msmc2_data,
                color=colors[model_idx],
                show_legend=(scenario_idx == 0 and model_idx == 0),
                ylim=(1e2, 1e5)
            )

            if scenario_idx == 0:
                ax.set_title(model_data["label"], fontsize=14)

    fig.supylabel("Population size", fontsize=16)
    fig.supxlabel("Generations ago", fontsize=16)

    for i, scenario_label in enumerate(args.scenario_labels):
        axes[i, n_models-1].yaxis.set_label_position("right")
        axes[i, n_models-1].set_ylabel(scenario_label, rotation=270, va="center",
                                        fontsize=14, labelpad=20)

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Panel plot with baselines saved to: {args.output}")


if __name__ == "__main__":
    main()
