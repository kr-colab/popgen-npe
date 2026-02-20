docstring = """
Plot posterior coverage and concentration as a function of increasing numbers of training
samples, across misspecified priors.
"""
import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np
from matplotlib.ticker import ScalarFormatter

parser = argparse.ArgumentParser(docstring)
parser.add_argument("--posterior-summary", type=str, 
    help="Pickled coverage estimates per model", 
    default="/sietch_colab/data_share/popgen_npe/relernn_prior/posterior_summaries.pkl",
)
parser.add_argument("--output-path", type=str, 
    help="Where to save output plot", 
    #default="/sietch_colab/data_share/popgen_npe/relernn_prior/coverage_comparison.png",
    default="/home/natep/public_html/popgennpe/relernn_coverage_comparison.png",
)
args = parser.parse_args()


data = pickle.load(open(args.posterior_summary, "rb"))

training_size_map = {
    "prior": r"$0$ (prior)",
    "100": r"$10^2$",
    "1000": r"$10^3$",
    "10000": r"$10^4$",
    "100000": r"$10^5$",
}
prior_name_map = {
    "true": "Exact prior\n" + r"Unif$(0, 10^{-8})$",
    "wide": "Less concentrated prior\n" + r"Unif$(0, 10^{-7})$",
    "concentrated": "More concentrated prior\n" + r"$10^{-8} \times$ Beta$(2, 2)$",
}

fig, axs = plt.subplots(2, 3, figsize=(8, 4.5), sharex=True, sharey="row", constrained_layout=True)
cmap = plt.get_cmap("magma")
cdim = len(training_size_map)
line_kwargs = {"markersize": 4}

# coverage
for u, prior in enumerate(["true", "wide", "concentrated"]):
    for i, training_size in enumerate(["100", "1000", "10000"]):
        config_name = f"npe-config/ReLERNN_{prior}_{training_size}.yaml"
        df = data[config_name]
        xax = 1 - 2 * df["alpha_grid"]
        if training_size == "100":
            axs[0, u].plot(xax, 
                df["prior_coverage"].flatten(), "-o", 
                label=training_size_map["prior"], color=cmap(0 / cdim), 
                **line_kwargs,
            )
        axs[0, u].plot(xax, 
            df["posterior_coverage"].flatten(), "-o", 
            label=training_size_map[training_size], color=cmap((i+1)/cdim), 
            **line_kwargs,
        )
    axs[0, u].axline((0,0), slope=1, linestyle="dashed", color="gray")
    axs[0, u].set_title(prior_name_map[prior], size=10)
    axs[0, u].set_ylim(0, 1.0)
    if u == 0:
        axs[0, u].set_ylabel("Posterior\ninterval coverage")

# concentration
line_kwargs = {"markersize": 4}
for u, prior in enumerate(["true", "wide", "concentrated"]):
    for i, training_size in enumerate(["100", "1000", "10000"]):
        config_name = f"npe-config/ReLERNN_{prior}_{training_size}.yaml"
        df = data[config_name]
        xax = 1 - 2 * df["alpha_grid"]
        true_width = df["true_upper"] - df["true_lower"]
        if training_size == "100":
            prior_width = df["prior_upper"] - df["prior_lower"]
            axs[1, u].plot(xax, 
                prior_width.flatten() / true_width.flatten(), "-o", 
                label=training_size_map["prior"], 
                color=cmap(0 / cdim), 
                **line_kwargs,
            )
        posterior_width = np.mean(df["posterior_upper"] - df["posterior_lower"], axis=0)
        axs[1, u].plot(xax, 
            posterior_width.flatten() / true_width.flatten(), "-o", 
            label=training_size_map[training_size], 
            color=cmap((i+1)/cdim), 
            **line_kwargs,
        )
    axs[1, u].axhline(y=1.0, linestyle="dashed", color="gray")
    axs[1, u].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1, u].set_xlim(0, 1.0)
    axs[1, u].set_ylim(0.1, 10.0)
    axs[1, u].set_yscale("log")
    axs[1, u].set_yticks([0.1, 1, 10])
    axs[1, u].set_yticklabels(["0.1", "1", "10"])
    if u == 0:
        axs[1, u].set_ylabel("Posterior / true\ninterval width")

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title="Training set size", loc="outside right", frameon=False)
fig.supxlabel("Expected interval coverage", size=10)

plt.savefig(args.output_path)



