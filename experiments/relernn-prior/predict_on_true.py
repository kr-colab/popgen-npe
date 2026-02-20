docstring = """
Draw posterior samples for targets simulated under a particular regime ("target"),
using a models trained under another regime ("query").
"""

import os
import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse
import sys
import glob
import pickle

from torch import Tensor
from torch.utils.data import DataLoader
from lightning import LightningModule, Trainer
from sbi.inference import DirectPosterior
from sbi.utils import BoxUniform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "workflow", "scripts"))
import ts_simulators
from data_handlers import ZarrDataset
from utils import get_least_busy_gpu, parse_paths

parser = argparse.ArgumentParser(docstring)
parser.add_argument("--query-configfiles", type=str, nargs="+", 
    help="List of NPE config file for query network", 
    default=glob.glob("npe-config/*.yaml"),
)
parser.add_argument("--target-configfile", type=str, 
    help="NPE config file for target network", 
    default="npe-config/ReLERNN_true_10000.yaml",
)
parser.add_argument("--output-path", type=str, 
    help="Where to save output", 
    default="/sietch_colab/data_share/popgen_npe/relernn_prior/posterior_summaries.pkl",
)
parser.add_argument("--random-seed", type=int, default=1024)
args = parser.parse_args()

rng = np.random.default_rng(args.random_seed)
torch.manual_seed(args.random_seed)

# Determine the device
if torch.cuda.is_available():
    best_gpu = get_least_busy_gpu()
    device = f"cuda:{best_gpu}"
    devices = [best_gpu]  # Set devices to the least busy GPU
else:
    device = "cpu"
    devices = 1  # Ensure CPU compatibility
    
# Get means and credibility intervals for the target test set given
# the various query networks
num_samples = 1000
num_quantiles = 9
alpha_grid = np.linspace(0, 0.5, num_quantiles + 2)[1:-1]

target_config = yaml.safe_load(open(args.target_configfile))
target_paths = parse_paths(target_config)
target_simulator_config = target_config["simulator"]
target_simulator = getattr(ts_simulators, target_simulator_config["class_name"])(target_simulator_config)

model_predictions = {}
for query_configfile in args.query_configfiles:

    print(query_configfile)
    query_config = yaml.safe_load(open(query_configfile))
    query_paths = parse_paths(query_config)
    query_simulator_config = query_config["simulator"]
    query_simulator = getattr(ts_simulators, query_simulator_config["class_name"])(query_simulator_config)
    assert query_simulator.parameters == target_simulator.parameters

    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.embedding_net = torch.load(
                query_paths["embedding_net"],
                weights_only=False,
            )
            self.normalizing_flow = torch.load(
                query_paths["normalizing_flow"], 
                weights_only=False,
            )
    
        def predict_step(self, data):
            targets, features = data
            embeddings = self.embedding_net(features)
            prior = BoxUniform(
                query_simulator.prior.base_dist.low.to(self.device), 
                query_simulator.prior.base_dist.high.to(self.device),
            )
            posterior = DirectPosterior(
                posterior_estimator=self.normalizing_flow,
                prior=prior,
                device=self.device,
            )
            samples = posterior.sample_batched(
                [num_samples], 
                x=embeddings, 
                show_progress_bars=False,
            ).permute(1, 2, 0) # dimensions are (batch, parameter, npe sample)
            return samples, targets

    # posterior sampling
    dataset = ZarrDataset(
        target_paths["zarr"],  # NB: use target test set
        split="sbi_test",
        packed_sequence=query_config["packed_sequence"],
        use_cache=query_config["use_cache"],
        ray_num_workers=10,
    )
    loader = DataLoader(
        dataset,
        batch_size=query_config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
        collate_fn=dataset.make_collate_fn(),
    )
    model = Model()
    trainer = Trainer(
        accelerator="gpu" if device.startswith("cuda") else "cpu",
        devices=devices,
        logger=False,
    )
    samples = trainer.predict(model=model, dataloaders=loader)
    posterior_draws = torch.cat([p[0] for p in samples]).cpu().numpy()
    true_values = torch.cat([p[1] for p in samples]).cpu().numpy()
    
    # Get true prior intervals and coverage
    prior_dataset = ZarrDataset(
        target_paths["zarr"], 
        split="sbi_train", 
        packed_sequence=target_config["packed_sequence"],
        use_cache=True, 
        ray_num_workers=10,
    )
    true_mean = np.mean(true_values, axis=0)
    true_lower = np.quantile(true_values, alpha_grid, axis=0).T
    true_upper = np.quantile(true_values, 1 - alpha_grid, axis=0).T
    true_coverage = np.logical_and(
        true_values[:, :, np.newaxis] >= true_lower[np.newaxis, :, :],
        true_values[:, :, np.newaxis] <= true_upper[np.newaxis, :, :],
    ).mean(axis=0)
    print(true_coverage)

    print(true_lower.shape, true_upper.shape, true_coverage.shape)

    # Get misspecified prior intervals and coverage
    prior_dataset = ZarrDataset(
        query_paths["zarr"], 
        split="sbi_train", 
        packed_sequence=query_config["packed_sequence"],
        use_cache=True, 
        ray_num_workers=10,
    )
    prior_samples = prior_dataset.theta.numpy()
    prior_mean = prior_samples.mean(axis=0)
    prior_lower = np.quantile(prior_samples, alpha_grid, axis=0).T
    prior_upper = np.quantile(prior_samples, 1 - alpha_grid, axis=0).T
    prior_coverage = np.logical_and(
        true_values[:, :, np.newaxis] >= prior_lower[np.newaxis, :, :],
        true_values[:, :, np.newaxis] <= prior_upper[np.newaxis, :, :],
    ).mean(axis=0)
    print(prior_coverage)

    
    # Get posterior intervals and coverage
    posterior_mean = posterior_draws.mean(axis=-1)
    posterior_lower = np.quantile(posterior_draws, alpha_grid, axis=-1).transpose(1, 2, 0)
    posterior_upper = np.quantile(posterior_draws, 1 - alpha_grid, axis=-1).transpose(1, 2, 0)
    posterior_coverage = np.logical_and(
        true_values[:, :, np.newaxis] >= posterior_lower,
        true_values[:, :, np.newaxis] <= posterior_upper,
    ).mean(axis=0)
    print(posterior_coverage)

    print(posterior_lower.shape, posterior_upper.shape, posterior_coverage.shape)

    model_predictions[query_configfile] = {
        "alpha_grid": alpha_grid,
        "posterior_mean": posterior_mean,
        "posterior_lower": posterior_lower,
        "posterior_upper": posterior_upper,
        "posterior_coverage": posterior_coverage,
        "prior_mean": prior_mean,
        "prior_lower": prior_lower,
        "prior_upper": prior_upper,
        "prior_coverage": prior_coverage,
        "true_mean": true_mean,
        "true_lower": true_lower,
        "true_upper": true_upper,
        "true_coverage": true_coverage,
    }

pickle.dump(model_predictions, open(args.output_path, "wb"))
