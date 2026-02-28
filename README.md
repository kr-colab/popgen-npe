# Snakemake workflow: `popgen-npe`

[![Snakemake](https://img.shields.io/badge/snakemake-≥6.3.0-brightgreen.svg)](https://snakemake.github.io)
[![Read the Docs](https://img.shields.io/readthedocs/pip/stable.svg)](https://popgen-npe.readthedocs.io/en/latest/)



A Snakemake workflow for neural posterior estimation in population genetics. It orchestrates end-to-end simulation, feature extraction, neural posterior estimation (“training”), and windowed inference on VCFs (“prediction”).


## Documentation

The documentation covers simulator/processor APIs, configuration options, and usage walkthroughs: <https://popgen-npe.readthedocs.io/en/latest/>. Building it locally requires Sphinx (see `docs/`).

## Workflows at a glance

- `workflow/training_workflow.smk`: runs the full neural posterior estimation pipeline (simulate → process → train embedding + normalizing flow).
- `workflow/prediction_workflow.smk`: infers trees from a VCF, processes them, and applies the trained posterior estimator window-by-window.

## Environment setup

Create and activate a Conda environment that includes Snakemake and the workflow dependencies:

```bash
conda env create -f environment.yaml
conda activate popgen_npe_env
```

## Training workflow

1. Copy or adapt one of the templates in `workflow/config/` (e.g. `AraTha_2epoch_cnn.yaml`). This YAML encapsulates:
   - Simulation settings (`simulator`, `n_train`, `n_val`, `n_test`, `n_chunk`)
   - Feature extraction (`processor`)
   - Embedding + posterior estimator hyperparameters (`embedding_network`, optimizer, etc.)
2. Launch the workflow:

```bash
snakemake --cores 8 \
          --configfile workflow/config/AraTha_2epoch_cnn.yaml \
          --snakefile workflow/training_workflow.smk
```

The run directory defaults to `<project_dir>/<sim>-<processor>-<embedding>-<seed>-<n_train>-e2e` unless you enable separate embedding pretraining. Refer to the [Usage](https://popgen-npe.readthedocs.io/en/latest/usage.html) docs for a field-by-field description of the config file.

## Example data for prediction

You can generate example inputs (VCF + ancillary files) compatible with the AraTha config using:

```bash
python resources/util/simulate-vcf.py \
  --outpath example_data/AraTha_2epoch \
  --window-size 1000000 \
  --configfile workflow/config/AraTha_2epoch_cnn.yaml
```

This script simulates data, writes the VCF, ancestral FASTA, population map, and a BED file of windows. These paths correspond to the defaults under the `prediction` block of `workflow/config/AraTha_2epoch_cnn.yaml`.

## Prediction workflow

To apply a trained model to a VCF, ensure your config file contains a `prediction` block with:

- `vcf`: gzipped VCF path
- `ancestral_fasta`: ancestral sequence for the same contigs (optional; reference allele is assumed if omitted)
- `population_map`: YAML mapping VCF sample IDs to simulator population names (sample counts must match the simulator defaults)
- `windows`: BED file describing genomic windows
- `min_snps_per_window`: minimum segregating variants per window
- `n_chunk`: number of scatter/gather jobs for prediction

Then run:

```bash
snakemake --cores 8 \
          --configfile workflow/config/AraTha_2epoch_cnn.yaml \
          --snakefile workflow/prediction_workflow.smk
```

Predictions, inferred trees, and QC plots are written to `<project_dir>/<vcf_basename>/`.

## Cluster usage quickstart

We ship an example Snakemake profile in `example_profile/` targeting a SLURM cluster (tested on UO’s kerngpu partition). To use it:

```bash
snakemake --executor slurm \
          --workflow-profile ~/.config/snakemake/yourprofile \
          --configfile workflow/config/AraTha_2epoch_cnn.yaml \
          --snakefile workflow/training_workflow.smk
```

Notes:

- The profile assumes Snakemake 8.x’s executor interface. Adjust `example_profile/config.yaml` if you are pinned to an older version.
- The workflow YAML `cpu_resources`/`gpu_resources` blocks drive per-rule SLURM resource requests (runtime/memory/GPU count, partitions, constraints, etc.).






