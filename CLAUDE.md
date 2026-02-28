# CLAUDE.md

## Overview

popgensbi is a Snakemake workflow for simulation-based inference (SBI) in population genetics. It uses neural posterior estimation (NPE) to infer demographic parameters from genetic data by training embedding networks and normalizing flows on simulated data.

## Commands

```bash
# Environment
conda env create -f environment.yaml
conda activate popgen_npe_env

# Training workflow
snakemake --configfile workflow/config/AraTha_2epoch_cnn.yaml --snakefile workflow/training_workflow.smk

# Prediction workflow
snakemake --configfile workflow/config/AraTha_2epoch_cnn.yaml --snakefile workflow/prediction_workflow.smk

# Tests
pytest tests/test_workflow_script.py

# Docs
cd docs && make html
```

## Architecture

**Training** (`workflow/training_workflow.smk`):
Setup → Simulate tree sequences → Process to tensors → Train embedding network → Train normalizing flow → Generate diagnostics

**Prediction** (`workflow/prediction_workflow.smk`):
VCF → Zarr conversion → Window processing → Apply trained model → Posterior samples

### Core Components (in `workflow/scripts/`)

| File | Purpose |
|------|---------|
| `ts_simulators.py` | Demographic models (AraTha_2epoch, YRI_CEU, DroMel_CO_FR, VariablePopulationSize) |
| `ts_processors.py` | Tree sequence → tensor converters (cnn_extract, genotypes_and_distances, tskit_sfs) |
| `embedding_networks.py` | Neural architectures (ExchangeableCNN, RNN, SPIDNA, SummaryStatisticsEmbedding) |
| `data_handlers.py` | ZarrDataset with Ray parallel loading |

### Configuration

YAML configs in `workflow/config/` specify simulator, processor, embedding network, and training parameters. Project directories use UID-based naming:
```
{project_dir}/{SIMULATOR}-{PROCESSOR}-{EMBEDDING}-{RANDOM_SEED}-{N_TRAIN}-{sep|e2e}/
```
