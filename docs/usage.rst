Usage
=====

This package is a Snakemake workflow for simulation-based inference in population genetics.
The goal of this package is to provide a flexible and modular framework for running
neural posterior estimation for population genetic inference.

The workflow is designed to be executed from the command line, and it can run either on a local machine
or a high performance computing cluster. The entire pipeline is managed by the Snakemake workflow manager.

Important files in this package include:

- ``workflow/training_workflow.smk``: The main training workflow file. This file contains the rules and
  functions that define the training workflow, executing a complete neural posterior
  estimation process based on a given configuration.
- ``workflow/prediction_workflow.smk``: The prediction workflow file for running inference on VCF data.
- ``environment.yaml``: The Conda environment file for the workflow that lists all necessary dependencies.

In addition, the workflow relies on a configuration file (in YAML format) that contains the parameters for both
simulation and inference. For example, the file ``workflow/config/AraTha_2epoch_cnn.yaml`` is used to run neural
posterior estimation for a two-epoch demographic model with a CNN embedding network.

Basic Usage
-----------

Training (simulation → feature extraction → neural posterior estimation):

.. code-block:: bash

    snakemake --configfile workflow/config/AraTha_2epoch_cnn.yaml \
              --snakefile workflow/training_workflow.smk

Prediction (apply trained models to a VCF using windowed inference):

.. code-block:: bash

    snakemake --configfile workflow/config/AraTha_2epoch_cnn.yaml \
              --snakefile workflow/prediction_workflow.smk

Both commands assume the config file defines where results live (``project_dir``), how many scatter/gather chunks to use (``n_chunk``) and, for predictions, how to window the VCF (``prediction.n_chunk`` plus BED windows).

Configuration Files
-------------------

The configuration file is organized into several sections controlling various aspects of the workflow:

1. **Project Directory**:

   - ``project_dir``: Specifies the directory containing the project files. Adjust this path to point to your
     actual project directory.

2. **Resource Allocation**:

   - ``cpu_resources``: Defines resources for CPU-only tasks, including:
     
     - ``runtime``: Maximum time allocated for the task.
     - ``mem_mb``: Memory (in MB) allocated for the task.
     
   - ``gpu_resources``: Defines resources for GPU tasks, including:
     
     - ``runtime``: Maximum time allocated for GPU tasks.
     - ``mem_mb``: Memory (in MB) allocated for GPU tasks.
     - ``gpus``: Number of GPUs to be used.
     - ``slurm_partition``: SLURM partition to use for job scheduling.
     - ``slurm_extra``: Additional SLURM options for GPU allocation.

3. **Simulation Parameters**:

   - ``random_seed``: Global seed for reproducible simulator, processor, and training behavior.
   - ``n_chunk``: Number of scatter/gather chunks used during simulation/processing. The workflow divides the total number of simulations (``n_train + n_val + n_test``) evenly across ``n_chunk`` helper YAML files so each worker operates on a disjoint slice of the Zarr store. Increasing ``n_chunk`` yields more parallel jobs with fewer simulations per job; decreasing it reduces concurrency at the cost of larger jobs.
   - ``n_train``, ``n_val``, ``n_test``: Counts of training, validation, and test simulations.

4. **Model Training Configuration**:

   - ``train_embedding_net_separately``: Boolean flag indicating whether to train the embedding network separately
     from the normalizing flow.
   - ``use_cache``: Boolean flag indicating whether to load features into CPU memory.
   - ``optimizer``: The optimization algorithm to be used (e.g., "Adam").
   - ``batch_size``: The size of the batches used during training.
   - ``learning_rate``: The learning rate for the optimizer.
   - ``max_num_epochs``: Maximum number of training epochs.
   - ``stop_after_epochs``: Number of epochs with no improvement after which training should stop.
   - ``clip_max_norm``: Maximum norm for gradient clipping.
   - ``packed_sequence``: Boolean flag indicating whether to use packed sequences during training.

5. **Simulator Configuration** (see :doc:`simulators` for details):

   - ``simulator``: Contains ``class_name`` plus any overrides for attributes listed in the simulator's ``default_config`` (e.g., ``samples``, ``sequence_length``, ``mutation_rate``). All keys must be supported by the selected simulator.

6. **Processor Configuration** (see :doc:`processors`):

   - ``processor``: Contains ``class_name`` plus any supported keyword arguments (e.g., ``n_snps``, ``maf_thresh``). Unsupported keys raise an error at runtime.

7. **Embedding Network Configuration**:

   - ``embedding_network``: Specifies the neural architecture that consumes processor outputs. Each block contains ``class_name`` and architecture-specific kwargs (e.g., ``ExchangeableCNN`` uses ``output_dim``, ``input_rows``, ``input_cols``; ``SummaryStatisticsEmbedding`` ignores these extras).

8. **Prediction Configuration** (only needed for ``prediction_workflow.smk``):

   - ``prediction``: Controls how the workflow subsets a VCF into windows and ensures compatibility with the training simulator.
     - ``vcf``: Path to the compressed VCF to process.
     - ``ancestral_fasta``: FASTA file providing ancestral alleles (optional; if omitted, the reference allele is assumed ancestral).
     - ``population_map``: YAML mapping sample IDs to simulator population names. Sample counts must match the simulator defaults so that inferred trees and trained networks share the same structure.
     - ``windows``: BED file describing genomic windows to analyze.
     - ``min_snps_per_window``: Minimum number of segregating variants required for a window to be processed.
     - ``n_chunk``: Number of scatter/gather chunks for prediction. ``setup_prediction.py`` caps this at the number of valid windows and divides them evenly across YAML files, mirroring how ``n_chunk`` works for training.

When running ``prediction_workflow.smk``, the ``simulator``, ``processor``, and ``embedding_network`` sections are reused exactly as they were during training: tree sequences inferred from the VCF are routed through the same processor class, embedded by the saved neural network, and scored by the trained normalizing flow. Keeping these sections synchronized between training and prediction avoids dimension or population-mismatch errors.
