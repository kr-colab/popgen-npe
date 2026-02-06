#!/usr/bin/env python3
"""
Utilities for running MSMC2 on tree sequences.

MSMC2 estimates effective population size trajectories from phased genotype data.
"""

import subprocess
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path


# Default MSMC2 binary location
MSMC2_DEFAULT = "/home/adkern/analysis2/ext/msmc2/build/release/msmc2"


def write_multihetsep(ts, output_path, chrom_name="1"):
    """
    Convert tree sequence to MSMC2 multihetsep format.

    MSMC2 input format (tab-separated):
    chrom  position  span  genotypes

    Where genotypes is a string of 0s and 1s for each haplotype.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Input tree sequence
    output_path : str
        Path to write the output file
    chrom_name : str
        Chromosome name to use in output
    """
    with open(output_path, 'w') as f:
        prev_pos = 0
        for var in ts.variants():
            cur_pos = int(var.site.position)
            if cur_pos > prev_pos:
                span = cur_pos - prev_pos
                geno = ''.join(map(str, var.genotypes))
                f.write(f"{chrom_name}\t{cur_pos}\t{span}\t{geno}\n")
                prev_pos = cur_pos


def subsample_tree_sequence(ts, n_diploids=4, seed=None):
    """
    Subsample diploid individuals from tree sequence.

    MSMC2 works optimally with 2-8 haplotypes (1-4 diploids).

    Parameters
    ----------
    ts : tskit.TreeSequence
        Input tree sequence
    n_diploids : int
        Number of diploid individuals to sample
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tskit.TreeSequence
        Simplified tree sequence with subsampled individuals
    """
    if seed is not None:
        np.random.seed(seed)

    n_haplotypes = n_diploids * 2
    if n_haplotypes > ts.num_samples:
        raise ValueError(f"Requested {n_haplotypes} haplotypes but tree sequence only has {ts.num_samples}")

    # Sample haplotype indices (pairing consecutive as diploids)
    all_samples = ts.samples()
    n_total_diploids = len(all_samples) // 2

    # Select random diploid individuals
    selected_diploids = np.random.choice(n_total_diploids, n_diploids, replace=False)
    selected_haplotypes = []
    for d in selected_diploids:
        selected_haplotypes.extend([all_samples[2*d], all_samples[2*d + 1]])

    return ts.simplify(samples=selected_haplotypes)


def run_msmc2(input_files, output_prefix, msmc2_path=MSMC2_DEFAULT,
              time_pattern="1*2+25*1+1*2+1*3", iterations=20, n_threads=1,
              fixed_recombination=False, rho_over_mu=None):
    """
    Run MSMC2 on input files.

    Parameters
    ----------
    input_files : list of str
        Input multihetsep files
    output_prefix : str
        Prefix for output files
    msmc2_path : str
        Path to MSMC2 binary
    time_pattern : str
        Time segment pattern
    iterations : int
        Number of EM iterations (default 20)
    n_threads : int
        Number of threads
    fixed_recombination : bool
        If True, keep recombination rate fixed (-R flag)
    rho_over_mu : float, optional
        Initial ratio of recombination over mutation rate (-r flag)

    Returns
    -------
    str
        Path to the loop output file (more reliable than final.txt)
    """
    if isinstance(input_files, str):
        input_files = [input_files]

    # Filter out empty files
    non_empty = [f for f in input_files if os.path.getsize(f) > 0]
    if not non_empty:
        raise ValueError("All input files are empty")

    cmd = [
        msmc2_path,
        "-o", output_prefix,
        "-p", time_pattern,
        "-i", str(iterations),
        "-t", str(n_threads),
    ]

    if fixed_recombination:
        cmd.append("--fixedRecombination")

    if rho_over_mu is not None:
        cmd.extend(["-r", str(rho_over_mu)])

    cmd.extend(non_empty)

    # Run MSMC2 - it may fail on optimization but still produce usable output
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check if we got output files even if MSMC2 crashed
    loop_file = f"{output_prefix}.loop.txt"
    final_file = f"{output_prefix}.final.txt"

    if os.path.exists(final_file) and os.path.getsize(final_file) > 0:
        return final_file
    elif os.path.exists(loop_file) and os.path.getsize(loop_file) > 0:
        print(f"  Note: MSMC2 did not produce final.txt, using loop.txt instead")
        return loop_file
    else:
        # Re-raise with the error message
        if result.returncode != 0:
            raise RuntimeError(f"MSMC2 failed: {result.stderr}")
        raise RuntimeError("MSMC2 produced no output files")


def parse_msmc2_output(results_file, mutation_rate):
    """
    Parse MSMC2 output and convert to Ne estimates.

    MSMC2 outputs times and rates scaled by the mutation rate per basepair
    per generation. To convert:
    - Time (generations) = scaled_time / mutation_rate
    - Ne = 1 / (2 * lambda * mutation_rate)

    Parameters
    ----------
    results_file : str
        Path to MSMC2 .final.txt or .loop.txt output
    mutation_rate : float
        Mutation rate per base per generation

    Returns
    -------
    dict
        Dictionary with 'times' and 'ne' arrays
    """
    if results_file.endswith('.loop.txt'):
        return parse_msmc2_loop_file(results_file, mutation_rate)
    else:
        return parse_msmc2_final_file(results_file, mutation_rate)


def parse_msmc2_final_file(results_file, mutation_rate):
    """Parse MSMC2 .final.txt format."""
    df = pd.read_csv(results_file, sep='\t')

    # MSMC2 output columns: time_index, left_time_boundary, right_time_boundary, lambda
    # Convert scaled times to generations
    times_left = df['left_time_boundary'].values / mutation_rate
    times_right = df['right_time_boundary'].values / mutation_rate

    # Use midpoint of each time interval
    times = (times_left + times_right) / 2

    # Convert lambda (coalescence rate) to Ne
    # Ne = 1 / (2 * lambda * mu)
    lambdas = df['lambda'].values
    ne = 1.0 / (2 * lambdas * mutation_rate)

    return {
        'times': times,
        'times_left': times_left,
        'times_right': times_right,
        'ne': ne,
        'lambda': lambdas
    }


def parse_msmc2_loop_file(results_file, mutation_rate):
    """
    Parse MSMC2 .loop.txt format and extract the iteration with the best
    log likelihood.

    Loop file format (tab-separated):
    recombination_rate  log_likelihood  time_boundaries  ne_values

    Where time_boundaries and ne_values are comma-separated strings.
    """
    with open(results_file, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Find iteration with best (highest) log likelihood
    best_idx = 0
    best_ll = -np.inf
    for i, line in enumerate(lines):
        ll = float(line.split('\t')[1])
        if ll > best_ll:
            best_ll = ll
            best_idx = i

    parts = lines[best_idx].split('\t')

    # Parse time boundaries (scaled)
    time_boundaries_str = parts[2]
    time_boundaries = [float(x) if x != 'inf' else np.inf
                       for x in time_boundaries_str.split(',')]
    time_boundaries = np.array(time_boundaries)

    # Parse Ne values
    ne_values_str = parts[3]
    ne_values = np.array([float(x) for x in ne_values_str.split(',')])

    # Convert scaled times to generations
    # Use the midpoint of each time bin
    times_left = time_boundaries[:-1] / mutation_rate
    times_right = time_boundaries[1:] / mutation_rate

    # Handle infinite boundaries
    finite_mask = np.isfinite(times_left) & np.isfinite(times_right)
    times_left = times_left[finite_mask]
    times_right = times_right[finite_mask]
    ne_values = ne_values[finite_mask]

    times = (times_left + times_right) / 2

    # Only filter out infinite values
    valid_mask = np.isfinite(ne_values)
    times = times[valid_mask]
    ne_values = ne_values[valid_mask]
    times_left = times_left[valid_mask]
    times_right = times_right[valid_mask]

    return {
        'times': times,
        'times_left': times_left,
        'times_right': times_right,
        'ne': ne_values,
    }


def run_msmc2_pipeline(ts, output_dir, mutation_rate, n_diploids=4, seed=42,
                       msmc2_path=MSMC2_DEFAULT, time_pattern="1*2+25*1+1*2+1*3",
                       n_threads=1, iterations=20, fixed_recombination=False,
                       rho_over_mu=None):
    """
    Full MSMC2 pipeline from tree sequence to Ne estimates.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Input tree sequence
    output_dir : str
        Directory for output files
    mutation_rate : float
        Mutation rate per base per generation
    n_diploids : int
        Number of diploid individuals to use (default 4)
    seed : int
        Random seed for subsampling
    msmc2_path : str
        Path to MSMC2 binary
    time_pattern : str
        MSMC2 time segment pattern
    n_threads : int
        Number of threads for MSMC2
    iterations : int
        Number of EM iterations
    fixed_recombination : bool
        If True, keep recombination rate fixed
    rho_over_mu : float, optional
        Initial ratio of recombination over mutation rate

    Returns
    -------
    dict
        Dictionary with MSMC2 results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subsample tree sequence
    ts_sub = subsample_tree_sequence(ts, n_diploids=n_diploids, seed=seed)
    print(f"  Subsampled to {ts_sub.num_samples} haplotypes ({n_diploids} diploids)")
    print(f"  Sites in subsampled ts: {ts_sub.num_sites}")

    # Write multihetsep file
    input_file = output_dir / "msmc2_input.txt"
    write_multihetsep(ts_sub, input_file)
    print(f"  Wrote input file: {input_file}")

    # Check file has content
    if os.path.getsize(input_file) == 0:
        raise ValueError("Input file is empty - no variants in tree sequence")

    # Run MSMC2
    output_prefix = output_dir / "msmc2_output"
    final_file = run_msmc2(
        str(input_file),
        str(output_prefix),
        msmc2_path=msmc2_path,
        time_pattern=time_pattern,
        n_threads=n_threads,
        iterations=iterations,
        fixed_recombination=fixed_recombination,
        rho_over_mu=rho_over_mu
    )
    print(f"  MSMC2 output: {final_file}")

    # Parse results
    results = parse_msmc2_output(final_file, mutation_rate)
    print(f"  Parsed {len(results['times'])} time points")

    return results
