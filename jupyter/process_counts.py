import numpy as np
import pandas as pd
import os
import re
from scipy import sparse
from iced.io import write_counts, write_lengths
from topsy.plot.plot_distances import plot_distance_matrix
from topsy.plot.plot_counts import plot_counts_single


def get_nghbr_bins(matrix, lengths):
    mask_intermol_nghbr = np.tile(lengths, 2).cumsum()[:-1] - 1
    nghbr_bins = np.diagonal(matrix, offset=1).copy().astype(float)
    nghbr_bins[mask_intermol_nghbr] = np.nan
    return nghbr_bins


def get_beta_ua(counts, lengths):
    # Get beta such that distance between neighbor beads is 1
    beta_ua = np.nanmean(get_nghbr_bins(counts, lengths=lengths))
    return beta_ua


def get_unambig_counts(lengths_df, counts, matrices, nreads, outdir_counts=None):
    if outdir_counts is not None:
        os.makedirs(outdir_counts, exist_ok=True)

    # Lengths & chromosome names
    lengths_s = lengths_df.groupby('chrom').size().sort_values(ascending=False)
    lengths = lengths_s.values
    if outdir_counts is not None:
        lengths_df = lengths_df.copy()
        np.savetxt(os.path.join(outdir_counts, "chromosomes.txt"), lengths_s.index.values, fmt="%s")
        lengths_df.columns = [f"#{c}" for c in lengths_df.columns]
        lengths_df.to_csv(os.path.join(outdir_counts, "counts.bed"), index=False, header=True, sep="\t")

    # Counts
    counts_int = np.triu(counts, 1)
    counts_int = (counts_int * nreads / counts_int.sum()).round().astype(int)
    if outdir_counts is not None:
        write_counts(os.path.join(outdir_counts, "ua_counts.matrix"), counts_int)

    # Distances, scaled such that mean distance between neighbor beads is 1
    dis_scaled = {}
    scale_factors = {}
    mask_intermol_nghbr = np.tile(lengths_s.values, 2).cumsum()[:-1] - 1
    for agg_func in ['mean', 'median']:
        nghbr_dis = np.diagonal(matrices[f'dis_{agg_func}'], offset=1).copy()
        nghbr_dis[mask_intermol_nghbr] = np.nan
        nghbr_dis_mean = np.nanmean(nghbr_dis)
        scale_factors[agg_func] = nghbr_dis_mean
        dis_scaled[agg_func] = matrices[f'dis_{agg_func}'] / nghbr_dis_mean
        if outdir_counts is not None:
            np.save(os.path.join(outdir_counts, f"distances_true.{agg_func}.npy"), dis_scaled[agg_func])

    # Get beta such that distance between neighbor beads is 1
    est_beta_ua = get_beta_ua(counts=counts_int, lengths=lengths_s.values)

    # Metadata
    if outdir_counts is not None:        
        dataset_info = pd.Series({
            'ploidy': 2, 'nreads': counts_int.sum(), 'ua': 1, 'pa': 0, 'lengths': lengths_s.values,
            'beta': None, 'beta_ua': None, 'alpha': None})
        for agg_func in ['mean', 'median']:
            dataset_info[f"nghbr_dis_mean.{agg_func}"] = scale_factors[agg_func]
        dataset_info.to_csv(os.path.join(outdir_counts, "dataset_info.txt"), sep="\t", header=False)

    # Plot counts & distances
    if outdir_counts is not None:
        outdir
        plot_counts_single(
            counts_int, lengths=lengths, title="Pseudo-counts, unambiguous",
            outfile=os.path.join(outdir_counts, "images", "ua_counts.png"), mark_excluded=True)
        for agg_func in ['mean', 'median']:
            plot_distance_matrix(
                dis_scaled[agg_func], lengths=lengths, ploidy=2, title=f"'True' distances\n{agg_func} across cells",
                outfile=os.path.join(outdir_counts, "images", f"distances_true.{agg_func}.png"))
    
    return counts_int, dis_scaled, lengths

