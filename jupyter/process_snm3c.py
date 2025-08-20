import numpy as np
import pandas as pd
import os
import cooler


def annotate_loci_snm3c(mcool_file, resolution=0.1, normalize=True, verbose=True):
    if verbose:
        if normalize:
            print("Getting locus information for normalized snm3c-seq data...", flush=True)
        else:
            print("Getting locus information for un-normalized snm3c-seq data...", flush=True)
    snm3c, clr_bins, lengths_clr = load_snm3c(
        mcool_file, resolution=resolution, normalize=normalize, verbose=False)

    lengths_df = clr_bins.copy().rename({'weight': 'snm3c_bias'}, axis=1)
    lengths_df['has_snm3c'] = ~lengths_df.snm3c_isnan
    lengths_df.drop('snm3c_isnan', axis=1, inplace=True)
    lengths_df['chrom'] = lengths_df.chrom.astype(str)
    lengths_df['idx_chrom'] = (lengths_df.start / int(resolution * 1e6)).round().astype(int)
    lengths_df['idx_genome'] = lengths_df.index.values

    assert len(lengths_df) == (lengths_df.groupby('chrom', observed=True).idx_chrom.max() + 1).sum()

    return lengths_df


def load_snm3c(mcool_file, resolution=2.5, normalize=True, mask_last_locus_in_chrom=True, verbose=True):
    if isinstance(normalize, str) and normalize.lower() == 'auto':
        normalize = True if 'Raw.' in mcool_file else False
    if verbose:
        if normalize:
            print("Loading and normalizing snm3c-seq data...", flush=True)
        else:
            print("Loading snm3c-seq data, skipping normalization...", flush=True)

    # Load snm3c-seq data via cooler
    resolution_bp = int(resolution * 1e6)
    uri = f'{mcool_file}::/resolutions/{resolution_bp:d}'
    clr = cooler.Cooler(uri)
    snm3c = clr.matrix(balance=normalize)[:, :]
    if clr.storage_mode != 'symmetric-upper':
        if not np.all(np.isnan(snm3c[np.tril_indices(snm3c.shape[0], -1)]) | (snm3c[np.tril_indices(snm3c.shape[0], -1)] == 0)):
            warnings.warn("snm3C-seq cooler matrix is not symmetric, has different values below diagonal")
        snm3c[np.tril_indices(snm3c.shape[0])] = 0
        snm3c += snm3c.T
    np.fill_diagonal(snm3c, np.nan)
    nan_loci_clr = np.nansum(snm3c, axis=0) == 0
    snm3c[nan_loci_clr, :] = np.nan
    snm3c[:, nan_loci_clr] = np.nan
        
    clr_bins = clr.bins()[:]
    assert len(clr_bins) == snm3c.shape[0]
    clr_bins['mid'] = clr_bins.start + int(round(resolution / 2 * 1e6))
    clr_bins['snm3c_isnan'] = False
    clr_bins.loc[nan_loci_clr, 'snm3c_isnan'] = True

    lengths_clr = clr_bins.groupby('chrom', observed=True).apply(
        lambda x: int(np.ceil(x.end.max() / resolution_bp)), include_groups=False)
    assert (np.ceil((clr.chromsizes / resolution_bp)) == lengths_clr).all()
    assert (clr_bins.groupby('chrom', observed=True).size() == lengths_clr).all()
    assert lengths_clr.sum() == snm3c.shape[0]

    # Remove data from last bin per chromosome, which is associated with fewer bp than other bins
    if mask_last_locus_in_chrom:
        small_bins_check = (clr_bins.groupby('chrom', observed=True).apply(
            lambda x: x.tail(1), include_groups=False)[['end']] % resolution_bp / resolution_bp).reset_index(
            level=1).rename({'level_1': 'idx', 'end': 'ratio'}, axis=1)
        # small_bins_check = small_bins_check[small_bins_check.ratio <= mask_last_locus_in_chrom]
        snm3c[small_bins_check.idx.values, :] = np.nan
        snm3c[:, small_bins_check.idx.values] = np.nan

    return snm3c, clr_bins, lengths_clr
