import os
import pandas as pd
import numpy as np


def annotate_loci():
    


def get_index_of_loci(df, spacing=2.5, start_col='chrom_start', end_col='chrom_end'):
    df['idx_chrom'] = np.floor(df[[start_col, end_col]].mean(axis=1).values / 1e6 / spacing).astype(int)
    df['idx_chrom'] = df[['chrom', 'idx_chrom']].groupby('chrom').apply(
        lambda x: x - x.min(), include_groups=False).reset_index(level=0, drop=True)
    
    chromsizes_bins = (df.groupby('chrom').idx_chrom.max() + 1).sort_values(ascending=False)
    cumsum_bins = pd.Series(index=chromsizes_bins.index, data=np.append(0, chromsizes_bins.values[:-1]).cumsum())
    df['idx_genome'] = cumsum_bins.loc[df.chrom].values + df.idx_chrom

    return df


def get_diff_from_median_offset(df, median_across_chrom=True):
    if median_across_chrom or ('chrom' not in df.columns):
        df['offset_vs_med'] = df.offset - df.offset.median()
        return df
    elif len(df.groupby('chrom').offset.mean().round(2).drop_duplicates()) == 1:
        df['offset_vs_med'] = df.offset - df.offset.median()
        return df
    else:
        return df.groupby('chrom').apply(
            get_diff_from_median_offset, include_groups=False).reset_index(level=0)


def get_gaps_by_mids(df):
    df['gap_from_prev'] = np.nan
    df['gap_to_next'] = np.nan
    idx = df.index.values
    vals = df.loc[idx[1:], 'mid'].values - df.loc[idx[:-1], 'mid'].values
    df.loc[idx[1:], 'gap_from_prev'] = vals
    df.loc[idx[:-1], 'gap_to_next'] = vals
    return df


def get_evenly_spaced_loci(loci, spacing=2.5, cutoff_neighbor=0.05, cutoff_median=0.05, outdir=None, verbose=True):
    if isinstance(loci, str):
        loci = pd.read_csv(loci, sep="\t", header=None, names=['chrom', 'start_bp', 'end_bp'])
    else:
        loci.rename({'chrom_start': 'start_bp', 'chrom_end': 'end_bp'}, axis=1, inplace=True)
    loci['chrom'] = loci.chrom.astype(str).str.replace('chr', '', regex=False)
    loci = loci.sort_values(['chrom', 'start_bp', 'end_bp']).reset_index(drop=True)
    loci = loci[loci.chrom != 'Y']
    loci['locus_size'] = (loci.end_bp - loci.start_bp) / 1e6  # Mb
    loci['mid'] = loci[['start_bp', 'end_bp']].mean(axis=1) / 1e6  # Mb
    loci['offset'] = loci.mid % spacing

    if verbose >= 2:
        print("\nSIZES OF LOCI (bp):", flush=True)
        print((loci.locus_size.describe()).round().astype(int).to_string(), flush=True)
        print(f"{(loci.locus_size > 0.02).sum() / len(loci) * 100:.3g}% of loci are > 20kb", flush=True)
        # print(loci.loc[loci.locus_size > 0.02, ['start', 'end', 'locus_size', 'gap_from_prev', 'gap_to_next']])
        print("\n\nLOCI OFFSETS (compared to median offset):", flush=True)
        print((loci.offset_vs_med).describe().to_string(), flush=True)

    loci = get_diff_from_median_offset(loci)
    loci = loci.groupby('chrom').apply(get_gaps_by_mids, include_groups=False).reset_index(
        level=0)
    if verbose:
        print(f"{len(loci)} loci... gaps between loci:", flush=True)
        print(loci.gap_from_prev.describe().to_string(), flush=True)
    
    # For loci whose midpoints differ by <[cutoff], choose locus with smallest absolute deviation from median offset
    if verbose:
        print((f"\nFor loci whose midpoints differ by <{cutoff_neighbor}, choose locus with smallest"
               " absolute deviation from median offset"), flush=True)
    while (loci.gap_to_next < cutoff_neighbor).sum():
        gap_to_next_pass = loci.gap_to_next.isnull() | (loci.gap_to_next >= cutoff_neighbor)
        gap_from_prev_pass = loci.gap_from_prev.isnull() | (loci.gap_from_prev >= cutoff_neighbor)
        smallgap_prev = loci.loc[~gap_to_next_pass, 'offset_vs_med'].abs()
        smallgap_next = loci.loc[~gap_from_prev_pass, 'offset_vs_med'].abs()
        smallgap_argmin = np.stack([smallgap_prev.values, smallgap_next.values], axis=1).argmin(axis=1)
        smallgap_best = smallgap_prev.index + smallgap_argmin
        loci = loci[(gap_to_next_pass & gap_from_prev_pass) | (loci.index.isin(smallgap_best))].reset_index(drop=True)
        loci = loci.groupby('chrom').apply(get_gaps_by_mids, include_groups=False).reset_index(
            level=0)
        loci = get_diff_from_median_offset(loci)
    if verbose:
        print(f"{len(loci)} loci... gaps between loci:", flush=True)
        print(loci.gap_from_prev.describe().to_string(), flush=True)
    
    # Remove loci whose absolute deviation from median offset >= [cutoff]
    if verbose:
        print(f"\nRemove loci whose absolute deviation from median offset >= {cutoff_median}", flush=True)
    loci = loci[(loci.offset_vs_med.abs() < cutoff_median)].copy()
    loci = loci.groupby('chrom').apply(get_gaps_by_mids, include_groups=False).reset_index(
        level=0)
    loci = get_diff_from_median_offset(loci)
    if verbose:
        print(f"{len(loci)} loci... gaps between loci:", flush=True)
        print(loci.gap_from_prev.describe().to_string(), flush=True)
        print(f"\n{len(loci[loci.gap_from_prev > 4])} loci are spaced >4 Mb apart", flush=True)

    # Get index of loci on each chrom
    locus_idx = np.floor(loci.mid.values / spacing)
    assert np.allclose(locus_idx, locus_idx.astype(int))
    loci = get_index_of_loci(loci, spacing=spacing, start_col='start_bp', end_col='end_bp')

    loci.sort_values(['chrom', 'idx_chrom'], inplace=True)

    # Save
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"loci.spaced{spacing * 1000:g}kb.csv")
        print(f"\nSaving to: {outfile}", flush=True)
        loci.to_csv(outfile, index=False)

    return loci
