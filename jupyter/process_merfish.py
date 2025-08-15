#!/usr/bin/env python3

import os
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tqdm import tqdm
from process_loci import get_evenly_spaced_loci


def remove_small_traces(df, nloci_per_chrom, min_nloci=3, min_nloci_ratio=0.1, by_chosen_loci_only=True):
    df['cutoff'] = np.maximum(min_nloci, min_nloci_ratio * nloci_per_chrom.loc[df.chrom.values].values)
    if by_chosen_loci_only:
        df = df.groupby('chrom').apply(lambda x: (
            x if x.chosen_loci.sum() >= x.cutoff.values[0] else None), include_groups=False).reset_index(level=0)
    else:
        df = df.groupby('chrom').apply(lambda x: (
            x if len(x) >= x.cutoff.values[0] else None), include_groups=False).reset_index(level=0)
    if len(df):
        return df.drop('cutoff', axis=1)


def remove_cells_with_extra_traces(df, max_nchrom=1):
    nchrom_with_extra_traces = df.groupby('chrom').apply(
        lambda x: len(x.trace_id.drop_duplicates()) > 2, include_groups=False).sum()
    if nchrom_with_extra_traces <= max_nchrom:
        return df


def get_cov_per_locus(df, min_ndetected_per_locus=None, verbose=True):
    """Remove loci where <[cutoff]% of cells are non-missing from one or both of the traces"""
    nrows_orig = len(df)
    ncells = len(df[['cell_id']].drop_duplicates())
    grouping_cols = ['chrom', 'chrom_order']
    if 'chosen_loci' in df.columns:
        grouping_cols.append('chosen_loci')
    cov_per_locus = df.groupby(grouping_cols).size().rename("cell_cov").to_frame()
    if 'chosen_loci' in df.columns:
        cov_per_locus.reset_index(level=2, inplace=True)
    cov_per_locus /= ncells
    if min_ndetected_per_locus is not None:
        cov_per_locus['pass_cutoff'] = cov_per_locus.cell_cov >= min_ndetected_per_locus
        pass_cutoff = cov_per_locus[cov_per_locus.pass_cutoff].index
        df.set_index(['chrom', 'chrom_order'], inplace=True)
        df = df[df.index.isin(pass_cutoff)].reset_index()
        if verbose:
            if 'chosen_loci' in df.columns:
                tmp = f" (={((~cov_per_locus.pass_cutoff) & cov_per_locus.chosen_loci).sum()}/{cov_per_locus.chosen_loci.sum()} chosen loci)"
            else:
                tmp = ""
            print((f"Removed {(~cov_per_locus.pass_cutoff).sum()}/{len(cov_per_locus)} LOCI{tmp}"
                   f" that were detected in <{min_ndetected_per_locus * 100:g}% of molecules"), flush=True)
            print(f" ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)
            print('   ' + cov_per_locus[cov_per_locus.pass_cutoff].describe().drop('count').to_string().replace(
                '\n', '\n   '), flush=True)
    else:
        print("\tRatio of cells in which each locus was detected:", flush=True)
        print('\t   ' + cov_per_locus.describe().drop('count').to_string().replace('\n', '\n\t   '), flush=True)
    
    return df, cov_per_locus


def filter_data(df, max_nchrom_gt2trace=None, min_ndetected_per_locus=0.1, trace_min_nloci=None,
                trace_min_nloci_ratio=None, by_chosen_loci_only=True, ntraces_per_cell=None, 
                enforce_even_spacing=True, spacing=2.5, verbose=True):
    nrows_orig = len(df)
    if verbose:
        print(f"FILTERING... original n={len(df):,}", flush=True)

    if min_ndetected_per_locus is None:
        min_ndetected_per_locus = 0  # Skip filtering on this
    if trace_min_nloci is None:
        trace_min_nloci = 0  # Skip filtering on this
    if trace_min_nloci_ratio is None:
        trace_min_nloci_ratio = 0  # Skip filtering on this

    if ntraces_per_cell is not None and 'ntraces_per_cell' not in df.columns:
        df = get_ntraces_per_cell(df)  # Determine *prior to filtering*

    # ==== Remove cells where >[cutoff] chromosomes have >2 traces
    if (max_nchrom_gt2trace is not None) and (max_nchrom_gt2trace < len(
            df.chrom.drop_duplicates()) and (len(df.trace_id.drop_duplicates()) > 0):
        if max_nchrom_gt2trace == 0:
            ncells_orig = len(df.cell_id.drop_duplicates())
            df = df.groupby('cell_id').apply(
                remove_cells_with_extra_traces, include_groups=False,
                max_nchrom=max_nchrom_gt2trace).reset_index(level=0)
            ncells_removed = ncells_orig - len(df.cell_id.drop_duplicates())
        else:  # Remove any extra traces from cell with >2 traces
            raise NotImplementedError()
        if verbose:
            print(f"\tRemoved {ncells_removed}/{ncells_orig} CELLS where >{max_nchrom_gt2trace}"
                 f" chromosomes have extra (>2) traces", flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)

    # ==== For each chromosome, only keep traces originating from cells with the specified number of traces per cell
    if ntraces_per_cell is not None:
        ntrace_orig = len(df[['cell_id', 'chrom', 'trace_id']].drop_duplicates())
        df = df[df.ntraces_per_cell == ntraces_per_cell]
        ntrace_removed = ntrace_orig - len(df[['cell_id', 'chrom', 'trace_id']].drop_duplicates())
        if verbose:
            print((f"\tRemoved {ntrace_removed}/{ntrace_orig} TRACES, only keeping traces originating from"
                   f" cells with {ntraces_per_cell} trace/cell (for the given chromosome)"), flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)

    # ==== Remove loci where <[cutoff]% data are non-missing (pool data from both traces together for this)
    if min_ndetected_per_locus:
        ncopies_per_chrom = df[['chrom', 'cell_id', 'trace_id']].drop_duplicates().groupby('chrom').size()
        cov_per_locus = df.groupby(['chrom', 'chrom_order', 'chosen_loci']).size().reset_index(
            level=[1, 2]).rename({0: 'ratio_ncopies'}, axis=1)
        cov_per_locus['ratio_ncopies'] /= ncopies_per_chrom
        cov_per_locus['pass_cutoff'] = cov_per_locus.ratio_ncopies >= min_ndetected_per_locus
        cov_per_locus = cov_per_locus.reset_index().set_index(['chrom', 'chrom_order'])
        pass_cutoff = cov_per_locus[cov_per_locus.pass_cutoff].index
        df.set_index(['chrom', 'chrom_order'], inplace=True)
        df = df[df.index.isin(pass_cutoff)].reset_index()
        if verbose:
            print((f"\tRemoved {(~cov_per_locus.pass_cutoff).sum()}/{len(cov_per_locus)} LOCI (="
                   f"{((~cov_per_locus.pass_cutoff) & cov_per_locus.chosen_loci).sum()}/{cov_per_locus.chosen_loci.sum()}"
                   f" chosen loci) which were detected in <{min_ndetected_per_locus * 100:g}% of cells"), flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)
            print('\t   ' + cov_per_locus.loc[cov_per_locus.pass_cutoff, 'ratio_ncopies'].describe().to_string().replace(
                '\n', '\n\t   '), flush=True)

    # ==== If multiple loci have same approx midpoint location, consolidate, retaining those with the most even spacing
    if enforce_even_spacing and not df.chosen_loci.all():
        loci_orig = df[['chrom', 'chrom_start', 'chrom_end']].drop_duplicates()
        loci_keep = get_evenly_spaced_loci(
            loci_orig, spacing=spacing, cutoff_neighbor=spacing / 10,
            cutoff_median=spacing / 2, verbose=False).set_index(['chrom', 'start_bp', 'end_bp']).index
        df = df[df.set_index(['chrom', 'chrom_start', 'chrom_end']).index.isin(loci_keep)]
        if verbose:
            print((f"\tRemoved {len(loci_orig) - len(loci_keep)}/{len(loci_orig)} LOCI that had the same approx"
                   f" midpoint location as another locus (midpoints within {spacing}/10)"), flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)

    # ==== Remove traces where very few loci were detected
    # (note: if by_chosen_loci_only=True: only considering number of chosen loci, NOT number of total loci)
    if trace_min_nloci or trace_min_nloci_ratio:
        if by_chosen_loci_only:
            nloci_per_chrom = df[['chrom', 'chrom_order', 'chosen_loci']].drop_duplicates().groupby(
                'chrom').chosen_loci.sum()
        else:
            nloci_per_chrom = df[['chrom', 'chrom_order']].drop_duplicates().groupby('chrom').size()
        ntrace_orig = len(df[['cell_id', 'chrom', 'trace_id']].drop_duplicates())
        df = df.groupby(['cell_id', 'trace_id']).apply(
            remove_small_traces, include_groups=False, nloci_per_chrom=nloci_per_chrom,
            min_nloci=trace_min_nloci, min_nloci_ratio=trace_min_nloci_ratio,
            by_chosen_loci_only=by_chosen_loci_only).reset_index(level=[0, 1])
        ntrace_removed = ntrace_orig - len(df[['cell_id', 'chrom', 'trace_id']].drop_duplicates())
        if verbose:
            print((f"\tRemoved {ntrace_removed}/{ntrace_orig} TRACES whose length <= {trace_min_nloci} loci or <= "
                   f"{trace_min_nloci_ratio * 100:.3g}% mappable chrom"), flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)

    # ==== Remove chromosomes with where NO cells have >1 trace
    # (note: if by_chosen_loci_only=True: only considering number of chosen loci, NOT number of total loci)
    if ntraces_per_cell is None:
        nchrom_orig = len(df.chrom.drop_duplicates())
        mask = ~df.chrom.str.lower().str.replace(
            r'[^A-z0-9]', '', regex=True).isin(['x', 'y', 'chrx', 'chry'])
        if by_chosen_loci_only:
            df[mask] = df[mask].groupby('chrom').apply(
                lambda x: (None if len(x.loc[x.chosen_loci, 'trace_id'].drop_duplicates()) == 1 else x),
                include_groups=False).reset_index(level=0)
        else:
            df[mask] = df[mask].groupby('chrom').apply(
                lambda x: (None if len(x.trace_id.drop_duplicates()) == 1 else x),
                include_groups=False).reset_index(level=0)
        nchrom_removed = nchrom_orig - len(df.chrom.drop_duplicates())
        if verbose:
            print(f"\tRemoved {nchrom_removed}/{nchrom_orig} CHROMOSOMES where no cells have >1 trace", flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)

    return df


def get_nloci_in_trace(df):
    return pd.Series({'nloci': len(df), 'nloci_2trace': (df.ntraces == 2).sum()})


def get_distance_matrix_idx(df):
    i, j = np.triu_indices(len(df), 1)
    row = df[['chrom_order']].values[i]
    col = df[['chrom_order']].values[j]
    idx = list(map(tuple, np.stack([row.ravel(), col.ravel()], axis=1)))
    return idx


def get_distance_matrix(df, invert_distances=False):
    dis = pdist(df[['x', 'y', 'z']].values)
    if invert_distances:
        dis = np.power(dis, -1)
    idx = get_distance_matrix_idx(df)
    return pd.DataFrame.from_dict({'idx': idx, 'dis': dis})


def get_ntraces_per_cell(df):
    df['ntraces_per_cell2'] = df.groupby(['cell_id', 'chrom']).apply(
        lambda x: len(x.loc[x.chosen_loci, 'trace_id'].drop_duplicates()),
        include_groups=False).rename('ntraces_per_cell').reset_index().ntraces_per_cell
    
    df.set_index(['cell_id', 'chrom'], inplace=True)
    has_2traces_idx = df[df.trace_id == 1].index.intersection(df[df.trace_id == 2].index)
    df['ntraces_per_cell'] = 1
    df.loc[has_2traces_idx, 'ntraces_per_cell'] = 2
    df.reset_index(inplace=True)
    return df


def get_full_outdir(output_dir, ntraces_per_cell=None):
    subdir = []
    if ntraces_per_cell is not None:
        subdir.append(f'ntraces_chrom-cell_{ntraces_per_cell}')
    if len(subdir) != 0:
        output_dir = os.path.join(output_dir, '.'.join(subdir))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def process_data(input_file, chosen_loci_file, max_nchrom_gt2trace=None, min_ndetected_per_locus=0.1,
                 trace_min_nloci=None, trace_min_nloci_ratio=None, by_chosen_loci_only=True, ntraces_per_cell=None,
                 enforce_even_spacing=True, spacing=2.5, output_dir=None, verbose=True):
    if output_dir is not None:
        output_dir = get_full_outdir(output_dir, ntraces_per_cell=ntraces_per_cell)
        output_file = os.path.join(
            output_dir, re.sub(r'\.csv$', '', os.path.basename(input_file)) + '.filter.csv')
        print(output_file + '\n', flush=True)

    # ==== Load data
    if os.path.exists(input_file + '.gz') and not os.path.exists(input_file):
        input_file = input_file + '.gz'
    df = pd.read_csv(input_file, dtype={
        'cell_id': str, 'rep_id': float, 'chrom': str, 'trace_id': int, 'chrom_order': int,
        'chrom_start': int, 'chrom_end': int, 'x': float, 'y': float, 'z': float})
    df.columns = [x.strip('#') for x in df.columns]

    # ==== Label 'chosen' loci (eg those spaced 2.5Mb apart across the genome)
    loci = pd.read_csv(chosen_loci_file).set_index(['chrom', 'start_bp', 'end_bp'])
    df['chosen_loci'] = df.set_index(['chrom', 'chrom_start', 'chrom_end']).index.isin(loci.index)
    if verbose:
        print(f"'Chosen' loci make up {df.chosen_loci.sum() / len(df) * 100:.4g}% of the data\n", flush=True)

    # ==== Count number of traces (per CHROM, per cell)... *prior to filtering*
    df = get_ntraces_per_cell(df)

    # ==== Filter data
    df = filter_data(
        df, max_nchrom_gt2trace=max_nchrom_gt2trace, min_ndetected_per_locus=min_ndetected_per_locus,
        trace_min_nloci=trace_min_nloci, trace_min_nloci_ratio=trace_min_nloci_ratio,
        by_chosen_loci_only=by_chosen_loci_only, ntraces_per_cell=ntraces_per_cell,
        enforce_even_spacing=enforce_even_spacing, spacing=spacing, verbose=verbose)

    # ==== Optional: only keep 'chosen' loci (eg those spaced 2.5Mb apart across the genome)
    if enforce_even_spacing:
        df = df[df.chosen_loci].drop('chosen_loci', axis=1)

    # ==== Optionally save results
    if output_dir is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"\nSaving to {output_file}", flush=True)
        df.to_csv(output_file, index=False)

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--chosen_loci", required=True, type=str)
    parser.add_argument("--max_nchrom_gt2trace", type=int)
    parser.add_argument("--min_ndetected_per_locus", default=0.1, type=float)
    parser.add_argument("--trace_min_nloci", type=int)
    parser.add_argument("--trace_min_nloci_ratio", type=float)
    parser.add_argument('--filter-via-all-loci', default=True,
                        dest="by_chosen_loci_only", action='store_false')
    parser.add_argument("--ntraces_per_cell", type=int)
    parser.add_argument('--dont-enforce-even-spacing', default=True,
                        dest="enforce_even_spacing", action='store_false')
    parser.add_argument("--spacing", default=2.5, type=float)    
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.data)
    process_data(
        input_file=args.data, chosen_loci_file=args.chosen_loci, max_nchrom_gt2trace=args.max_nchrom_gt2trace,
        min_ndetected_per_locus=args.min_ndetected_per_locus, trace_min_nloci=args.trace_min_nloci,
        trace_min_nloci_ratio=args.trace_min_nloci_ratio, by_chosen_loci_only=args.by_chosen_loci_only,
        ntraces_per_cell=args.ntraces_per_cell, enforce_even_spacing=args.enforce_even_spacing,
        spacing=args.spacing, output_dir=output_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
