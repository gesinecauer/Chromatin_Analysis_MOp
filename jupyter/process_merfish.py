#!/usr/bin/env python3

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from process_loci import get_evenly_spaced_loci
from process_snm3c import annotate_loci_snm3c


def get_nloci_in_trace(df):
    return pd.Series({'nloci': len(df), 'nloci_2trace': (df.ntraces == 2).sum()})


def get_ntraces_per_cell(df):
    df.set_index(['cell_id', 'chrom'], inplace=True)
    df['ntraces_per_cell'] = df.groupby(level=[0, 1]).apply(
        lambda x: len(x.trace_id.drop_duplicates()),
        include_groups=False)
    df.reset_index(inplace=True)
    return df


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


def ratio_detected_per_locus(df, min_detected_per_locus=None, MAD_min_detected_per_locus=None,
                             per_chrom=False, nrows_orig=None, verbose=True):
    """Remove LOCI where <[cutoff]% data are non-missing (pool data from both traces together)"""
    
    # Setup
    if nrows_orig is None:
        nrows_orig = len(df)
    if min_detected_per_locus is None:
        min_detected_per_locus = 0
    if MAD_min_detected_per_locus is None:
        MAD_min_detected_per_locus = 0

    if per_chrom:  # Number of molecules (across cells & traces) - PER CHROM
        denom = df[['chrom', 'cell_id', 'trace_id']].drop_duplicates().groupby('chrom').size()
        desc = "of molecules (for the given chromosome)"
    else:  # Number of molecules (across cells & traces) - in entire DATASET
        denom = len(df[['cell_id', 'trace_id']].drop_duplicates())
        desc = "of molecules (across chromosomes)"

    grouping_cols = ['chrom', 'chrom_order']
    if 'chosen_loci' in df.columns:
        grouping_cols.append('chosen_loci')
    cov_per_locus = df.groupby(grouping_cols).size().rename("ratio_detected").to_frame()
    if 'chosen_loci' in df.columns:
        cov_per_locus.reset_index(level=2, inplace=True)
    cov_per_locus["ratio_detected"] /= denom

    cov_per_locus['pass_cutoff'] = True
    if min_detected_per_locus or MAD_min_detected_per_locus:
        cutoff = min_detected_per_locus
        if MAD_min_detected_per_locus:
            mad = median_abs_deviation(cov_per_locus.ratio_detected.values)
            mad_cutoff = cov_per_locus.ratio_detected.median() - MAD_min_detected_per_locus * mad
            cutoff = max(cutoff, mad_cutoff)
        
        cov_per_locus['pass_cutoff'] = cov_per_locus.ratio_detected >= cutoff
        pass_cutoff = cov_per_locus[cov_per_locus.pass_cutoff].index
        df.set_index(['chrom', 'chrom_order'], inplace=True)
        df = df[df.index.isin(pass_cutoff)].reset_index()
        if verbose:
            if 'chosen_loci' in df.columns:
                tmp = (f" (={((~cov_per_locus.pass_cutoff) & cov_per_locus.chosen_loci).sum()}"
                       f"/{cov_per_locus.chosen_loci.sum()} chosen loci)")
            else:
                tmp = ""
            print((f"Removed {(~cov_per_locus.pass_cutoff).sum()}/{len(cov_per_locus)} LOCI{tmp}"
                   f" that were detected in <{cutoff * 100:g}% of {desc}"), flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)
            print('\t   ' + cov_per_locus[cov_per_locus.pass_cutoff].describe().drop('count').to_string().replace(
                '\n', '\n\t   '), flush=True)
    elif verbose:
        print(f"\tRatio {desc} in which each locus was detected:", flush=True)
        print('\t   ' + cov_per_locus.describe().drop('count').to_string().replace('\n', '\n\t   '), flush=True)
    
    return df, cov_per_locus


def filter_data(df, max_nchrom_gt2trace=None, min_detected_per_locus=0.1,
                MAD_min_detected_per_locus=5, trace_min_nloci=None,
                trace_min_nloci_ratio=None, by_chosen_loci_only=True, ntraces_per_cell=None, 
                verbose=True):
    nrows_orig = len(df)
    if verbose:
        print(f"FILTERING... original n={len(df):,}", flush=True)

    # ==== Setup
    if min_detected_per_locus is None:
        min_detected_per_locus = 0
    if MAD_min_detected_per_locus is None:
        MAD_min_detected_per_locus = 0
    if trace_min_nloci is None:
        trace_min_nloci = 0
    if trace_min_nloci_ratio is None:
        trace_min_nloci_ratio = 0

    # ==== Count number of traces (per CHROM, per cell)... *prior to filtering*
    if 'ntraces_per_cell' not in df.columns:
        df = get_ntraces_per_cell(df)

    # ==== Remove CELLS where >[cutoff] chromosomes have >2 traces
    if (max_nchrom_gt2trace is not None) and (max_nchrom_gt2trace < len(
            df.chrom.drop_duplicates())) and (len(df.trace_id.drop_duplicates()) > 0):
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

    # ==== For each chromosome, only keep traces originating from CELLS with the specified number of traces per cell
    if ntraces_per_cell is not None:
        ntrace_orig = len(df[['cell_id', 'chrom', 'trace_id']].drop_duplicates())
        df = df[df.ntraces_per_cell == ntraces_per_cell]
        ntrace_removed = ntrace_orig - len(df[['cell_id', 'chrom', 'trace_id']].drop_duplicates())
        if verbose:
            print((f"\tRemoved {ntrace_removed}/{ntrace_orig} TRACES, only keeping traces originating from"
                   f" cells with {ntraces_per_cell} trace/cell (for the given chromosome)"), flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)

    # ==== Remove LOCI where <[cutoff]% data are non-missing - PER CHROM
    if min_detected_per_locus or MAD_min_detected_per_locus:
        df, _ = ratio_detected_per_locus(
            df, min_detected_per_locus=min_detected_per_locus, MAD_min_detected_per_locus=MAD_min_detected_per_locus,
            per_chrom=True, nrows_orig=nrows_orig, verbose=verbose)

    # ==== Remove TRACES where very few loci were detected
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

    # ==== Remove CHROMOSOMES with where NO cells have >1 trace (filter autosomes only)
    # (note: if by_chosen_loci_only=True: only considering number of chosen loci, NOT number of total loci)
    if ntraces_per_cell is None:
        mask = ~(df.chrom.astype(str).str.lower().str.replace('chr', '', regex=False).str.replace(
            r'[^A-z0-9]', '', regex=True).isin(['x', 'y']))
        nchrom_orig = len(df.loc[mask, 'chrom'].drop_duplicates())
        if by_chosen_loci_only:
            chrom_to_remove = df[mask].groupby('chrom').apply(
                lambda x: (x if (len(x.loc[x.chosen_loci, 'trace_id'].drop_duplicates()) == 1) else None),
                include_groups=False)
        else:
            chrom_to_remove = df[mask].groupby('chrom').apply(
                lambda x: (x if (len(x.trace_id.drop_duplicates()) == 1) else None),
                include_groups=False)
        if chrom_to_remove.size:
            df = df[~df.chrom.isin(chrom_to_remove)]
        if verbose:
            print(f"\tRemoved {chrom_to_remove.size}/{nchrom_orig} AUTOSOMES where no cells have >1 trace", flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)

    return df


def annotate_loci_merfish(df):
    # Create chromosome lengths for bed file, annotate with data coverage
    lengths_df = df[['merfish_id', 'chrom', 'chrom_start', 'chrom_end', 'chrom_order']].drop_duplicates().rename(
        {'chrom_start': 'start', 'chrom_end': 'end'}, axis=1).sort_values(['chrom', 'chrom_order'])
    lengths_df['mid'] = lengths_df[['start', 'end']].mean(axis=1)

    # Annotate chromosome lengths with data coverage information
    _, cov_per_locus = ratio_detected_per_locus(
        df, min_detected_per_locus=0, MAD_min_detected_per_locus=0, per_chrom=True, verbose=False)
    print(len(cov_per_locus), len(lengths_df), len(lengths_df[['chrom', 'chrom_order']].drop_duplicates()))  # TODO remove
    assert len(cov_per_locus) == len(lengths_df)
    lengths_df = lengths_df.merge(
        cov_per_locus.reset_index(), on=['chrom', 'chrom_order'], how='outer').rename({
        'pass_cutoff': 'has_merfish', 'ratio_detected': 'merfish_ratio'}, axis=1).drop(
        'chrom_order', axis=1)
    assert lengths_df.isnull().values.sum() == 0

    lengths_df['chrom'] = 'chr' + lengths_df.chrom.astype(str)  # After adding data coverage info
    return lengths_df


def annotate_loci(df, mcool_file, snm3c_resolution=0.1, normalize_snm3c=True, outdir=None, verbose=True):
    merfish_df = annotate_loci_merfish(df).drop(['start', 'end'], axis=1).rename({
        'mid': 'merfish_mid'}, axis=1)
    snm3c_df = annotate_loci_snm3c(
        mcool_file, resolution=snm3c_resolution, normalize=normalize_snm3c, verbose=verbose)

    # Merge locus info
    factor = int(snm3c_resolution * 1e6 / 2)
    merfish_df['mid'] = ((merfish_df.merfish_mid / factor).round() * factor).astype(int)
    lengths_df = snm3c_df.merge(merfish_df, on=['chrom', 'mid'], how='outer')    
    assert lengths_df.idx_genome.isnull().sum() == 0
    lengths_df.index = lengths_df.idx_genome
    lengths_df.index.name = None
    lengths_df = lengths_df[[
        'chrom', 'start', 'end', 'mid', 'idx_genome', 'idx_chrom', 'has_snm3c',
        'has_merfish', 'merfish_ratio', 'merfish_id', 'merfish_mid', 'snm3c_bias']]
    assert len(lengths_df.columns) == len(merfish_df.columns) + len(snm3c_df.columns) - 2

    if verbose:
        print(f"{(lengths_df.has_merfish & (~lengths_df.has_snm3c)).sum()} genomic loci"
              " are present in MERFISH but masked out from snm3c-seq", flush=True)    

    # Save bed file of chromosome lengths
    if outdir is not None:
        lengths_df_cp = lengths_df.copy()
        lengths_df_cp.columns = [f"#{c}" for c in lengths_df_cp.columns]
        lengths_df_cp.to_csv(os.path.join(outdir, "counts.bed"), index=False, header=True, sep="\t")

    return lengths_df


def process_data(input_file, chosen_loci_file, max_nchrom_gt2trace=None, min_detected_per_locus=0.1,
                 MAD_min_detected_per_locus=5, trace_min_nloci=None, trace_min_nloci_ratio=None,
                 by_chosen_loci_only=True, ntraces_per_cell=None, enforce_even_spacing=True,
                 mark_for_interhmlg=True, spacing=2.5, mcool_file=None, snm3c_resolution=0.1,
                 normalize_snm3c=True, output_dir=None, verbose=True):

    # ==== Setup
    if trace_min_nloci is None:
        trace_min_nloci = 0
    if trace_min_nloci_ratio is None:
        trace_min_nloci_ratio = 0
    if mark_for_interhmlg and ntraces_per_cell is not None and ntraces_per_cell != 2:
        raise ValueError(
            "When preparing inter-homolog data (mark_for_interhmlg=True), must"
            " filter for 2 traces per cell (ntraces_per_cell=2)")
    if output_dir is not None:
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

    # ==== Filter data
    nrows_orig = len(df)
    df = filter_data(
        df, max_nchrom_gt2trace=max_nchrom_gt2trace, min_detected_per_locus=min_detected_per_locus,
        MAD_min_detected_per_locus=MAD_min_detected_per_locus, trace_min_nloci=trace_min_nloci,
        trace_min_nloci_ratio=trace_min_nloci_ratio, by_chosen_loci_only=by_chosen_loci_only,
        ntraces_per_cell=ntraces_per_cell, verbose=verbose)

    # ==== Optional: re-filter data to note which rows are suitable for inter-homolog analyses
    if mark_for_interhmlg:
        if verbose:
            print("\nMark data to use for homolog-related analyses", flush=True)
        df_interhmlg = filter_data(
            df, max_nchrom_gt2trace=None, min_detected_per_locus=None,
            MAD_min_detected_per_locus=None, trace_min_nloci=max(trace_min_nloci, 3),
            trace_min_nloci_ratio=max(trace_min_nloci_ratio, 0.1),
            by_chosen_loci_only=by_chosen_loci_only, ntraces_per_cell=2, verbose=verbose)
        df.set_index(['cell_id', 'chrom', 'chrom_order', 'trace_id', 'rep_id'], inplace=True)
        df_interhmlg.set_index(['cell_id', 'chrom', 'chrom_order', 'trace_id', 'rep_id'], inplace=True)
        df['for_interhmlg'] = df.index.isin(df_interhmlg.index)
        df.reset_index(inplace=True)

    # ==== Optional: only keep 'chosen' loci (eg those spaced 2.5Mb apart across the genome)
    if enforce_even_spacing:
        df = df[df.chosen_loci].drop('chosen_loci', axis=1)

    # ==== Annotate each locus with 'merfish_id': chrom, chrom_order
    df['merfish_id'] = list(map(tuple, np.stack(
        [df['chrom'], df['chrom_order']], axis=1).tolist()))

    # ==== Optional: filter for loci present in snm3c-seq dataset
    if mcool_file is not None:
        if verbose:
            print("\nFilter for loci found in snm3c-seq data", flush=True)
        lengths_df = annotate_loci(
            df, mcool_file=mcool_file, snm3c_resolution=snm3c_resolution,
            normalize_snm3c=normalize_snm3c, outdir=output_dir, verbose=verbose)
        included_loci = lengths_df.loc[lengths_df.has_snm3c, 'merfish_id']
        df = df[df.merfish_id.isin(included_loci)]

    # ==== Optional: save results
    if output_dir is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"\nSaving to {output_file}", flush=True)
        df.to_csv(output_file, index=False)

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    if parser.parse_args().data == 'load':
        return
    parser.add_argument("--chosen_loci", required=True, type=str)
    parser.add_argument("--max_nchrom_gt2trace", type=int)
    parser.add_argument("--min_detected_per_locus", default=0.1, type=float)
    parser.add_argument("--MAD_min_detected_per_locus", default=5, type=float)
    parser.add_argument("--trace_min_nloci", type=int)
    parser.add_argument("--trace_min_nloci_ratio", type=float)
    parser.add_argument('--filter-via-all-loci', default=True,
                        dest="by_chosen_loci_only", action='store_false')
    parser.add_argument("--ntraces_per_cell", type=int)
    parser.add_argument('--dont-enforce-even-spacing', default=True,
                        dest="enforce_even_spacing", action='store_false')
    parser.add_argument('--dont-mark-for-interhmlg', default=True,
                        dest="mark_for_interhmlg", action='store_false')
    parser.add_argument("--spacing", default=2.5, type=float)

    parser.add_argument("--mcool_file", type=str)
    parser.add_argument("--snm3c_resolution", default=0.1, type=float)
    parser.add_argument('--dont-normalize-snm3c', default=True,
                        dest="normalize_snm3c", action='store_false')
    
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.data)
    process_data(
        input_file=args.data, chosen_loci_file=args.chosen_loci,
        max_nchrom_gt2trace=args.max_nchrom_gt2trace, min_detected_per_locus=args.min_detected_per_locus,
        MAD_min_detected_per_locus=args.MAD_min_detected_per_locus, trace_min_nloci=args.trace_min_nloci,
        trace_min_nloci_ratio=args.trace_min_nloci_ratio, by_chosen_loci_only=args.by_chosen_loci_only,
        ntraces_per_cell=args.ntraces_per_cell, enforce_even_spacing=args.enforce_even_spacing,
        mark_for_interhmlg=args.mark_for_interhmlg, spacing=args.spacing,
        mcool_file=args.mcool_file, snm3c_resolution=args.snm3c_resolution,
        normalize_snm3c=args.normalize_snm3c, output_dir=output_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
