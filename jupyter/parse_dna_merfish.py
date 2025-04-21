#!/usr/bin/env python3

import os
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tqdm import tqdm


def remove_small_traces(df, nloci_per_chrom, min_nloci=3, min_nloci_ratio=0.1):
    df['cutoff'] = np.maximum(min_nloci, min_nloci_ratio * nloci_per_chrom.loc[df.chrom.values].values)
    # df = df.groupby('chrom').apply(lambda x: (x if len(x) >= x.cutoff.values[0] else None), include_groups=False).reset_index(level=0)
    df = df.groupby('chrom').apply(lambda x: (
        x if x.chosen_loci.sum() >= x.cutoff.values[0] else None), include_groups=False).reset_index(level=0)
    if len(df):
        return df.drop('cutoff', axis=1)


def remove_cells_with_extra_traces(df, max_nchrom=1):
    nchrom_with_extra_traces = df.groupby('chrom').apply(
        lambda x: len(x.trace_id.drop_duplicates()) > 2, include_groups=False).sum()
    if nchrom_with_extra_traces <= max_nchrom:
        return df


def filter_data_per_hmlg(df, min_nonmissing_per_phased_locus=0.1, verbose=True):
    if not min_nonmissing_per_phased_locus:
        return df
    # Remove loci where >[cutoff] cells are missing from one or more of the traces
    ncopies_per_chrom = df[['hmlg', 'chrom', 'cell_id']].drop_duplicates().groupby(
        ['hmlg', 'chrom']).size().unstack(level=0)
    cov_per_locus = df.groupby(['hmlg', 'chrom', 'chrom_order', 'chosen_loci']).size().unstack(
        level=0).reset_index(level=2)
    cov_per_locus[[1, 2]] /= ncopies_per_chrom
    cov_per_locus['hmlg_min'] = cov_per_locus[[1, 2]].min(axis=1)
    cov_per_locus['pass_cutoff'] = cov_per_locus.hmlg_min >= min_nonmissing_per_locus
    pass_cutoff = cov_per_locus[cov_per_locus.pass_cutoff].index
    df.set_index(['chrom', 'chrom_order'], inplace=True)
    df = df[df.index.isin(pass_cutoff)].reset_index()
    if verbose:
        print((f"\tRemoved {(~cov_per_locus.pass_cutoff).sum()}/{len(cov_per_locus)} LOCI (="
               f"{((~cov_per_locus.pass_cutoff) & cov_per_locus.chosen_loci).sum()}/{cov_per_locus.chosen_loci.sum()}"
               f" chosen loci) where one or both homologs were detected in <{min_nonmissing_per_locus * 100:g}% of cells"), flush=True)
        print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)
        print('\t   ' + cov_per_locus.loc[cov_per_locus.pass_cutoff, 'ratio_ncopies'].describe().to_string().replace(
            '\n', '\n\t   '), flush=True)
    return df


def filter_data(df, max_nchrom_gt2trace=0, min_nonmissing_per_locus=0.1, trace_min_nloci=3,
                trace_min_nloci_ratio=0.1, verbose=True):
    nrows_orig = len(df)
    if verbose:
        print(f"FILTERING... original n={len(df):,}", flush=True)

    if max_nchrom_gt2trace is None:
        max_nchrom_gt2trace = 0  # Strictest, no chrom have extra (>2) traces
    if min_nonmissing_per_locus is None:
        min_nonmissing_per_locus = 0  # Skip filtering on this
    if trace_min_nloci is None:
        trace_min_nloci = 0  # Skip filtering on this
    if trace_min_nloci_ratio is None:
        trace_min_nloci_ratio = 0  # Skip filtering on this

    # Remove cells where >[cutoff] chromosomes have >2 traces
    # If [cutoff] > 0, also remove any extra traces
    if len(df.trace_id.drop_duplicates()) > 0:
        if max_nchrom_gt2trace < len(df.chrom.drop_duplicates()):
            ncells_orig = len(df.cell_id.drop_duplicates())
            df = df.groupby('cell_id').apply(
                remove_cells_with_extra_traces, include_groups=False,
                max_nchrom=max_nchrom_gt2trace).reset_index(level=0)
            ncells_removed = ncells_orig - len(df.cell_id.drop_duplicates())
            if verbose:
                print(f"\tRemoved {ncells_removed}/{ncells_orig} CELLS where >{max_nchrom_gt2trace}"
                     f" chromosomes have extra (>2) traces", flush=True)
                print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)
        if max_nchrom_gt2trace > 0: # Remove any extra traces from cell with >2 traces
            raise NotImplementedError()

    # Remove loci where >[cutoff] cells are missing (pool data from both traces together for this)
    if min_nonmissing_per_locus:
        ncopies_per_chrom = df[['chrom', 'cell_id', 'trace_id']].drop_duplicates().groupby('chrom').size()
        cov_per_locus = df.groupby(['chrom', 'chrom_order', 'chosen_loci']).size().reset_index(
            level=[1, 2]).rename({0: 'ratio_ncopies'}, axis=1)
        cov_per_locus['ratio_ncopies'] /= ncopies_per_chrom
        cov_per_locus['pass_cutoff'] = cov_per_locus.ratio_ncopies >= min_nonmissing_per_locus
        cov_per_locus = cov_per_locus.reset_index().set_index(['chrom', 'chrom_order'])
        pass_cutoff = cov_per_locus[cov_per_locus.pass_cutoff].index
        df.set_index(['chrom', 'chrom_order'], inplace=True)
        df = df[df.index.isin(pass_cutoff)].reset_index()
        if verbose:
            print((f"\tRemoved {(~cov_per_locus.pass_cutoff).sum()}/{len(cov_per_locus)} LOCI (="
                   f"{((~cov_per_locus.pass_cutoff) & cov_per_locus.chosen_loci).sum()}/{cov_per_locus.chosen_loci.sum()}"
                   f" chosen loci) which were detected in <{min_nonmissing_per_locus * 100:g}% of cells"), flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)
            print('\t   ' + cov_per_locus.loc[cov_per_locus.pass_cutoff, 'ratio_ncopies'].describe().to_string().replace(
                '\n', '\n\t   '), flush=True)

    # Remove very small traces
    # (NOTE: trace size is determined by number of chosen loci, NOT number of total loci)
    if trace_min_nloci or trace_min_nloci_ratio:
        # nloci_per_chrom = df[['chrom', 'chrom_order']].drop_duplicates().groupby('chrom').size()
        nloci_per_chrom = df[['chrom', 'chrom_order', 'chosen_loci']].drop_duplicates().groupby('chrom').chosen_loci.sum()
        ntrace_orig = len(df[['cell_id', 'chrom', 'trace_id']].drop_duplicates())
        df = df.groupby(['cell_id', 'trace_id']).apply(
            remove_small_traces, include_groups=False, nloci_per_chrom=nloci_per_chrom,
            min_nloci=trace_min_nloci, min_nloci_ratio=trace_min_nloci_ratio).reset_index(level=[0, 1])
        ntrace_removed = ntrace_orig - len(df[['cell_id', 'chrom', 'trace_id']].drop_duplicates())
        if verbose:
            print((f"\tRemoved {ntrace_removed}/{ntrace_orig} TRACES whose length <= {trace_min_nloci} loci or <= "
                   f"{trace_min_nloci_ratio * 100:.3g}% mappable chrom"), flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)

    # Remove chromosomes with where no cells have >1 trace
    # (NOTE: only considering chosen loci, not all loci)
    # df = df.groupby('chrom').apply(
    #     lambda x: (None if len(x.trace_id.drop_duplicates()) == 1 else x),
    #     include_groups=False).reset_index(level=0)
    nchrom_orig = len(df.chrom.drop_duplicates())
    df = df.groupby('chrom').apply(
        lambda x: (None if len(x.loc[x.chosen_loci, 'trace_id'].drop_duplicates()) == 1 else x),
        include_groups=False).reset_index(level=0)
    nchrom_removed = nchrom_orig - len(df.chrom.drop_duplicates())
    if verbose:
        print(f"\tRemoved {nchrom_removed}/{nchrom_orig} CHROMOSOMES where no cells have >1 trace", flush=True)
        print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)

    return df


def get_nloci_in_trace(df):
    return pd.Series({'nloci': len(df), 'nloci_2trace': (df.ntraces == 2).sum()})


def summarize_labeling(df, prefix=''):
    ncells_labeled = df.loc[~df.hmlg.isnull(), ['cell_id', 'trace_id', 'hmlg']].drop_duplicates().groupby(
        ['trace_id', 'hmlg']).size().reset_index(level=[0, 1]).rename({0: 'ncells'}, axis=1)
    ncells_labeled['pairing'] = 'A'
    ncells_labeled.loc[ncells_labeled.trace_id != ncells_labeled.hmlg, 'pairing'] = 'B'
    ncells_labeled = ncells_labeled.set_index('pairing').sort_index()

    trace1 = ncells_labeled.loc[ncells_labeled.trace_id == 1, 'ncells']
    trace2 = ncells_labeled.loc[ncells_labeled.trace_id == 2, 'ncells']
    cells_with_only_1trace = trace1 - trace2

    desc = [
        "Pairings for cells with only 1 trace:",
        pd.DataFrame(cells_with_only_1trace).reset_index().to_string(index=False),
        "\nPairings for cells with 2 traces:",
        pd.DataFrame(trace2).reset_index().to_string(index=False)]

    print(prefix + f'\n{prefix}'.join(desc), flush=True)
    
    return ncells_labeled


def disterror(struct_hmlg, struct_candidate):
    # While struct_hmlg only includes loci present in struct_candidate...
    # ...we need to make sure that struct_candidate only includes loci present in struct_hmlg
    # for this particular trace of this particular previously labeled cell
    struct_candidate = struct_candidate[struct_candidate.chrom_order.isin(
        struct_hmlg.chrom_order)]
    dis_candidate = pdist(struct_candidate[['x', 'y', 'z']].values)
    dis_hmlg = pdist(struct_hmlg[['x', 'y', 'z']].values)
    # Take the sum of the squared error and record the number of bins ('n') so that
    # we can more easily aggregate data across previously labeled cells
    # (Don't want to take the mean of multiple MSE values since 'n' might differ per cell)
    return pd.Series({'err': np.square(dis_candidate - dis_hmlg).sum(), 'n': dis_hmlg.size})


def label_homologs_for_chrom(df, df_cell_desc, chrom, compare_to_labeled_loci_with_2_traces=False, verbose=True):
    if verbose:
        print(f"\nASSIGNING HOMOLOGS FOR: chr{chrom}", flush=True)
    
    cells = df_cell_desc.loc[df_cell_desc.chrom == chrom, ['cell_id', 'ntraces_in_cell']].set_index(
        'cell_id').ntraces_in_cell
    
    # Reset homolog identities for this chromosome
    if 'hmlg' in df.columns:
        df.loc[df.chrom == chrom, 'hmlg'] = None
    else:
        df['hmlg'] = None
    
    # Arbitrarily label homologs of the first cell
    first_cell = cells.index[0]
    assert cells.loc[first_cell] == 2  # The first cell has 2 traces for the give chrom
    mask = (df.chrom == chrom) & (df.cell_id == first_cell)
    df.loc[mask, 'hmlg'] = df.loc[mask, 'trace_id']
    
    # cells_dict_items = cells.to_dict().items()
    for cell_id, ntraces in tqdm(cells.to_dict().items()):
        if cell_id == first_cell:  # First cell has already been labeled
            continue

        # Re-select data for chrom (because 'hmlg' label in 'df' got updated for prev cell)
        df_chrom = df[(df.chrom == chrom)]
    
        # Get all traces for the candidate cell, only including loci present in all of the candidate cell's traces
        candidate = df_chrom.loc[(df_chrom.cell_id == cell_id) & (df_chrom.ntraces == ntraces)]
        candidate = [candidate.loc[
                     candidate.trace_id == trace_id,
                     ['chrom_order', 'x', 'y', 'z']] for trace_id in np.arange(1, ntraces + 1)]
        loci_candidate = candidate[0].chrom_order.values
    
        # Get all labeled traces from previous cells, only including loci present in all of the candidate cell's traces
        if compare_to_labeled_loci_with_2_traces:
            hmlgs = df_chrom.loc[(~df_chrom.hmlg.isnull()) & (df_chrom.ntraces == 2) & df_chrom.chrom_order.isin(
                loci_candidate), ['cell_id', 'hmlg', 'chrom_order', 'x', 'y', 'z']]
        else:
            hmlgs = df_chrom.loc[(~df_chrom.hmlg.isnull()) & df_chrom.chrom_order.isin(
                loci_candidate), ['cell_id', 'hmlg', 'chrom_order', 'x', 'y', 'z']]
    
        # Calculate distance error between each candidate trace and the labeled traces from each previous cell
        res = [hmlgs.groupby(['cell_id', 'hmlg']).apply(
            disterror, include_groups=False, struct_candidate=x).groupby(level=1).sum() for x in candidate]
        res = [x.err / x.n for x in res]  # Get MSE of distance matrices per pairing of traces 
    
        # print('***')
        # for i in range(ntraces):
        #     print(i); print(res[i])
    
        # Indices that determine how the candidate cell's trace(s) get paired to the pre-labled hmlgs
        pairingA = np.arange(ntraces) + 1  # Pair trace1 to hmlg1 (and trace2 to hmlg2)
        pairingB = 3 - pairingA  # Pair trace1 to hmlg2 (and trace2 to hmlg1)
        # print(pairingA); print(pairingB)
    
        # Compare mean distance error for each pairing
        # (Can take the mean because the number of loci is the same for all traces in the candidate cell)
        resA = np.mean([res[i].loc[pairingA[i]] for i in range(ntraces)])
        resB = np.mean([res[i].loc[pairingB[i]] for i in range(ntraces)])
    
        # print(resA); print(resB)
    
        # pairing = [pairingA, pairingB][np.argmin([resA, resB])]
        # print(pairing)
    
        # Assign the homolog itentities to candidate cell based on min distance error of the pairings
        mask = (df.chrom == chrom) & (df.cell_id == cell_id)
        if resA <= resB:
            # print('A')
            df.loc[mask, 'hmlg'] = df.loc[mask, 'trace_id']
        else:
            # print('B')
            df.loc[mask, 'hmlg'] = 3 - df.loc[mask, 'trace_id']

    # Assess labeling results
    if verbose:
        ncells_labeled = summarize_labeling(df, prefix='\t')

    return df


def process_data(input_file, chosen_loci_file, max_nchrom_gt2trace=0, min_nonmissing_per_locus=0.1, trace_min_nloci=3,
                 trace_min_nloci_ratio=0.1, chrom_to_filter=None, compare_to_labeled_loci_with_2_traces=False,
                 output_file=None, verbose=True):
    df = pd.read_csv(input_file, dtype={
        'cell_id': str, 'rep_id': float, 'chrom': str, 'trace_id': int, 'chrom_order': int,
        'chrom_start': int, 'chrom_end': int, 'x': float, 'y': float, 'z': float})
    df.columns = [x.strip('#') for x in df.columns]
    if 'hmlg' in df.columns:  # TODO temp fix
        df.rename({'hmlg': 'trace_id'}, axis=1, inplace=True)

    # ==== Label 'chosen' loci (eg those spaced 2.5Mb apart across the genome)
    loci = pd.read_csv(chosen_loci_file).set_index(['chrom', 'start_bp', 'end_bp'])
    df['chosen_loci'] = df.set_index(['chrom', 'chrom_start', 'chrom_end']).index.isin(loci.index)
    if verbose:
        print(f"'Chosen' loci make up {df.chosen_loci.sum() / len(df) * 100:.4g}% of the data\n", flush=True)

    # ==== Filter data
    df = filter_data(
        df, max_nchrom_gt2trace=max_nchrom_gt2trace, min_nonmissing_per_locus=min_nonmissing_per_locus,
        trace_min_nloci=trace_min_nloci, trace_min_nloci_ratio=trace_min_nloci_ratio, verbose=verbose)

    # ==== Count number of traces (per locus, per cell)
    df.set_index(['cell_id', 'chrom', 'chrom_order'], inplace=True)
    has_2traces_idx = df[df.trace_id == 1].index.intersection(df[df.trace_id == 2].index)
    df['ntraces'] = 1
    df.loc[has_2traces_idx, 'ntraces'] = 2
    df.reset_index(inplace=True)

    # ==== Per chromosome: get ordering of cells in which homologs will be labled
    # For each trace: nloci per trace + nloci that are on both traces in the given cell
    df_cell_desc = df.groupby(['cell_id', 'chrom', 'trace_id']).apply(
        get_nloci_in_trace, include_groups=False).reset_index(level=[2])
    
    # For each cell/chrom: total nloci (summed across both traces) + difference in nloci between homolog
    df_cell_desc['nloci_per_cell'] = df_cell_desc['nloci'].groupby(level=[0,1]).sum()
    df_cell_desc['nloci_diff'] = df_cell_desc['nloci'].groupby(level=[0,1]).apply(
        lambda x: (-np.inf if len(x) == 1 else -np.abs(x.values[0] - x.values[1])), include_groups=False)
    df_cell_desc.reset_index(level=[0,1], inplace=True)

    # Sort such that cells with the most data are on top (for each chromosome)
    cols_to_sort = ['nloci_2trace', 'nloci_per_cell', 'nloci_diff']
    df_cell_desc = df_cell_desc.groupby(['cell_id', 'chrom'] + cols_to_sort).size().reset_index(
        level=[0, 1, 2, 3, 4]).rename({0: 'ntraces_in_cell'}, axis=1)
    df_cell_desc.sort_values(['chrom', 'ntraces_in_cell'] + cols_to_sort, ascending=False, inplace=True)

    # ==== Assign homolog identities
    # Choose which chromosomes to assign homolog identities for
    chromosomes = df_cell_desc.chrom.drop_duplicates().sort_values().values
    if chrom_to_filter is None:
        chrom_to_filter = chromosomes
    else:
        if isinstance(chrom_to_filter, (int, str)):
            chrom_to_filter = [chrom_to_filter]
        chrom_to_filter = np.array([str(x).replace('chr', '') for x in chrom_to_filter])
        if not isinstance(chromosomes[0], str):
            chrom_to_filter = chrom_to_filter.astype(chromosomes.dtype)
        chrom_to_filter = np.array([x for x in chrom_to_filter if x in chromosomes])
        if len(chromosomes) != len(chrom_to_filter):
            df = df[df.chrom.isin(chrom_to_filter)]

    # Assign homolog identities for each cell
    # NOTE: considering distances between ALL loci for this process, not just 'chosen' loci
    if verbose:
        print(f"\nASSIGNING HOMOLOGS FOR: {', '.join(['chr' + str(x) for x in chrom_to_filter])}", flush=True)
        print("\tWhen comparing candidate cell's traces to homologs of previously labeled cells...", flush=True)
        if compare_to_labeled_loci_with_2_traces:
            print("\t...require each previously labeled cell's loci to be present in BOTH homologs!", flush=True)
        else:
            print("\t...compare to all loci in each previously labeled cell (even if it's only present in 1 homolog)",
                  flush=True)
    df['hmlg'] = None
    df.sort_values(['chrom', 'cell_id', 'chrom_order'], inplace=True)
    for chrom in chrom_to_filter:
        df = label_homologs_for_chrom(
            df, df_cell_desc=df_cell_desc, chrom=chrom,
            compare_to_labeled_loci_with_2_traces=compare_to_labeled_loci_with_2_traces,
            verbose=verbose)

    # ==== Only keep 'chosen' loci (eg those spaced 2.5Mb apart across the genome)
    df = df[df.chosen_loci].drop('chosen_loci', axis=1)

    # ==== Optionally save results
    if output_file is not None:
        if len(chromosomes) != len(chrom_to_filter):
            output_file = os.path.join(
                os.path.dirname(output_file), 'per_chrom',
                re.sub(r'\.csv$', '', os.path.basename(output_file)))
            chrom_label = '_'.join(map(str, chrom_to_filter.tolist()))
            output_file = f"{output_file}.chrom_{chrom_label}.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"\nSaving to {output_file}", flush=True)
        df.to_csv(output_file, index=False)

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--chosen_loci", required=True, type=str)
    parser.add_argument("--max_nchrom_gt2trace", default=0, type=int)
    parser.add_argument("--min_nonmissing_per_locus", default=0.1, type=float)
    parser.add_argument("--trace_min_nloci", default=3, type=int)
    parser.add_argument("--trace_min_nloci_ratio", default=0.1, type=float)
    parser.add_argument('--compare_to_labeled_loci_with_2_traces', default=False,
                        action='store_true')
    parser.add_argument("--chrom", type=str, nargs='+')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    output_file = re.sub(r'\.csv$', '', args.data) + '.filtered.csv'
    process_data(
        input_file=args.data, chosen_loci_file=args.chosen_loci, max_nchrom_gt2trace=args.max_nchrom_gt2trace,
        min_nonmissing_per_locus=args.min_nonmissing_per_locus, trace_min_nloci=args.trace_min_nloci,
        trace_min_nloci_ratio=args.trace_min_nloci_ratio,
        compare_to_labeled_loci_with_2_traces=args.compare_to_labeled_loci_with_2_traces,
        chrom_to_filter=args.chrom, output_file=output_file, verbose=args.verbose)


if __name__ == "__main__":
    main()
