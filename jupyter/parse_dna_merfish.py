#!/usr/bin/env python3

import os
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tqdm import tqdm
from process_loci import get_evenly_spaced_loci
from interpolate import interp_all


def remove_small_traces(df, nloci_per_chrom, min_nloci=3, min_nloci_ratio=0.1, by_chosen_loci_only=True):
    df['cutoff'] = np.maximum(min_nloci, min_nloci_ratio * nloci_per_chrom.loc[df.chrom.values].values)
    # df = df.groupby('chrom').apply(lambda x: (x if len(x) >= x.cutoff.values[0] else None), include_groups=False).reset_index(level=0)
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


def filter_data_per_hmlg(df, min_nonmissing_per_phased_locus=None, verbose=True):
    nrows_orig = len(df)
    # Remove loci where <[cutoff]% of cells are non-missing from one or both of the traces
    ncells = len(df[['cell_id']].drop_duplicates())
    # ncopies_per_chrom = df[['hmlg', 'chrom', 'cell_id']].drop_duplicates().groupby(
    #     ['hmlg', 'chrom']).size().unstack(level=0)
    grouping_cols = ['hmlg', 'chrom', 'chrom_order']
    if 'chosen_loci' in df.columns:
        grouping_cols.append('chosen_loci')
    cov_per_locus = df.groupby(grouping_cols).size().unstack(level=0)
    if 'chosen_loci' in df.columns:
        grouping_cols.reset_index(level=2, inplace=True)
    cov_per_locus[[1, 2]] /= ncells
    cov_per_locus['cell_cov_avg'] = cov_per_locus[[1, 2]].mean(axis=1)
    cov_per_locus['cell_cov_min'] = cov_per_locus[[1, 2]].min(axis=1)
    cov_per_locus.rename({1: "cell_cov_h1", 2: "cell_cov_h2"}, axis=1, inplace=True)
    if min_nonmissing_per_phased_locus is not None:
        cov_per_locus['pass_cutoff'] = cov_per_locus.cell_cov_min >= min_nonmissing_per_phased_locus
        pass_cutoff = cov_per_locus[cov_per_locus.pass_cutoff].index
        df.set_index(['chrom', 'chrom_order'], inplace=True)
        df = df[df.index.isin(pass_cutoff)].reset_index()
        if verbose:
            if 'chosen_loci' in df.columns:
                tmp = f" (={((~cov_per_locus.pass_cutoff) & cov_per_locus.chosen_loci).sum()}/{cov_per_locus.chosen_loci.sum()} chosen loci)"
            else:
                tmp = ""
            print((f"Removed {(~cov_per_locus.pass_cutoff).sum()}/{len(cov_per_locus)} LOCI{tmp}"
                   f" where one or both homologs were detected in <{min_nonmissing_per_phased_locus * 100:g}% of cells"), flush=True)
            print(f" ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)
            print('   ' + cov_per_locus[cov_per_locus.pass_cutoff].describe().drop('count').to_string().replace(
                '\n', '\n   '), flush=True)
    else:
        print("\tRatio of cells in which each locus was detected:", flush=True)
        print('\t   ' + cov_per_locus.describe().drop('count').to_string().replace('\n', '\n\t   '), flush=True)
    
    return df, cov_per_locus


def filter_data(df, max_nchrom_gt2trace=0, min_nonmissing_per_locus=0.1, trace_min_nloci=3,
                trace_min_nloci_ratio=0.1, by_chosen_loci_only=True, ntraces_per_cell=None, 
                enforce_even_spacing=True, spacing=2.5, verbose=True):
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

    if ntraces_per_cell is not None and 'ntraces_per_cell' not in df.columns:
        df = get_ntraces_per_cell(df)

    # ==== Remove cells where >[cutoff] chromosomes have >2 traces
    # If [cutoff] > 0, also remove any extra traces
    if len(df.trace_id.drop_duplicates()) > 0:
        if max_nchrom_gt2trace < len(df.chrom.drop_duplicates()):
            if max_nchrom_gt2trace == 0:
                ncells_orig = len(df.cell_id.drop_duplicates())
                df = df.groupby('cell_id').apply(
                    remove_cells_with_extra_traces, include_groups=False,
                    max_nchrom=max_nchrom_gt2trace).reset_index(level=0)
                ncells_removed = ncells_orig - len(df.cell_id.drop_duplicates())
            else:
                # Remove any extra traces from cell with >2 traces
                raise NotImplementedError()  # Would also need to edit get_ntraces_per_cell function
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
    # (if by_chosen_loci_only=True: only considering number of chosen loci, NOT number of total loci)
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

    # ==== Remove chromosomes with where no cells have >1 trace
    # (if by_chosen_loci_only=True: only considering number of chosen loci, NOT number of total loci)
    if ntraces_per_cell is None:
        nchrom_orig = len(df.chrom.drop_duplicates())
        if by_chosen_loci_only:
            df = df.groupby('chrom').apply(
                lambda x: (None if len(x.loc[x.chosen_loci, 'trace_id'].drop_duplicates()) == 1 else x),
                include_groups=False).reset_index(level=0)
        else:
            df = df.groupby('chrom').apply(
                lambda x: (None if len(x.trace_id.drop_duplicates()) == 1 else x),
                include_groups=False).reset_index(level=0)
        nchrom_removed = nchrom_orig - len(df.chrom.drop_duplicates())
        if verbose:
            print(f"\tRemoved {nchrom_removed}/{nchrom_orig} CHROMOSOMES where no cells have >1 trace", flush=True)
            print(f"\t ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)

    return df


def get_nloci_in_trace(df):
    return pd.Series({'nloci': len(df), 'nloci_2trace': (df.ntraces == 2).sum()})


def summarize_labeling(df, prefix='', verbose=True):
    nmol_labeled = df.loc[~df.hmlg.isnull(), ['cell_id', 'trace_id', 'hmlg', 'ntraces_per_cell']].drop_duplicates(
        ).groupby(['trace_id', 'hmlg', 'ntraces_per_cell']).size().reset_index(level=[0, 1, 2]).rename({0: 'nmol'}, axis=1)
    nmol_labeled['pairing'] = 'A'
    nmol_labeled.loc[nmol_labeled.trace_id != nmol_labeled.hmlg, 'pairing'] = 'B'
    nmol_labeled = nmol_labeled.set_index('pairing').sort_index()

    desc = []
    if (df.ntraces_per_cell == 1).sum():
        ntraces_per_cell_1 = nmol_labeled.loc[nmol_labeled.ntraces_per_cell == 1, 'nmol'].drop_duplicates()
        desc.extend([
            "Pairings for cells with only 1 trace:",
            pd.DataFrame(ntraces_per_cell_1).reset_index().to_string(index=False)])
    if (df.ntraces_per_cell == 2).sum():
        ntraces_per_cell_1 = nmol_labeled.loc[nmol_labeled.ntraces_per_cell == 2, 'nmol'].drop_duplicates()
        desc.extend([
            "Pairings for cells with 2 traces:",
            pd.DataFrame(ntraces_per_cell_1).reset_index().to_string(index=False)])

    if verbose:
        print(prefix + f'\n{prefix}'.join(desc), flush=True)
    
    return nmol_labeled


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


def disterror(struct_hmlg, struct_candidate, invert_distances=False, nrmse_denom=None):
    if len(struct_hmlg) == 0 or len(struct_candidate) == 0:
        print(('(a)', len(struct_hmlg), len(struct_candidate)), flush=True)
        return pd.Series({'err': 0, 'n': 0})
    # While struct_hmlg only includes loci present in struct_candidate...
    # ...we need to make sure that struct_candidate only includes loci present in struct_hmlg
    # for this particular trace of this particular previously-labeled cell
    struct_candidate = struct_candidate[struct_candidate.chrom_order.isin(
        struct_hmlg.chrom_order)]
    if len(struct_candidate) == 0:
        print(('(b)', len(struct_hmlg), len(struct_candidate)), flush=True)
        return pd.Series({'err': 0, 'n': 0})
    # Get bead-bead distances
    dis_candidate = pdist(struct_candidate[['x', 'y', 'z']].values)
    dis_hmlg = pdist(struct_hmlg[['x', 'y', 'z']].values)
    # Get distance error
    if invert_distances:
        dis_candidate = np.power(dis_candidate, -1)
        dis_hmlg = np.power(dis_hmlg, -1)
    err = np.square(dis_candidate - dis_hmlg)
    if nrmse_denom is not None:
        idx = get_distance_matrix_idx(struct_candidate)
        err /= nrmse_denom.loc[idx].values
    err = err.sum()
    # Take the sum of the squared error and record the number of bins ('n') so that
    # we can more easily aggregate data across previously labeled cells
    # (Don't want to take the mean of multiple MSE values since 'n' might differ per cell)
    return pd.Series({'err': err, 'n': dis_hmlg.size})


def label_homologs_for_chrom(df, df_cell_desc, chrom, compare_to_labeled_loci_with_2_traces=False, 
                             compare_to_2traces_per_cell=False, compare_to_minimal_cells=False, 
                             compared_loci_cutoff=None, inverse_disterror=False, nrmse_method=None, 
                             root_mse=False, hmlg_err_ratio_cutoff=None, verbose=True):
    if verbose:
        print(f"\nASSIGNING HOMOLOGS FOR: chr{chrom}", flush=True)
    if nrmse_method is not None:
        nrmse_method = nrmse_method.lower()
        if nrmse_method not in ('mean', 'range', 'iqr'):
            raise ValueError(f"{nrmse_method=}, must be 'mean', 'range' 'iqr', or None.")
    
    cells = df_cell_desc.loc[df_cell_desc.chrom == chrom, ['cell_id', 'ntraces_in_cell']].set_index(
        'cell_id').ntraces_in_cell
    
    # Reset homolog labeling results for this chromosome
    if 'hmlg' in df.columns:
        df.loc[df.chrom == chrom, 'hmlg'] = None
    else:
        df['hmlg'] = None
    if 'hmlg_err_ratio' in df.columns:
        df.loc[df.chrom == chrom, 'hmlg_err_ratio'] = None
    else:
        df['hmlg_err_ratio'] = None
    
    # If None of the cells have >1 trace for this chromosome...
    if 2 not in cells.drop_duplicates().values:
        raise NotImplementedError

    # Sort values (in case using NRMSE)
    df = df.sort_values(['cell_id', 'trace_id', 'chrom_order']).reset_index(drop=True)

    # Select which loci to use for this chromosome
    chrom_loci_to_compare = df.chrom == chrom  # Using all loci for the given chrom
    if compared_loci_cutoff is not None and compared_loci_cutoff > 0:
        if compared_loci_cutoff >= 1:
            raise ValueError(f"{compared_loci_cutoff=:.g}, must be between 0 and 1")
        top_loci = df.loc[chrom_loci_to_compare, ['chrom_order', 'trace_id', 'cell_id']].groupby(
            'chrom_order').size().sort_values()
        nloci_orig = len(top_loci)
        top_loci = top_loci[top_loci >= top_loci.quantile(compared_loci_cutoff)].index.values
        print(f"...Only using loci in top {compared_loci_cutoff * 100:.4g} percentile... {len(top_loci)}/{nloci_orig}", flush=True)
        chrom_loci_to_compare = chrom_loci_to_compare & df.chrom_order.isin(top_loci)

    # Which loci have data for this chromosome?
    loci_with_data = df.loc[chrom_loci_to_compare, 'chrom_order'].drop_duplicates().sort_values().values
    all_loci_labeled = False

    # Get distance matrix mean (across molecules) for each locus pair
    nrmse_denom = None
    if nrmse_method is not None:
        print(f"\tSetting up for NRMSE (method={nrmse_method})...", flush=True)
        if compare_to_2traces_per_cell:
            df_tmp = df[df.ntraces_per_cell == 2]
        else:
            df_tmp = df
        nrmse_denom = df_tmp.groupby(['cell_id', 'trace_id']).apply(
            get_distance_matrix, include_groups=False, invert_distances=inverse_disterror).reset_index(
            [0, 1, 2], drop=True).groupby('idx')
        if nrmse_method == 'mean':
            nrmse_denom = nrmse_denom.dis.mean()
        elif nrmse_method == 'iqr':
            nrmse_denom = nrmse_denom.dis.quantile(0.75) - nrmse_denom.dis.quantile(0.25)
        else:
            raise NotImplementedError
        nrmse_denom = nrmse_denom ** 2  # Because will divide by denom before taking sqrt
        print("\t\t...Done with setting up for NRMSE!", flush=True)

    # Arbitrarily label homologs of the first cell
    first_cell = cells.index[0]
    assert cells.loc[first_cell] == 2  # The first cell has 2 traces for the give chrom
    mask = (df.chrom == chrom) & (df.cell_id == first_cell)
    df.loc[mask, 'hmlg'] = df.loc[mask, 'trace_id']

    pass_hmlg_err_ratio_cutoff = ncells_eligible = 0
    for cell_id, ntraces_per_cell in tqdm(cells.to_dict().items()):
        if cell_id == first_cell:  # First cell has already been labeled
            df_chrom = df[chrom_loci_to_compare]  # Initial selection of data for chrom
            ntraces_per_prev_cell = 2
            continue

        # Re-select data for chrom (because 'hmlg' label in 'df' got updated for prev cell)
        if ntraces_per_prev_cell == 2 or not (compare_to_labeled_loci_with_2_traces or compare_to_2traces_per_cell):
            # Check if all loci have been labled at least once in each hmlg
            if compare_to_minimal_cells and not all_loci_labeled:
                loci_with_labeled_data = df.loc[chrom_loci_to_compare & (
                    ~df.hmlg.isnull()), ['chrom_order', 'hmlg']].drop_duplicates(
                    ).groupby('chrom_order').size()
                loci_with_labeled_data = loci_with_labeled_data[loci_with_labeled_data == 2].index.values
                if len(loci_with_data) == len(loci_with_labeled_data):
                    all_loci_labeled = True

            if all_loci_labeled or not compare_to_minimal_cells:
                # Only re-select data if the new data will be used as a baseline for labeling
                df_chrom = df[chrom_loci_to_compare]
    
        # Get all traces for the candidate cell, only including loci present in all of the candidate cell's traces
        candidate = df_chrom.loc[(df_chrom.cell_id == cell_id) & (df_chrom.ntraces == ntraces_per_cell)]
        candidate = [candidate.loc[
                     candidate.trace_id == trace_id,
                     ['chrom_order', 'x', 'y', 'z']] for trace_id in np.arange(1, ntraces_per_cell + 1)]
        loci_candidate = candidate[0].chrom_order.values
    
        # Get all labeled traces from previous cells, only including loci present in all of the candidate cell's traces
        prev_labeled = df_chrom[(~df_chrom.hmlg.isnull()) & df_chrom.chrom_order.isin(loci_candidate)]
        if compare_to_labeled_loci_with_2_traces:
            prev_labeled = prev_labeled[prev_labeled.ntraces == 2]
        if hmlg_err_ratio_cutoff is not None and hmlg_err_ratio_cutoff < 1:
            prev_labeled = prev_labeled[prev_labeled.hmlg_err_ratio.isnull() | (prev_labeled.hmlg_err_ratio <= hmlg_err_ratio_cutoff)]
        hmlgs = prev_labeled[['cell_id', 'hmlg', 'chrom_order', 'x', 'y', 'z']]
    
        # Calculate distance error between each candidate trace and the labeled traces from each previous cell
        res = [hmlgs.groupby(['cell_id', 'hmlg']).apply(
            disterror, include_groups=False, struct_candidate=x,
            invert_distances=inverse_disterror, nrmse_denom=nrmse_denom).groupby(level=1).sum() for x in candidate]
        res = [x.err / x.n for x in res if (x.n != 0).all()]  # Get MSE of distance matrices per pairing of traces
        if root_mse or nrmse_method is not None:
            res = [np.sqrt(x) for x in res]

        # Only proceed to labeling homologs if we have similarity scores for all candidate traces
        if len(res) != ntraces_per_cell:
            ntraces_per_prev_cell = ntraces_per_cell  # Keep track of prev cell's info
            continue
            
        # Indices that determine how the candidate cell's trace(s) get paired to the pre-labled hmlgs
        pairingA = np.arange(ntraces_per_cell) + 1  # Pair trace1 to hmlg1 (and trace2 to hmlg2)
        pairingB = 3 - pairingA  # Pair trace1 to hmlg2 (and trace2 to hmlg1)
    
        # Compare mean distance error for each pairing
        # (Can take the mean because the number of loci is the same for all traces in the candidate cell)
        resA = np.mean([res[i].loc[pairingA[i]] for i in range(ntraces_per_cell)])
        resB = np.mean([res[i].loc[pairingB[i]] for i in range(ntraces_per_cell)])
    
        # Assign the homolog itentities to candidate cell based on min distance error of the pairings
        mask = (df.chrom == chrom) & (df.cell_id == cell_id)
        if resA <= resB:
            df.loc[mask, 'hmlg'] = df.loc[mask, 'trace_id']
        else:
            df.loc[mask, 'hmlg'] = 3 - df.loc[mask, 'trace_id']
        hmlg_err_ratio = min(resA, resB) / max(resA, resB)
        df.loc[mask, 'hmlg_err_ratio'] = hmlg_err_ratio

        # TODO
        if ntraces_per_prev_cell == 2 or not (compare_to_labeled_loci_with_2_traces or compare_to_2traces_per_cell):
            if hmlg_err_ratio_cutoff is not None and hmlg_err_ratio_cutoff < 1:
                if hmlg_err_ratio <= hmlg_err_ratio_cutoff:
                    pass_hmlg_err_ratio_cutoff += 1
                ncells_eligible += 1
        
        # median_hmlg_err_ratio = df_chrom.loc[~df_chrom.hmlg.isnull(), ['cell_id', 'hmlg', 'hmlg_err_ratio']].drop_duplicates().hmlg_err_ratio.median()
        # print(f"{('✓' if hmlg_err_ratio <= hmlg_err_ratio_cutoff else ' ')}\tcurrent={hmlg_err_ratio * 100:.3g}\tmedian={median_hmlg_err_ratio * 100:.3g}", flush=True) # ✗

        # Keep track of this cell's ntraces_per_cell when analyzing the next cell
        ntraces_per_prev_cell = ntraces_per_cell

    # Assess labeling results
    if verbose:
        nmol_labeled = summarize_labeling(df, prefix='\t')
        if hmlg_err_ratio_cutoff is not None and hmlg_err_ratio_cutoff < 1:
            print(f"\n\t{round(pass_hmlg_err_ratio_cutoff / ncells_eligible * 100)}% of cells pass hmlg_err_ratio cutoff ({pass_hmlg_err_ratio_cutoff}/{ncells_eligible})", flush=True)

    return df


def get_ntraces_per_cell(df):
    df.set_index(['cell_id', 'chrom'], inplace=True)
    has_2traces_idx = df[df.trace_id == 1].index.intersection(df[df.trace_id == 2].index)
    df['ntraces_per_cell'] = 1
    df.loc[has_2traces_idx, 'ntraces_per_cell'] = 2
    df.reset_index(inplace=True)
    return df


def preprocess_data(input_file, chosen_loci_file, max_nchrom_gt2trace=0, min_nonmissing_per_locus=0.1, trace_min_nloci=3,
                    trace_min_nloci_ratio=0.1, by_chosen_loci_only=True, enforce_even_spacing=True, ntraces_per_cell=None,
                    interpolate=False, spacing=2.5, verbose=True):
    if os.path.exists(input_file + '.gz') and not os.path.exists(input_file):
        input_file = input_file + '.gz'
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

    # ==== Count number of traces (per CHROM, per cell)... *prior to filtering*
    df = get_ntraces_per_cell(df)

    # ==== Filter data
    if interpolate:  # Edit filtering criteria when interpolating
        min_nonmissing_per_locus = 0
        trace_min_nloci = 3
        trace_min_nloci_ratio = 0
        by_chosen_loci_only = False
        enforce_even_spacing = False
    df = filter_data(
        df, max_nchrom_gt2trace=max_nchrom_gt2trace, min_nonmissing_per_locus=min_nonmissing_per_locus,
        trace_min_nloci=trace_min_nloci, trace_min_nloci_ratio=trace_min_nloci_ratio,
        by_chosen_loci_only=by_chosen_loci_only, ntraces_per_cell=ntraces_per_cell,
        enforce_even_spacing=enforce_even_spacing, spacing=spacing, verbose=verbose)

    # ==== Optional: Interpolate
    if interpolate:
        df = interp_all(df, spacing=spacing, verbose=verbose)
        df['chrom_order'] = df.idx_chrom

    return df


def restrict_to_equal_nmol_per_hmlg(data, cell_desc_file=None, cutoff_ratio=1, verbose=True):
    if verbose:
        print("Filtering data for each chromosome such that the number of cells in each homolog are"
              f" within {cutoff_ratio:.3g}x of each other", flush=True)

    if isinstance(data, str):
        df = pd.read_csv(data)
        if cell_desc_file is None:
            cell_desc_file = os.path.join(os.path.dirname(data), 'cell_data_for_hmlg_labeling.csv')
    else:
        df = data
        if cell_desc_file is None:
            raise ValueError("Must input cell_desc_file.")
    if cutoff_ratio is None:
        return df
    nrows_orig = len(df)

    # For each chromosome, order cells by how much data they have
    df_cell_desc = pd.read_csv(cell_desc_file).sort_values(
        ['chrom', 'ntraces_in_cell', 'nloci_per_cell', 'nloci_2trace'], ascending=False).reset_index(drop=True)
    df_cell_desc['labeling_order'] = df_cell_desc.groupby('chrom').apply(lambda x: pd.Series(
        np.arange(1, len(x) + 1), index=x.index), include_groups=False).reset_index(level=0, drop=True).sort_index()
    # Ensure dtypes match up between cell description and sc data
    if (np.issubdtype(df.chrom.dtype, int) or np.issubdtype(df.chrom.dtype, float)) and not (
            np.issubdtype(df_cell_desc.chrom.dtype, int) or np.issubdtype(df_cell_desc.chrom.dtype, float)):
        df_cell_desc = df_cell_desc[df_cell_desc.chrom.isin(df.chrom.drop_duplicates().astype(str))]
        try:
            df_cell_desc['chrom'] = pd.to_numeric(df_cell_desc.chrom)  # Try to make numeric
        except ValueError:
            pass
        if not (np.issubdtype(df_cell_desc.chrom.dtype, int) or np.issubdtype(df_cell_desc.chrom.dtype, float)):
            df['chrom'] = df.chrom.astype(str)
    # Merge data
    df = df.merge(df_cell_desc[['cell_id', 'chrom', 'labeling_order']], on=['cell_id', 'chrom'], how='left')

    # For each chromosome, get the minimum number of cells detected across the two homologs
    min_ncell_per_hmlg = df[['chrom', 'hmlg', 'cell_id']].drop_duplicates().groupby(
        ['chrom', 'hmlg']).size().reset_index(level=1).groupby(level=0).min().reset_index().rename(
        {0: 'min_nmol'}, axis=1).drop('hmlg', axis=1)
    df = df.merge(min_ncell_per_hmlg, on='chrom', how='left')

    df = df[df.labeling_order <= df.min_nmol * cutoff_ratio]

    if verbose:
        print(f" ↳ Current n={len(df):,}, {len(df) / nrows_orig * 100:.3g}% of original", flush=True)

    df.drop(['labeling_order', 'min_nmol'], axis=1, inplace=True)

    return df


def get_full_outdir(output_dir, ntraces_per_cell=None, compare_to_labeled_loci_with_2_traces=False,
                    compare_to_2traces_per_cell=False, compared_loci_cutoff=None, compare_to_minimal_cells=False,
                   interpolate=False, inverse_disterror=False, nrmse_method=None, root_mse=False,
                   hmlg_err_ratio_cutoff=None):
    subdir = []
    if ntraces_per_cell is not None:
        subdir.append(f'ntraces_chrom-cell_{ntraces_per_cell}')
    if hmlg_err_ratio_cutoff == 0:
        subdir.append('compare_to_1cell')
    if compare_to_labeled_loci_with_2_traces:
        subdir.append('compare_to_2traces_per_locus')
    elif compare_to_2traces_per_cell and ntraces_per_cell is None and hmlg_err_ratio_cutoff != 0:
        subdir.append('compare_to_2traces_per_chrom-cell')
    if compared_loci_cutoff is not None and compared_loci_cutoff > 0:
        subdir.append(f'compared_loci_{compared_loci_cutoff * 100:.4g}p')
    if compare_to_minimal_cells:
        subdir.append('compare_to_minimal_cells')
    if interpolate:
        subdir.append('interp')
    if inverse_disterror or root_mse or nrmse_method is not None:
        tmp = ['disterror']
        if inverse_disterror:
            tmp.append('inverse')
        if nrmse_method is not None:
            tmp.append(f'nrmse-{nrmse_method}')
        elif root_mse:
            tmp.append('rmse')
        subdir.append('_'.join(tmp))
    if hmlg_err_ratio_cutoff is not None and hmlg_err_ratio_cutoff < 1 and hmlg_err_ratio_cutoff > 0:
        subdir.append(f'hmlg_err_ratio_{hmlg_err_ratio_cutoff * 100:.4g}p')
    if len(subdir) != 0:
        output_dir = os.path.join(output_dir, '.'.join(subdir))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def process_data(input_file, chosen_loci_file, max_nchrom_gt2trace=0, min_nonmissing_per_locus=0.1, trace_min_nloci=3,
                 trace_min_nloci_ratio=0.1, by_chosen_loci_only=True, ntraces_per_cell=None, enforce_even_spacing=True,
                 interpolate=False, compare_to_labeled_loci_with_2_traces=False, compare_to_2traces_per_cell=False,
                 compare_to_minimal_cells=False, compared_loci_cutoff=None, inverse_disterror=False, nrmse_method=None,
                 root_mse=False, hmlg_err_ratio_cutoff=None, spacing=2.5, chrom_to_filter=None, output_dir=None, verbose=True):
    if output_dir is not None:
        output_dir = get_full_outdir(
            output_dir, ntraces_per_cell=ntraces_per_cell,
            compare_to_labeled_loci_with_2_traces=compare_to_labeled_loci_with_2_traces,
            compare_to_2traces_per_cell=compare_to_2traces_per_cell, compared_loci_cutoff=compared_loci_cutoff,
            compare_to_minimal_cells=compare_to_minimal_cells, interpolate=interpolate, inverse_disterror=inverse_disterror,
            nrmse_method=nrmse_method, root_mse=root_mse, hmlg_err_ratio_cutoff=hmlg_err_ratio_cutoff)
        output_file = os.path.join(
            output_dir, re.sub(r'\.csv$', '', os.path.basename(input_file)) + '.filter.hmlg.csv')
        print(output_file + '\n', flush=True)

    # ==== Load & filter data
    default_preprocessing_file = None
    if (max_nchrom_gt2trace == 0) and (min_nonmissing_per_locus == 0.1) and (trace_min_nloci == 3) and (
            trace_min_nloci_ratio == 0.1) and (by_chosen_loci_only) and (ntraces_per_cell is None) and (
            enforce_even_spacing) and (not interpolate):
        default_preprocessing_file = re.sub(r'\.csv$', '', input_file) + '.filter-default.csv'
        print(f'Preprocessed data file: {default_preprocessing_file}\n', flush=True)
    if default_preprocessing_file is not None and os.path.exists(default_preprocessing_file):
        df = pd.read_csv(default_preprocessing_file)
    else:
        df = preprocess_data(
            input_file, chosen_loci_file=chosen_loci_file, max_nchrom_gt2trace=max_nchrom_gt2trace,
            min_nonmissing_per_locus=min_nonmissing_per_locus, trace_min_nloci=trace_min_nloci,
            trace_min_nloci_ratio=trace_min_nloci_ratio, by_chosen_loci_only=by_chosen_loci_only,
            ntraces_per_cell=ntraces_per_cell, enforce_even_spacing=enforce_even_spacing,
            interpolate=interpolate, spacing=spacing, verbose=verbose)
        if default_preprocessing_file is not None:
            df.to_csv(default_preprocessing_file, index=False)

    # ==== Per chromosome: get ordering of cells in which homologs will be labled
    # Count number of traces in data (per LOCUS, per cell)
    df.set_index(['cell_id', 'chrom', 'chrom_order'], inplace=True)
    has_2traces_idx = df[df.trace_id == 1].index.intersection(df[df.trace_id == 2].index)
    df['ntraces'] = 1
    df.loc[has_2traces_idx, 'ntraces'] = 2
    df.reset_index(inplace=True)
    
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

    # Save cell/chrom info
    if output_dir is not None:
        df_cell_desc.to_csv(os.path.join(output_dir, 'cell_data_for_hmlg_labeling.csv'), index=False)

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
        elif compare_to_2traces_per_cell:
            print("\t...only compare to previously labeled cells in which BOTH homologs were detected!", flush=True)
        else:
            print("\t...compare to all loci in each previously labeled cell (even if it's only present in 1 homolog)",
                  flush=True)
        if compared_loci_cutoff is not None and compared_loci_cutoff > 0:
            print(f"\t...compare ONLY to the most commonly detected loci - detection in top {compared_loci_cutoff * 100:.4g} percentile", flush=True)
        if compare_to_minimal_cells:
            print("\t...compare to minimal set of previously labeled cells in which all loci are represented", flush=True)
        if inverse_disterror:
            print("\t...comparing INVERSE distances for similarity score", flush=True)
        if nrmse_method is not None:
            print(f"...using NRMSE with denominator={nrmse_method}", flush=True)
        elif root_mse:
            print("...using RMSE (not MSE)", flush=True)
        if hmlg_err_ratio_cutoff is not None and hmlg_err_ratio_cutoff < 1:
            print(f"\t...compare ONLY to the most easily-labeled cells... cutoff ratio = {hmlg_err_ratio_cutoff * 100:.4g}", flush=True)
            
    df['hmlg'] = None
    df['hmlg_err_ratio'] = None
    df.sort_values(['chrom', 'cell_id', 'trace_id', 'chrom_order'], inplace=True)
    for chrom in chrom_to_filter:
        df = label_homologs_for_chrom(
            df, df_cell_desc=df_cell_desc, chrom=chrom,
            compare_to_labeled_loci_with_2_traces=compare_to_labeled_loci_with_2_traces,
            compare_to_2traces_per_cell=compare_to_2traces_per_cell, compare_to_minimal_cells=compare_to_minimal_cells,
            compared_loci_cutoff=compared_loci_cutoff, inverse_disterror=inverse_disterror, nrmse_method=nrmse_method,
            root_mse=root_mse, hmlg_err_ratio_cutoff=hmlg_err_ratio_cutoff, verbose=verbose)

    # ==== Only keep 'chosen' loci (eg those spaced 2.5Mb apart across the genome)
    df = df[df.chosen_loci].drop('chosen_loci', axis=1)

    # ==== Optionally save results
    if output_dir is not None:
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
    parser.add_argument('--filter-via-all-loci', default=True,
                        dest="by_chosen_loci_only", action='store_false')
    parser.add_argument("--ntraces_per_cell", default=None, type=int)
    parser.add_argument('--dont-enforce-even-spacing', default=True,
                        dest="enforce_even_spacing", action='store_false')
    parser.add_argument('--interpolate', default=False,
                        action='store_true')
    parser.add_argument('--compare_to_labeled_loci_with_2_traces', default=False,
                        action='store_true')
    parser.add_argument('--compare_to_2traces_per_cell', default=False, action='store_true')
    parser.add_argument('--compare_to_minimal_cells', default=False, action='store_true')
    parser.add_argument('--compared_loci_cutoff', type=float)
    parser.add_argument('--inverse_disterror', default=False, action='store_true')
    parser.add_argument("--nrmse_method", type=str)
    parser.add_argument('--root_mse', default=False, action='store_true')
    parser.add_argument('--hmlg_err_ratio_cutoff', type=float)
    parser.add_argument("--spacing", default=2.5, type=float)    
    parser.add_argument("--chrom", type=str, nargs='+')
    parser.add_argument('--verbose', default=False, action='store_true')


    
    args = parser.parse_args()

    output_dir = os.path.dirname(args.data)
    process_data(
        input_file=args.data, chosen_loci_file=args.chosen_loci, max_nchrom_gt2trace=args.max_nchrom_gt2trace,
        min_nonmissing_per_locus=args.min_nonmissing_per_locus, trace_min_nloci=args.trace_min_nloci,
        trace_min_nloci_ratio=args.trace_min_nloci_ratio, by_chosen_loci_only=args.by_chosen_loci_only,
        ntraces_per_cell=args.ntraces_per_cell, enforce_even_spacing=args.enforce_even_spacing,
        interpolate=args.interpolate, compare_to_labeled_loci_with_2_traces=args.compare_to_labeled_loci_with_2_traces,
        compare_to_2traces_per_cell=args.compare_to_2traces_per_cell, compare_to_minimal_cells=args.compare_to_minimal_cells,
        compared_loci_cutoff=args.compared_loci_cutoff, inverse_disterror=args.inverse_disterror,
        nrmse_method=args.nrmse_method, root_mse=args.root_mse, hmlg_err_ratio_cutoff=args.hmlg_err_ratio_cutoff,
        chrom_to_filter=args.chrom, spacing=args.spacing, output_dir=output_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
