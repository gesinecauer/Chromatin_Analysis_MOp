import os
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from parse_dna_merfish import preprocess_data
from tqdm import tqdm
from scipy import sparse
from functools import partial


def disterror(molecule_i, molecule_j):
    molecule_i = molecule_i[molecule_i.chrom_order.isin(molecule_j.chrom_order)]
    molecule_j = molecule_j[molecule_j.chrom_order.isin(molecule_i.chrom_order)]
    if not len(molecule_j):
        return np.nan
    dis_i = pdist(molecule_i[['x', 'y', 'z']].values)
    dis_j = pdist(molecule_j[['x', 'y', 'z']].values)
    return np.square(dis_i - dis_j).sum() / dis_i.size



def filter_molecules(df, restrict_single_hmlg_to=None, max_molecules_per_grp=None, choose_molecules_randomly=True, rng=None):
    if restrict_single_hmlg_to is None and max_molecules_per_grp is None:
        return df

    if not choose_molecules_randomly:
        df.sort_values(['nloci', 'ntraces_per_cell'], ascending=False, inplace=True)
    elif rng is None:
        rng = np.random.default_rng(seed)
    df['keep'] = False

    if restrict_single_hmlg_to is not None:
        by_num_hmlg = df.groupby('ntraces_per_cell').size()
        df.loc[df.ntraces_per_cell == 2, 'keep'] = True
        if choose_molecules_randomly:
            cutoff_perc = by_num_hmlg.loc[2] * restrict_single_hmlg_to / by_num_hmlg.loc[1]
            df.loc[df.ntraces_per_cell == 1, 'keep'] = rng.uniform(size=by_num_hmlg.loc[1]) < cutoff_perc
        else:
            full_idx = df[df.ntraces_per_cell == 1].index.values
            keep_idx = full_idx[:(by_num_hmlg.loc[2] * restrict_single_hmlg_to)]
            df.loc[keep_idx, 'keep'] = True
    if max_molecules_per_grp is not None:
        for ntraces in [1, 2]:
            full_idx = df[df.ntraces_per_cell == ntraces].index.values
            if choose_molecules_randomly:
                keep_idx = rng.choice(full_idx, size=min(max_molecules_per_grp, full_idx.size), replace=False)
            else:
                keep_idx = full_idx[:max_molecules_per_grp]
            df.loc[keep_idx, 'keep'] = True

    df = df[df.keep].drop('keep', axis=1)
    return df


def get_chrom_idx(df):
    df['idx'] = np.arange(len(df), dtype=int)
    return df


def get_ntraces_per_cell(df):
    df.set_index(['cell_id', 'chrom'], inplace=True)
    has_2traces_idx = df[df.trace_id == 1].index.intersection(df[df.trace_id == 2].index)
    df['ntraces_per_cell'] = 1
    df.loc[has_2traces_idx, 'ntraces_per_cell'] = 2
    df.reset_index(inplace=True)
    return df


def molecule_umap(input_file, chosen_loci_file, trace_min_nloci=20, trace_min_nloci_ratio=0.4,
                  min_nonmissing_per_locus=0.1, restrict_single_hmlg_to=None, max_molecules_per_grp=None,
                  seed=0, choose_molecules_randomly=True, chromosomes=None, verbose=True):

    output_dir = os.path.join(
        os.path.dirname(input_file), "molecule_umap", f"trace_min_loci_{trace_min_nloci_ratio * 100:g}perc")
    if restrict_single_hmlg_to is not None:
        output_dir += f".restrict_nhmlg_ratio_{restrict_single_hmlg_to:d}x"
    if max_molecules_per_grp is not None:
        output_dir += f".max{max_molecules_per_grp:d}_mol_per_grp"
    if choose_molecules_randomly:
        output_dir += f".choose_random"
    else:
        output_dir += f".choose_maxloci"
    print(output_dir, flush=True)
    os.makedirs(output_dir, exist_ok=True)
    molecules_file = os.path.join(output_dir, "molecules.csv")
    mse_file = os.path.join(output_dir, "mse.npz")
    rng = np.random.default_rng(seed)
    df_loader = partial(
        preprocess_data, input_file=input_file, chosen_loci_file=chosen_loci_file,
        trace_min_nloci=trace_min_nloci, trace_min_nloci_ratio=trace_min_nloci_ratio, max_nchrom_gt2trace=0,
        min_nonmissing_per_locus=min_nonmissing_per_locus)

    # if not (os.path.exists(molecules_file) and os.path.exists(mse_file)):
    #     df = preprocess_data(
    #         input_file=input_file, chosen_loci_file=chosen_loci_file, trace_min_nloci=trace_min_nloci,
    #         trace_min_nloci_ratio=trace_min_nloci_ratio, max_nchrom_gt2trace=0,
    #         min_nonmissing_per_locus=min_nonmissing_per_locus, verbose=verbose)
    #     df.set_index(['cell_id', 'chrom'], inplace=True)
    #     has_2traces_idx = df[df.trace_id == 1].index.intersection(df[df.trace_id == 2].index)
    #     df['ntraces_per_cell'] = 1
    #     df.loc[has_2traces_idx, 'ntraces_per_cell'] = 2
    #     df.reset_index(inplace=True)

    if os.path.exists(molecules_file):
        df = None
        molecules = pd.read_csv(molecules_file)
    else:
        df = get_ntraces_per_cell(df_loader(verbose=verbose))
        print(flush=True)
        molecules = df.groupby(['cell_id', 'chrom', 'trace_id', 'ntraces_per_cell']).size().reset_index(level=[0, 1, 2, 3]).rename(
            {0: 'nloci'}, axis=1)
        # Filter molecules
        if (restrict_single_hmlg_to is not None) or (max_molecules_per_grp is not None):
            print(molecules.groupby(['chrom', 'ntraces_per_cell']).size().unstack(level=1).fillna(0).astype(int).to_string(), flush=True)
            if restrict_single_hmlg_to is not None:
                print(f"\nRestrict number of molecules from cells with only 1 homolog: {restrict_single_hmlg_to:g}x", flush=True)
            if max_molecules_per_grp is not None:
                print(f"\nMax number of molecules per group: {max_molecules_per_grp:d}", flush=True)
            molecules = molecules.sort_values(['chrom', 'nloci', 'ntraces_per_cell'], ascending=False).reset_index(drop=True)
            molecules = molecules.groupby('chrom').apply(
                filter_molecules, include_groups=False, restrict_single_hmlg_to=restrict_single_hmlg_to,
                max_molecules_per_grp=max_molecules_per_grp, choose_molecules_randomly=choose_molecules_randomly,
                rng=rng).reset_index(level=0)
        # Get index per molecule
        molecules = molecules.sort_values(['chrom', 'ntraces_per_cell', 'nloci'], ascending=False).reset_index(drop=True)
        molecules = molecules.groupby('chrom').apply(get_chrom_idx, include_groups=False).reset_index(level=0).sort_index()
        # Save
        molecules.to_csv(molecules_file, index=False)
    print(molecules.groupby(['chrom', 'ntraces_per_cell']).size().unstack(level=1).fillna(0).astype(int).to_string(), flush=True)

    if chromosomes is None:
        chromosomes = molecules.chrom.drop_duplicates()
    elif isinstance(chromosomes, (str, int)):
        chromosomes = [str(chromosomes)]
    mse = {chrom: None for chrom in chromosomes}
    for chrom in chromosomes:
        mse_file = os.path.join(output_dir, f"chr{chrom}_mse.npz")
        if os.path.exists(mse_file):
            mse_matrix = sparse.load_npz(mse_file)
        else:
            if df is None:
                df = get_ntraces_per_cell(df_loader(verbose=False))
            print(f"\nCHROMOSOME {chrom}", flush=True)
            molecules_chrom = molecules[molecules.chrom == chrom].set_index('idx')

            nmolecules = len(molecules_chrom)
            idx = np.indices((nmolecules, nmolecules)).reshape(-1, nmolecules ** 2)
            idx = idx[:, idx[0] > idx[1]].T
            
            mse_arr = np.zeros(idx.shape[0], dtype=float)
            for k in tqdm(idx):
                i, j = k
                molecule_i = df[(df.cell_id == molecules_chrom.loc[i, 'cell_id']) & (df.chrom == molecules_chrom.loc[i, 'chrom']) & (
                    df.trace_id == molecules_chrom.loc[i, 'trace_id'])]
                molecule_j = df[(df.cell_id == molecules_chrom.loc[j, 'cell_id']) & (df.chrom == molecules_chrom.loc[j, 'chrom']) & (
                    df.trace_id == molecules_chrom.loc[j, 'trace_id'])]
                mse_arr[k] = disterror(molecule_i, molecule_j)
            
            mse_matrix = sparse.coo_matrix((mse_arr, (idx[:, 0], idx[:, 1])), shape=(nmolecules, nmolecules))
            sparse.save_npz(mse_file, mse_matrix)
        mse[chrom] = mse_matrix

    return molecules, mse


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--chosen_loci", required=True, type=str)
    parser.add_argument("--min_nonmissing_per_locus", default=0.1, type=float)
    parser.add_argument("--trace_min_nloci", default=3, type=int)
    parser.add_argument("--trace_min_nloci_ratio", default=0.1, type=float)
    parser.add_argument("--restrict_single_hmlg_to", type=int)
    parser.add_argument("--max_molecules_per_grp", type=int)
    parser.add_argument('--maximize_nloci', dest="choose_molecules_randomly", default=True, action='store_false')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument("--chromosomes", nargs="+", type=str)
    args = parser.parse_args()

    molecule_umap(
        input_file=args.data, chosen_loci_file=args.chosen_loci,
        min_nonmissing_per_locus=args.min_nonmissing_per_locus, trace_min_nloci=args.trace_min_nloci,
        trace_min_nloci_ratio=args.trace_min_nloci_ratio, restrict_single_hmlg_to=args.restrict_single_hmlg_to,
        max_molecules_per_grp=args.max_molecules_per_grp, choose_molecules_randomly=args.choose_molecules_randomly,
        chromosomes=args.chromosomes, verbose=args.verbose)


if __name__ == "__main__":
    main()
