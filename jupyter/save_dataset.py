import numpy as np
import pandas as pd
import os
import re
import ast
import glob
from scipy import sparse
from coords_to_matrix import process_sc_dna_coords
from iced.io import write_counts, write_lengths
from topsy.plot.plot_distances import plot_distance_matrix
from topsy.plot.plot_counts import plot_counts_single
from topsy.analysis.compare_distances import make_matrix_df, get_other_struct_features
from topsy.analysis.compare_distances import load_sc_dis_per_locus
from topsy.analysis.utils import get_nghbr_dis_var
from topsy.utils.misc import symlink
from estimate_alpha import infer_alpha_float_vs_int


def filter_matrix_df(matrix_df, mask):
    """Convenience function for filtering by boolean column"""
    if mask is None:
        matrix_df = matrix_df
    elif isinstance(mask, str):
        mask_cols = [c for c in matrix_df.columns if np.issubdtype(
            matrix_df[c].dtype, bool)]
        if mask in mask_cols:
            matrix_df = matrix_df[matrix_df[mask]]
        elif f"~{mask}" in mask_cols:
            matrix_df = matrix_df[~matrix_df[mask]]
        else:
            raise ValueError(f"{mask=} not understood")
    else:
        matrix_df = matrix_df[mask]
    return matrix_df


def prep_matrix_df(lengths_df, matrices, scale_dis_by='mean', nonmissing_bin_percentile=None):
    if set(list(matrices.keys())) != {'counts', 'nonmissing', 'dis_mean', 'dis_median'}:
        raise ValueError("Inputted matrices must contain: 'counts', 'nonmissing', 'dis_mean', 'dis_median'")
    
    if scale_dis_by is not None:
        scale_dis_by = scale_dis_by.lower()
        if scale_dis_by not in ('mean', 'median'):
            raise ValueError("'scale_dis_by' must be 'mean', 'median', or None.")

    rows, cols = np.triu_indices(len(lengths_df) * 2, 1)

    matrix_df = make_matrix_df(lengths_df=lengths_df, matrix_dict=matrices)
    matrix_df['nonmissing'] = matrix_df['nonmissing'].astype(int)

    matrix_df['genomic_dis'] = None
    matrix_df.loc[matrix_df['mask.sameC-sameH'], 'genomic_dis'] = (matrix_df['i.idx'] - matrix_df['j.idx']).abs()
    matrix_df['mask.nghbr'] = matrix_df['mask.sameC-sameH'] & (matrix_df.genomic_dis == 1)

    # Remove loci that are entirely missing in sc dataset
    n = len(lengths_df)  # n = nbeads / ploidy
    excluded_loci = np.where((matrices['nonmissing'] == 0).all(axis=0))[0]
    excluded_loci[excluded_loci >= n] -= n
    excluded_loci = np.unique(excluded_loci)  # If excluded in one hmlg, exclude in both
    mask = (~matrix_df['i.idx_ambig'].isin(excluded_loci)) & (~matrix_df['j.idx_ambig'].isin(excluded_loci))
    matrix_df = matrix_df[mask]

    # Scale distances by distance between neighboring beads (along a molecule)
    if scale_dis_by is None:
        dist_scale_factor = 1
    else:
        if scale_dis_by == 'mean':
            # dist_scale_factors['mean'] = matrix_df.loc[matrix_df['mask.nghbr'], 'dis_mean'].mean()
            dist_scale_factor = matrix_df.loc[matrix_df['mask.nghbr'], 'dis_median'].mean()
        elif scale_dis_by == 'median':
            # dist_scale_factors['mean'] = matrix_df.loc[matrix_df['mask.nghbr'], 'dis_mean'].median()
            dist_scale_factor = matrix_df.loc[matrix_df['mask.nghbr'], 'dis_median'].median()
        matrix_df['dis_mean'] /= dist_scale_factor
        matrix_df['dis_median'] /= dist_scale_factor

    # Filter locus pairs / bins
    if nonmissing_bin_percentile is not None:
        raise ValueError("nonmissing_bin_percentile should be none, best to filter by locus, not locus pair / bin")
        matrix_df = matrix_df[matrix_df.nonmissing > matrix_df.nonmissing.quantile(
            nonmissing_bin_percentile)]

    return matrix_df, dist_scale_factor


def determine_max_nreads(matrix_df, verbose=True):
    matrix_df['numerator'] = (matrix_df.counts * matrix_df.nonmissing).round(2).astype(int)

    mask = (~matrix_df.nonmissing.isnull()) & (matrix_df.counts != 0) & (matrix_df.numerator == 1)
    nonmissing_filt = matrix_df.loc[mask, 'nonmissing']
    if verbose > 1:
        print(nonmissing_filt.describe().round(2).to_string(), flush=True)
    
    fact = nonmissing_filt.mean()
    nreads = np.nansum(matrix_df.counts.values) * fact
    if verbose:
        print(f"{fact = :.3g} = 1/{1/fact:.3g}...    {nreads = :.3g}", flush=True)

    return nreads


def get_integer_counts(matrix_df, nreads='auto', verbose=True):
    if nreads is None or isinstance(nreads, str) and nreads.lower() == 'auto':
        nreads = determine_max_nreads(matrix_df, verbose=verbose)
    matrix_df['counts_int'] = (
        matrix_df.counts * nreads / matrix_df.counts.sum()).round().astype(int)

    return matrix_df


def filter_loci_by_cell_cov_percentile(lengths_df, percentile, verbose=True):
    cell_cov_min = lengths_df.cell_cov_min.copy()
    cell_cov_min[cell_cov_min.isnull()] = 0

    if percentile is None or not percentile or not cell_cov_min.quantile(percentile):
        if verbose:
            print("No additional filtering of loci performed, minimum coverage"
                  f" of included loci is {cell_cov_min[cell_cov_min != 0].min() * 100:.3g}%.", flush=True)
        return

    cutoff = cell_cov_min.quantile(percentile)
    remove_loci_ambig = lengths_df.loc[lengths_df.cell_cov_min.isnull() | (
        lengths_df.cell_cov_min < cutoff), 'idx_genome'].values

    if verbose:
        n = len(lengths_df)
        extra_remove_nloci_ambig = lengths_df.loc[(~lengths_df.cell_cov_min.isnull()) & (
            lengths_df.cell_cov_min != 0) & (lengths_df.cell_cov_min < cutoff), 'idx_genome'].values.size
        print(f"Filtering out an additional {extra_remove_nloci_ambig} ("
              f"{extra_remove_nloci_ambig / n * 100:.3g}%) loci from each homolog", flush=True)
        print(f"A total of {remove_loci_ambig.size} ("
              f"{remove_loci_ambig.size / n * 100:.3g}%) loci are excluded from each homolog", flush=True)

    remove_loci = np.append(remove_loci_ambig, remove_loci_ambig + len(lengths_df))
    return remove_loci


def ambiguate_matrix_df(matrix_df, select_cols=None):
    agg_func = {
        'i.idx': 'count', 'nonmissing': 'sum', 'numerator': 'sum',
        'genomic_dis': 'mean', 'counts': 'sum', 'counts_int': 'sum',
        'dis_mean': lambda x: x.values.tolist(),
        'dis_median': lambda x: x.values.tolist(),
        'mask.nghbr': 'sum', 'mask.sameC-sameH': 'sum'}
    agg_func = {k: v for k, v in agg_func.items() if k in matrix_df.columns}
    if select_cols is not None:
        if isinstance(select_cols, str):
            select_cols = [select_cols]
        agg_func = {k: v for k, v in agg_func.items() if k in select_cols or k == 'i.idx'}

    matrix_df_ambig = matrix_df.copy()

    # Ambig index pair should fall within upper triangular of matrix
    mask_swap = matrix_df['i.idx_ambig'] > matrix_df['j.idx_ambig']
    for col in ('idx_ambig', 'idx_chrom', 'chrom'):
        matrix_df_ambig.loc[mask_swap, f'i.{col}'] = matrix_df.loc[mask_swap, f'j.{col}']
        matrix_df_ambig.loc[mask_swap, f'j.{col}'] = matrix_df.loc[mask_swap, f'i.{col}']
    matrix_df_ambig = matrix_df_ambig[
        matrix_df_ambig['i.idx_ambig'] != matrix_df_ambig['j.idx_ambig']]

    # Ambiguate data for each column
    matrix_df_ambig = matrix_df_ambig.groupby(
        ['i.idx_ambig', 'j.idx_ambig', 'i.idx_chrom', 'j.idx_chrom', 'i.chrom', 'j.chrom']).agg(
        agg_func).reset_index().sort_values(['i.idx_ambig', 'j.idx_ambig']).rename(
        {'mask.sameC-sameH': 'mask.sameC', 'i.idx': 'nbins'}, axis=1, errors='ignore')
    if 'mask.sameC' in matrix_df_ambig.columns:
        matrix_df_ambig['mask.sameC'] = matrix_df_ambig['mask.sameC'].astype(bool)
        matrix_df_ambig['mask.diffC'] = ~matrix_df_ambig['mask.sameC']
    if 'mask.nghbr' in matrix_df_ambig.columns:
        matrix_df_ambig['mask.nghbr'] = matrix_df_ambig['mask.nghbr'].astype(bool)
    if 'dis_mean' in matrix_df_ambig.columns:
        matrix_df_ambig['dis_mean'] = matrix_df_ambig['dis_mean'].apply(np.array)
    if 'dis_median' in matrix_df_ambig.columns:
        matrix_df_ambig['dis_median'] = matrix_df_ambig['dis_median'].apply(np.array)
    matrix_df_ambig.index = list(map(
        tuple, matrix_df_ambig[['i.idx_ambig', 'j.idx_ambig']].values.tolist()))

    # Check that all data is present - 4 UA bins per ambig bin
    missing_ua_data = matrix_df_ambig.nbins != 4
    if missing_ua_data.any():
        raise ValueError(f"Incomplete unambiguous data for {missing_ua_data.sum()}"
                         f" locus pairs:\n{matrix_df_ambig[missing_ua_data]}")

    return matrix_df_ambig


def save_matrices(lengths_df, matrix_df, ambiguity='ua', alpha=None, beta=None,
                  dist_scale_factor=None, counts_col='counts_int', ploidy=2, outdir_counts=None):
    ambiguity = ambiguity.lower()
    if ambiguity not in ('ua', 'pa', 'ambig'):
        raise ValueError(f"{ambiguity=:}, must be 'ua', 'pa', or 'ambig'")
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
    n = lengths.sum()

    # Counts
    beta_counts = beta
    ua_ratio = pa_ratio = 0
    counts_dtype = {'counts_int': int, 'counts': float}[counts_col]
    if ambiguity == 'ua' or ploidy == 1:
        ua_ratio = 1
        counts = np.zeros((n * ploidy, n * ploidy), dtype=counts_dtype)
        counts[matrix_df['i.idx'], matrix_df['j.idx']] = matrix_df[counts_col]
    elif ambiguity == 'pa':
        pa_ratio = 1
        raise NotImplementedError("Implement partially ambig (and adjust beta for PA)")
        if beta is not None:
            beta_counts = None # divide/multiply (?) by 2...?
    else:
        counts = np.zeros((n, n), dtype=counts_dtype)
        matrix_df_ambig = ambiguate_matrix_df(matrix_df, select_cols='counts_int')
        counts[matrix_df_ambig['i.idx_ambig'], matrix_df_ambig['j.idx_ambig']] = matrix_df_ambig
    if outdir_counts is not None:
        write_counts(os.path.join(outdir_counts, f"{ambiguity}_counts.matrix"), counts)

    # Distances
    dis = {}
    for agg_func in ['mean', 'median']:
        dis[agg_func] = np.zeros((n * ploidy, n * ploidy))
        dis[agg_func][matrix_df['i.idx'], matrix_df['j.idx']] = matrix_df[f'dis_{agg_func}']
        dis[agg_func] += dis[agg_func].T  # Fill in lower triangular
        if outdir_counts is not None:
            np.save(os.path.join(outdir_counts, f"struct_true.distances.{agg_func}.npy"), dis[agg_func])

    # Metadata
    if outdir_counts is not None:        
        dataset_info = pd.Series({
            'ploidy': 2, 'ua': ua_ratio, 'pa': pa_ratio, 'lengths': lengths_s.values, 'beta': beta,
            f'beta_{ambiguity}': beta_counts, 'dist_scale_factor': dist_scale_factor})
        if counts_col == 'counts_int':
            dataset_info['nreads'] = counts.sum()
        if isinstance(alpha, (float, int)):
            dataset_info['alpha'] = alpha
        elif isinstance(alpha, (dict, pd.Series)):
            if isinstance(alpha, dict):
                alpha = pd.Series(alpha)
            dataset_info = pd.concat([dataset_info, alpha])
        elif alpha is not None:
            raise ValueError(f"Alpha not understood: {alpha}")
        dataset_info.to_csv(os.path.join(outdir_counts, "dataset_info.txt"), sep="\t", header=False)

    # Plot counts & distances
    if outdir_counts is not None:
        plot_counts_single(
            counts, lengths=lengths, title="Pseudo-counts," + {
                'ua': 'unambiguous', 'ambig': 'ambiguous', 'pa': 'partially ambiguous'}[ambiguity],
            outfile=os.path.join(outdir_counts, "images", f"{ambiguity}_counts.png"), mark_excluded=True)
        for agg_func in ['mean', 'median']:
            plot_distance_matrix(
                dis[agg_func], lengths=lengths, ploidy=2, title=f"'True' distances\n{agg_func} across cells",
                outfile=os.path.join(outdir_counts, "images", f"struct_true.distances.{agg_func}.png"))
    
    return counts, dis, lengths


def get_struct_features_of_sc_true(lengths_df, sc_dis_intramol, dist_scale_factor,
                                   outfile=None, redo=False, verbose=True):
    if os.path.isfile(outfile) and not redo:
        return pd.read_csv(outfile, index_col=0, sep='\t')
    
    if isinstance(sc_dis_intramol, str):
        if verbose:
            print("Loading intra-molecular sc distances...", flush=True)
        sc_dis_intramol = pd.read_csv(
            sc_dis_intramol, sep='\t', header=None, index_col=0,
            converters={0: ast.literal_eval})
        if dist_scale_factor is not None and dist_scale_factor != 1:
            sc_dis_intramol /= dist_scale_factor
    elif dist_scale_factor is not None and dist_scale_factor != 1:
        sc_dis_intramol = sc_dis_intramol / dist_scale_factor

    if isinstance(lengths_df, str):
        lengths_df = pd.read_csv(lengths_df, sep="\t")
    lengths_df.columns = [c.replace('#', '') for c in lengths_df.columns]
    matrix_df = make_matrix_df(lengths_df)

    sc_all = get_other_struct_features(
        matrix_df, dis_df=sc_dis_intramol)
    sc_all = {k: v.tolist() for k, v in sc_all.items()}
    sc_other_feat = [
        get_other_struct_features(matrix_df, dis_df=sc_dis_intramol.mean(axis=1)),
        get_other_struct_features(matrix_df, dis_df=sc_dis_intramol.median(axis=1)),
        sc_all]

    sc_other_feat = pd.DataFrame(
        sc_other_feat, index=('sc_mean', 'sc_med', 'sc_all')).T

    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        sc_other_feat.to_csv(outfile, index=True, header=True, sep='\t')

    return sc_other_feat


def save_sc_data(dir_matrix2d, lengths_df, dist_scale_factor, outdir, redo=False, verbose=True):
    sc_dis_file = glob.glob(os.path.join(dir_matrix2d, '*.distances.per_locus.tsv.gz'))
    sc_dis_intramol_file = glob.glob(os.path.join(dir_matrix2d, '*.distances.intramol.tsv.gz'))
    if len(sc_dis_file) != 1:
        raise ValueError("Couldn't find unique file for single cell distances per locus")
    if len(sc_dis_intramol_file) != 1:
        raise ValueError("Couldn't find unique file for intra-mol single cell distances")
    sc_dis_file = sc_dis_file[0]
    sc_dis_intramol_file = sc_dis_intramol_file[0]

    os.makedirs(outdir, exist_ok=True)
    outfile_per_locus = os.path.join(outdir, 'struct_true.distances.per_locus.tsv')
    outfile_features = os.path.join(outdir, 'struct_true.features.tsv')

    if (not redo) and os.path.isfile(outfile_per_locus) and os.path.isfile(
            outfile_features):
        return outfile_per_locus, outfile_features

    if verbose:
        print("\nSAVING SINGLE-CELL DISTANCE DATA:", flush=True)

    if redo or not os.path.isfile(outfile_per_locus):
        sc_dis = load_sc_dis_per_locus(sc_dis_file, verbose=verbose)
        for col in ['dis_mean', 'dis_med', 'dis']:
            sc_dis[col] /= dist_scale_factor
    
        sc_dis['dis'] = sc_dis.dis.apply(lambda x: x.tolist())
        if verbose:
            print("\tSaving scaled sc distances...", flush=True)
        sc_dis.to_csv(outfile_per_locus, index=True, header=False, sep='\t')

    if redo or not os.path.isfile(outfile_features):
        get_struct_features_of_sc_true(
            lengths_df, sc_dis_intramol=sc_dis_intramol_file,
            dist_scale_factor=dist_scale_factor, outfile=outfile_features, redo=redo, verbose=verbose)

    return outfile_per_locus, outfile_features


def load_and_filter_data(input_file, min_percentile_loci_cov,
                         min_nonmissing_per_phased_locus=0.05, nmol_per_hmlg_ratio=1,
                         spacing=2.5, contact_th=0.75, alpha=None, k=None, d0=None, nu=None,
                         scale_counts_by=None, name=None, verbose=True):
    if name is None:
        name = re.sub(r'(^|.*/)cluster/([^/]+)(/.*|$)', r'\2', os.path.dirname(input_file))
    if nmol_per_hmlg_ratio >= 1000:
        nmol_per_hmlg_ratio = None
    matrices, lengths_df, dir_matrix2d = process_sc_dna_coords(
        input_file=input_file, min_nonmissing_per_phased_locus=min_nonmissing_per_phased_locus,
        nmol_per_hmlg_ratio=nmol_per_hmlg_ratio, spacing=spacing, name=name,
        contact_th=contact_th, alpha=alpha, k=k, d0=d0, nu=nu, scale_counts_by=scale_counts_by,
        verbose=False)

    # Additional filtering of loci by missingness across cells
    if verbose and min_percentile_loci_cov > 0:
        print(f"\nREMOVING LOCI PRESENT IN <{min_percentile_loci_cov * 100:.3g}% OF CELLS:", flush=True)
    remove_loci = filter_loci_by_cell_cov_percentile(
        lengths_df, percentile=min_percentile_loci_cov, verbose=verbose)
    if remove_loci is not None:
        for key in matrices.keys():
            if key in ('dis_mean', 'dis_median'):
                matrices[key][remove_loci, :] = np.nan
                matrices[key][:, remove_loci] = np.nan
            else:
                matrices[key][remove_loci, :] = 0
                matrices[key][:, remove_loci] = 0

    return matrices, lengths_df, dir_matrix2d, name


def save_dataset(input_file, outdir, min_percentile_loci_cov=0.10, nreads=None, ambiguity='ua',
                 infer_alpha_mods='beta_from_intra_only', infer_alpha_dis='mean', num_infer_alpha=10,
                 only_include=None, redo=False, min_nonmissing_per_phased_locus=0.05,
                 nmol_per_hmlg_ratio=1, spacing=2.5, contact_th=None, alpha=None, k=None, d0=None,
                 nu=None, scale_counts_by=None, name=None, verbose=True):

    matrices, lengths_df, dir_matrix2d, name = load_and_filter_data(
        input_file, min_percentile_loci_cov=min_percentile_loci_cov,
        min_nonmissing_per_phased_locus=min_nonmissing_per_phased_locus,
        nmol_per_hmlg_ratio=nmol_per_hmlg_ratio, spacing=spacing,
        contact_th=contact_th, alpha=alpha, k=k, d0=d0, nu=nu, scale_counts_by=scale_counts_by,
        name=name, verbose=verbose)
    matrix_df, dist_scale_factor = prep_matrix_df(lengths_df, matrices=matrices)

    # Get integer counts
    if nreads is not None and isinstance(nreads, (int, float)) and nreads < 0:
        counts_as_int = False
        counts_col_for_matrix = 'counts'
        desc = "non-integer"
    else:
        if verbose:
            print("\nCONVERTING COUNTS TO INTEGERS:", flush=True)
        counts_as_int = True
        counts_col_for_matrix = 'counts_int'
        if nreads is None or isinstance(nreads, str) and nreads.lower() == 'auto':
            nreads = determine_max_nreads(matrix_df, verbose=verbose)
        matrix_df = get_integer_counts(matrix_df, nreads=nreads)
        nreads = matrix_df.counts_int.sum()  # Update with exact number of reads
        desc = f"nreads{nreads:.3g}".replace('e+0', 'e').replace('e+', 'e')
    if alpha is not None:
        desc += f".sc_alpha{alpha:.4g}"
    if not (k is None or d0 is None or nu is None):
        desc += f".logistic_k{k:.4g}_m{d0:.4g}_v{nu:.4g}"

    # Prepare for (optional) filtering of data by bin type (intra-mol / intra-chrom)
    if only_include is not None:
        only_include = only_include.lower().replace('-', '')
        if only_include not in ('intramol', 'intrachr'):
            raise ValueError(f"{only_include=} not understood")
        desc += f".{only_include}"

    # Get output directories
    if outdir is None:
        outdirs = {'ua': None, 'ambig': None, 'pa': None, 'dist': None}
        infer_alpha_outdir = None
    else:
        outdirs = {
            'ua': os.path.join(outdir, "unambig", f"{name}.{desc}"),
            'ambig': os.path.join(outdir, "ambig", f"{name}.{desc}"),
            'pa': os.path.join(outdir, "partial-ambig", f"{name}.{desc}"),
            'dist': os.path.join(outdir, "distances", name)}
        infer_alpha_outdir = os.path.join(outdirs[ambiguity], "struct_true.infer_alpha_from_dis")
        os.makedirs(infer_alpha_outdir, exist_ok=True)
        os.makedirs(outdirs['dist'], exist_ok=True)

    # Save single-cell distances and structural features
    if outdir is not None:
        outfile_per_locus, outfile_features = save_sc_data(
            dir_matrix2d=dir_matrix2d, lengths_df=lengths_df, dist_scale_factor=dist_scale_factor,
            outdir=outdirs['dist'], redo=redo, verbose=verbose)
        symlink(source=outfile_per_locus, dest=os.path.join(outdirs[ambiguity], os.path.basename(outfile_per_locus)))
        symlink(source=outfile_features, dest=os.path.join(outdirs[ambiguity], os.path.basename(outfile_features)))

    # Infer alpha
    alpha_intra, alpha_inter, beta, alpha_inf_results = infer_alpha_float_vs_int(
        matrix_df, ambiguity=ambiguity, dis_agg_func=infer_alpha_dis, infer_alpha_mask=None,
        infer_alpha_mods=infer_alpha_mods, plot=True, outdir=infer_alpha_outdir,
        num_infer=num_infer_alpha, verbose=verbose)

    # Optionally filter for intra-mol or intra-chrom data only
    if only_include == 'intramol':
        matrix_df = matrix_df[matrix_df['mask.sameC-sameH']]
    elif only_include == 'intrachr':
        matrix_df = matrix_df[matrix_df['mask.sameC-sameH'] | matrix_df['mask.sameC-diffH']]

    # Save counts
    counts, dis, lengths = save_matrices(
        lengths_df, matrix_df=matrix_df, ambiguity=ambiguity,
        alpha={'alpha': alpha_intra, 'alpha_intra': alpha_intra, 'alpha_inter': alpha_inter},
        beta=beta, dist_scale_factor=dist_scale_factor, counts_col=counts_col_for_matrix,
        outdir_counts=outdirs[ambiguity])

    
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)

    # Preparing dataset
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--min_percentile_loci_cov", default=0.10, type=float)
    parser.add_argument("--nreads", default=None, type=float)
    parser.add_argument("--ambiguity", default='ua', type=str)
    parser.add_argument("--infer_alpha_mods", nargs='+', type=str)
    parser.add_argument("--infer_alpha_dis", default='mean', type=str, choices=['mean', 'median'])
    parser.add_argument("--num_infer_alpha", default=10, type=int)
    parser.add_argument("--only_include", type=str, choices=['intramol', 'intrachr'])
    parser.add_argument('--redo', default=False, action='store_true')

    # Making consensus pseudo-counts from sc distances
    parser.add_argument("--spacing", default=2.5, type=float)
    parser.add_argument("--min_nonmissing_per_phased_locus", default=0.05, type=float)
    parser.add_argument("--nmol_per_hmlg_ratio", default=1, type=float)
    parser.add_argument("--name", type=str)
    parser.add_argument("--contact_th", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--d0", type=float)
    parser.add_argument("--nu", type=float)
    parser.add_argument("--scale_counts_by", type=float, default=1)

    # Verbosity
    parser.add_argument('--verbose', default=True, action='store_true')
    parser.add_argument('--silent', dest='verbose', default=True, action='store_false')
    args = parser.parse_args()

    name = args.name
    if name is None:
        name = re.sub(r'(^|.*/)cluster(?:\.LINK){0,1}/([^/]+)(/.*|$)', r'\2', os.path.dirname(args.data))

    nmol_per_hmlg_ratio = args.nmol_per_hmlg_ratio
    if args.nmol_per_hmlg_ratio >= 1000:
        nmol_per_hmlg_ratio = None

    save_dataset(
        input_file=args.data, outdir=args.outdir, min_percentile_loci_cov=args.min_percentile_loci_cov,
        nreads=args.nreads, ambiguity=args.ambiguity, infer_alpha_mods=args.infer_alpha_mods,
        infer_alpha_dis=args.infer_alpha_dis, num_infer_alpha=args.num_infer_alpha,
        only_include=args.only_include, redo=args.redo,
        min_nonmissing_per_phased_locus=args.min_nonmissing_per_phased_locus,
        nmol_per_hmlg_ratio=nmol_per_hmlg_ratio,  spacing=args.spacing,
        contact_th=args.contact_th, alpha=args.alpha, k=args.k, d0=args.d0, nu=args.nu,
        scale_counts_by=args.scale_counts_by, name=name, verbose=args.verbose)


if __name__ == "__main__":
    main()