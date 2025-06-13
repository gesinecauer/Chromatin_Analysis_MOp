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
from topsy.analysis.compare_distances import load_sc_dis_per_locus, scale_sc_distances
from topsy.analysis.utils import get_nghbr_dis_var
from estimate_alpha import estimate_alphas_from_true_dis


def prep_matrix_df(lengths_df, matrices, scale_dis_by='mean', nonmissing_bin_percentile=None):
    if set(list(matrices.keys())) != {'counts', 'nonmissing', 'dis_mean', 'dis_median'}:
        raise ValueError("Inputted matrices must contain: 'counts', 'nonmissing', 'dis_mean', 'dis_median'")
    
    if scale_dis_by is not None:
        scale_dis_by = scale_dis_by.lower()
        if scale_dis_by not in ('mean', 'median'):
            raise ValueError("'scale_dis_by' must be 'mean', 'median', or None.")

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


# def get_nghbr_bins(matrix, lengths):
#     mask_intermol_nghbr = np.tile(lengths, 2).cumsum()[:-1] - 1
#     nghbr_bins = np.diagonal(matrix, offset=1).copy().astype(float)
#     nghbr_bins[mask_intermol_nghbr] = np.nan
#     return nghbr_bins


# def get_beta_ua(counts, lengths):
#     # Get beta such that distance between neighbor beads is 1
#     beta_ua = np.nanmean(get_nghbr_bins(counts, lengths=lengths))
#     return beta_ua


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

    # Counts
    beta_counts = beta
    ua_ratio = pa_ratio = 0
    counts_dtype = {'counts_int': int, 'counts': float}
    if ambiguity == 'ua' or ploidy == 1:
        ua_ratio = 1
        counts = np.zeros((lengths.sum() * ploidy, lengths.sum() * ploidy), dtype=counts_dtype)
        counts[matrix_df['i.idx'], matrix_df['j.idx']] = matrix_df[counts_col]
    elif ambiguity == 'pa':
        pa_ratio = 1
        raise NotImplementedError("Implement partially ambig (and adjust beta for PA)")
        if beta is not None:
            beta_counts = None # divide/multiply (?) by 2...?
    else:
        counts = np.zeros((lengths.sum(), lengths.sum()), dtype=counts_dtype)
        matrix_df_ambig = matrix_df.groupby(
            ['i.idx_ambig', 'j.idx_ambig'])[counts_col].sum().reset_index()
        counts[matrix_df_ambig['i.idx_ambig'], matrix_df_ambig['j.idx_ambig']] = matrix_df_ambig
    if outdir_counts is not None:
        write_counts(os.path.join(outdir_counts, f"{ambiguity}_counts.matrix"), counts)

    # Distances
    dis = {}
    for agg_func in ['mean', 'median']:
        dis[agg_func] = np.zeros((lengths.sum() * ploidy, lengths.sum() * ploidy))
        dis[agg_func][matrix_df['i.idx'], matrix_df['j.idx']] = matrix_df[f'dis_{agg_func}']
        if outdir_counts is not None:
            np.save(os.path.join(outdir_counts, f"distances_true.{agg_func}.npy"), dis[agg_func])

    # Metadata
    if outdir_counts is not None:        
        dataset_info = pd.Series({
            'ploidy': 2, 'nreads': counts_int.sum(), 'ua': ua_ratio, 'pa': pa_ratio, 'lengths': lengths_s.values,
            'beta': beta, f'beta_{ambiguity}': beta_counts, 'alpha': alpha, 'dist_scale_factor': dist_scale_factor})
        dataset_info.to_csv(os.path.join(outdir_counts, "dataset_info.txt"), sep="\t", header=False)

    # Plot counts & distances
    if outdir_counts is not None:
        plot_counts_single(
            counts_int, lengths=lengths, title="Pseudo-counts, unambiguous",
            outfile=os.path.join(outdir_counts, "images", "ua_counts.png"), mark_excluded=True)
        for agg_func in ['mean', 'median']:
            plot_distance_matrix(
                dis[agg_func], lengths=lengths, ploidy=2, title=f"'True' distances\n{agg_func} across cells",
                outfile=os.path.join(outdir_counts, "images", f"distances_true.{agg_func}.png"))
    
    return counts_int, dis_scaled, lengths, scale_factors["nghbr_dis_mean.sc_mean"]


def get_struct_features_of_sc_true(lengths_df, sc_dis_intramol, dist_scale_factor,
                                   outfile=None, redo=False):
    if os.path.exist(outfile) and not redo:
        return pd.read_csv(outfile, index_col=0, sep='\t')
    
    if isinstance(sc_dis_intramol, str):
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


def save_sc_data(dir_matrix2d, lengths_df, dist_scale_factor, outdir, redo=False):
    sc_dis_file = glob.glob(os.path.join(dir_matrix2d, '*.distances.per_locus.tsv.gz'))
    sc_dis_intramol_file = glob.glob(os.path.join(dir_matrix2d, '*.distances.intramol.tsv.gz'))
    if len(sc_dis_file) != 1:
        raise ValueError("Couldn't find unique file for single cell distances per locus")
    if len(sc_dis_intramol_file) != 1:
        raise ValueError("Couldn't find unique file for intra-mol single cell distances")
    sc_dis_file = sc_dis_file[0]
    sc_dis_intramol_file = sc_dis_intramol_file[0]
    
    os.makedirs(outdir, exist_ok=True)
    outfile_per_locus = os.path.join(outdir, 'distances_true.per_locus.tsv')
    outfile_features = os.path.join(outdir, 'features.struct_true.tsv')
    
    if redo or not os.path.exist(outfile_per_locus):
        sc_dis = load_sc_dis_per_locus(sc_dis_file, scale=False, verbose=verbose)
        for col in ['dis_mean', 'dis_med', 'dis']:
            sc_dis[col] /= dist_scale_factor
    
        sc_dis['dis'] = sc_dis.dis.apply(lambda x: x.tolist())
        sc_dis.to_csv(outfile, index=True, header=False, sep='\t')

    if redo or not os.path.exist(outfile_features):
        get_struct_features_of_sc_true(
            lengths_df, sc_dis_intramol=sc_dis_intramol_file,
            dist_scale_factor=dist_scale_factor, outfile=outfile_features, redo=redo)


def load_and_filter_data(input_file, min_percentile_loci_cov,
                         min_nonmissing_per_phased_locus=0.05, nmol_per_hmlg_ratio=1,
                         spacing=2.5, contact_th=0.75, name=None, verbose=True):
    if name is None:
        name = re.sub(r'(^|.*/)cluster/([^/]+)(/.*|$)', r'\2', os.path.dirname(input_file))
    if nmol_per_hmlg_ratio >= 1000:
        nmol_per_hmlg_ratio = None
    matrices, lengths_df, dir_matrix2d = process_sc_dna_coords(
        input_file=input_file, min_nonmissing_per_phased_locus=min_nonmissing_per_phased_locus,
        nmol_per_hmlg_ratio=nmol_per_hmlg_ratio, spacing=spacing, name=name,
        contact_th=contact_th, verbose=False)

    # Additional filtering of loci
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


def save_dataset(input_file, nreads, outdir, min_percentile_loci_cov,
                 min_nonmissing_per_phased_locus=0.05, nmol_per_hmlg_ratio=1,
                 spacing=2.5, contact_th=0.75, redo=False, name=None, verbose=True):

    matrices, lengths_df, dir_matrix2d, name = load_and_filter_data(
        input_file, min_percentile_loci_cov=min_percentile_loci_cov,
        min_nonmissing_per_phased_locus=min_nonmissing_per_phased_locus,
        nmol_per_hmlg_ratio=nmol_per_hmlg_ratio, spacing=spacing,
        contact_th=contact_th, name=name, verbose=verbose)

    matrix_df, dist_scale_factor = prep_matrix_df(lengths_df, matrices=matrices)

    if nreads is None or isinstance(nreads, str) and nreads.lower() == 'auto':
        nreads = determine_max_nreads(matrix_df, verbose=verbose)
    matrix_df = get_integer_counts(matrix_df, nreads=nreads)

    outdir_ua_counts = os.path.join(
        outdir, "counts", "unambig", f"{name}.nreads{nreads:.3g}".replace('e+0', 'e').replace('e+', 'e'))

    if verbose:
        print("\nALPHA INFERENCE WITH 'TRUE' SINGLE-CELL DISTANCES:\n", flush=True)
    alpha_floatcounts, beta_floatcounts = estimate_alphas_from_true_dis(
        matrix_df, use_poisson=False, integer_counts=False, dis_agg_func='mean', infer_alpha_mask=None,
        infer_alpha_mods='beta_from_intra_only', plot=True,
        outdir_fig=os.path.join(outdir_ua_counts, 'images'), verbose=verbose)
    if verbose:
        print(flush=True)
    alpha_intcounts, beta_intcounts = estimate_alphas_from_true_dis(
        matrix_df, use_poisson=True, integer_counts=True, dis_agg_func='mean', infer_alpha_mask=None,
        infer_alpha_mods='beta_from_intra_only', plot=True,
        outdir_fig=os.path.join(outdir_ua_counts, 'images'), verbose=verbose)

    # _, _, _, dist_scale_factor = save_matrices(
    #     lengths_df, counts=matrices['counts'], matrices=matrices, nreads=nreads,
    #     outdir_counts=outdir_ua_counts)

    save_sc_data(
        dir_matrix2d=dir_matrix2d, lengths_df=lengths_df, dist_scale_factor=dist_scale_factor,
        outdir=outdir_ua_counts, redo=False)


    
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--spacing", default=2.5, type=float)
    parser.add_argument("--min_nonmissing_per_phased_locus", default=0.05, type=float)
    parser.add_argument("--nmol_per_hmlg_ratio", default=1, type=float)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--contact_th", default=0.75, type=float)
    # parser.add_argument("--chrom", type=str, nargs='+')
    parser.add_argument('--verbose', default=True, action='store_true')
    parser.add_argument('--silent', dest='verbose', default=True, action='store_false')
    args = parser.parse_args()

    name = args.name
    if name is None:
        name = re.sub(r'(^|.*/)cluster/([^/]+)(/.*|$)', r'\2', os.path.dirname(args.data))

    nmol_per_hmlg_ratio = args.nmol_per_hmlg_ratio
    if args.nmol_per_hmlg_ratio >= 1000:
        nmol_per_hmlg_ratio = None

    save_dataset(
        input_file=args.data, min_nonmissing_per_phased_locus=args.min_nonmissing_per_phased_locus,
        nmol_per_hmlg_ratio=nmol_per_hmlg_ratio, spacing=args.spacing, outdir=args.outdir, name=name,
        contact_th=args.contact_th, verbose=args.verbose)


if __name__ == "__main__":
    main()