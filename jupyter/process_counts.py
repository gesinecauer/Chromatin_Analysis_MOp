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

def get_nghbr_bins(matrix, lengths):
    mask_intermol_nghbr = np.tile(lengths, 2).cumsum()[:-1] - 1
    nghbr_bins = np.diagonal(matrix, offset=1).copy().astype(float)
    nghbr_bins[mask_intermol_nghbr] = np.nan
    return nghbr_bins


def get_beta_ua(counts, lengths):
    # Get beta such that distance between neighbor beads is 1
    beta_ua = np.nanmean(get_nghbr_bins(counts, lengths=lengths))
    return beta_ua


def get_unambig_counts(lengths_df, counts, matrices, nreads, outdir_counts=None, infer_alpha=False):
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
        scale_factors[f"nghbr_dis_mean.sc_{agg_func}"] = nghbr_dis_mean
        scale_factors[f"nghbr_dis_med.sc_{agg_func}"] = np.nanmedian(nghbr_dis)
        dis_scaled[agg_func] = matrices[f'dis_{agg_func}'] / nghbr_dis_mean
        if outdir_counts is not None:
            np.save(os.path.join(outdir_counts, f"distances_true.{agg_func}.npy"), dis_scaled[agg_func])

    # # Get beta such that distance between neighbor beads is 1
    # est_beta_ua = get_beta_ua(counts=counts_int, lengths=lengths_s.values)

    if infer_alpha:
        pass

    # Metadata
    if outdir_counts is not None:        
        dataset_info = pd.Series({
            'ploidy': 2, 'nreads': counts_int.sum(), 'ua': 1, 'pa': 0, 'lengths': lengths_s.values,
            'beta': None, 'beta_ua': None, 'alpha': None})
        for key, value in scale_factors.items():
            dataset_info[key] = value
        dataset_info.to_csv(os.path.join(outdir_counts, "dataset_info.txt"), sep="\t", header=False)

    # Plot counts & distances
    if outdir_counts is not None:
        plot_counts_single(
            counts_int, lengths=lengths, title="Pseudo-counts, unambiguous",
            outfile=os.path.join(outdir_counts, "images", "ua_counts.png"), mark_excluded=True)
        for agg_func in ['mean', 'median']:
            plot_distance_matrix(
                dis_scaled[agg_func], lengths=lengths, ploidy=2, title=f"'True' distances\n{agg_func} across cells",
                outfile=os.path.join(outdir_counts, "images", f"distances_true.{agg_func}.png"))
    
    return counts_int, dis_scaled, lengths, scale_factors["nghbr_dis_mean.sc_mean"]


def get_struct_features_of_sc_true(lengths_df, sc_dis_intramol, scale_factor,
                                   outfile=None, redo=False):
    if os.path.exist(outfile) and not redo:
        return pd.read_csv(outfile, index_col=0, sep='\t')
    
    if isinstance(sc_dis_intramol, str):
        sc_dis_intramol = pd.read_csv(
            sc_dis_intramol, sep='\t', header=None, index_col=0,
            converters={0: ast.literal_eval})
        if scale_factor is not None and scale_factor != 1:
            sc_dis_intramol /= scale_factor
    elif scale_factor is not None and scale_factor != 1:
        sc_dis_intramol = sc_dis_intramol / scale_factor

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


def save_sc_data(dir_matrix2d, lengths_df, scale_factor, outdir, redo=False):
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
            sc_dis[col] /= scale_factor
    
        sc_dis['dis'] = sc_dis.dis.apply(lambda x: x.tolist())
        sc_dis.to_csv(outfile, index=True, header=False, sep='\t')

    if redo or not os.path.exist(outfile_features):
        get_struct_features_of_sc_true(
            lengths_df, sc_dis_intramol=sc_dis_intramol_file,
            scale_factor=scale_factor, outfile=outfile_features, redo=redo)


def save_dataset(input_file, nreads, outdir, min_nonmissing_per_phased_locus=0.05, nmol_per_hmlg_ratio=1,
                 spacing=2.5, contact_th=0.5, redo=False, name=None, verbose=False):

    if name is None:
        name = re.sub(r'(^|.*/)cluster/([^/]+)(/.*|$)', r'\2', os.path.dirname(input_file))
    if nmol_per_hmlg_ratio >= 1000:
        nmol_per_hmlg_ratio = None
    matrices, lengths_df, dir_matrix2d = process_sc_dna_coords(
        input_file=input_file, min_nonmissing_per_phased_locus=min_nonmissing_per_phased_locus,
        nmol_per_hmlg_ratio=nmol_per_hmlg_ratio, spacing=spacing, name=name,
        contact_th=contact_th, outdir=os.path.dirname(input_file), verbose=verbose)

    outdir_ua_counts = os.path.join(
        outdir, "counts", "unambig", f"{name}.nreads{nreads:.3g}".replace('e+0', 'e').replace('e+', 'e'))
    _, _, _, scale_factor = get_unambig_counts(
        lengths_df, counts=matrices['counts'], matrices=matrices, nreads=nreads,
        outdir_counts=outdir_ua_counts)
    save_sc_data(
        dir_matrix2d=dir_matrix2d, lengths_df=lengths_df, scale_factor=scale_factor,
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
    parser.add_argument("--contact_th", default=0.5, type=float)
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