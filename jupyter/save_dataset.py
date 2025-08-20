import numpy as np
import pandas as pd
import os
import re
import ast
import glob
from scipy import sparse
from struct_to_dis import process_sc_dna_coords
from iced.io import write_counts, write_lengths
from topsy.plot.plot_distances import plot_distance_matrix
from topsy.plot.plot_counts import plot_counts_single
from topsy.analysis.compare_distances import make_matrix_df, get_other_struct_features
from topsy.analysis.compare_distances import load_sc_dis_per_locus
from topsy.analysis.utils import get_nghbr_dis_var
from topsy.utils.misc import symlink


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

    # Remove loci that are entirely missing in sc dataset
    n = len(lengths_df)
    excluded_loci = np.where((matrices['nonmissing'] == 0).all(axis=0))[0]
    matrix_df = matrix_df[~matrix_df['i.idx'].isin(excluded_loci)]

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


def save_matrices(lengths_df, matrix_df, alpha=None, beta=None,
                  dist_scale_factor=None, counts_col='counts_int', ploidy=2, outdir_counts=None):
    if ploidy == 1:
        ambiguity = 'ua'
    else:
        ambiguity = 'ambig'
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
    counts_dtype = {'counts_int': int, 'counts': float}[counts_col]
    counts = np.zeros((n, n), dtype=counts_dtype)
    counts[matrix_df['i.idx'], matrix_df['j.idx']] = matrix_df[counts_col]
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
            'ploidy': 2, 'ua': 1 if ploidy == 1 else 0, 'pa': 0,
            'lengths': lengths_s.values, 'dist_scale_factor': dist_scale_factor})
        if counts_col == 'counts_int':
            dataset_info['nreads'] = counts.sum()
        dataset_info.to_csv(os.path.join(outdir_counts, "dataset_info.txt"), sep="\t", header=False)

    # Plot counts & distances
    if outdir_counts is not None:
        plot_counts_single(
            counts, lengths=lengths, title="snm3c-seq counts",
            outfile=os.path.join(outdir_counts, "images", f"{ambiguity}_counts.png"), mark_excluded=True)
        for agg_func in ['mean', 'median']:
            plot_distance_matrix(
                dis[agg_func], lengths=lengths, ploidy=2, title=f"DNA-MERFISH distances\n{agg_func} across cells",
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


def save_dataset(input_file, outdir, nreads=None, redo=False, spacing=2.5, name=None, verbose=True):

    if name is None:
        name = re.sub(r'(^|.*/)cluster/([^/]+)(/.*|$)', r'\2', os.path.dirname(input_file))

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

    # Get output directories
    if name is None:
        name = re.sub(r'(^|.*/)cluster/([^/]+)(/.*|$)', r'\2', os.path.dirname(input_file))
    outdir_counts = outdir_dist = None
    if outdir is not None:
        outdir_counts = os.path.join(outdir, "snm3c-seq", f"{name}.{desc}")
        outdir_dist = os.path.join(outdir, "distances", name)
        os.makedirs(outdir_dist, exist_ok=True)

    # Save single-cell distances and structural features
    if outdir is not None:
        outfile_per_locus, outfile_features = save_sc_data(
            dir_matrix2d=dir_matrix2d, lengths_df=lengths_df, dist_scale_factor=dist_scale_factor,
            outdir=outdir_dist, redo=redo, verbose=verbose)
        symlink(source=outfile_per_locus, dest=os.path.join(outdir_counts, os.path.basename(outfile_per_locus)))
        symlink(source=outfile_features, dest=os.path.join(outdir_counts, os.path.basename(outfile_features)))

    # Save counts
    counts, dis, lengths = save_matrices(
        lengths_df, matrix_df=matrix_df, dist_scale_factor=dist_scale_factor,
        counts_col=counts_col_for_matrix, outdir_counts=outdir_counts)

    
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)

    # Preparing dataset
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--nreads", default=None, type=float)
    parser.add_argument('--redo', default=False, action='store_true')

    # Making consensus pseudo-counts from sc distances
    parser.add_argument("--spacing", default=2.5, type=float)
    parser.add_argument("--name", type=str)

    # Verbosity
    parser.add_argument('--verbose', default=True, action='store_true')
    parser.add_argument('--silent', dest='verbose', default=True, action='store_false')
    args = parser.parse_args()

    name = args.name
    if name is None:
        name = re.sub(r'(^|.*/)cluster(?:\.LINK){0,1}/([^/]+)(/.*|$)', r'\2', os.path.dirname(args.data))

    save_dataset(
        input_file=args.data, outdir=args.outdir, nreads=args.nreads, redo=args.redo,
        spacing=args.spacing, name=name, verbose=args.verbose)


if __name__ == "__main__":
    main()