import pandas as pd
import glob
import os
import ast
import numpy as np
import re
from functools import partial

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='', category=UserWarning)
    warnings.filterwarnings('ignore', message='', category=FutureWarning)
    from topsy.analysis.compare_distances import make_matrix_df


def load(dir_matrix2d, mcool_file, lengths_file=None, resolution=2.5, normalize_snm3c=True, verbose=True):
    # Load snm3c-seq data via cooler
    snm3c, clr_bins, lengths_clr = load_snm3c(
        mcool_file, resolution=resolution, normalize=normalize_snm3c, verbose=verbose)

    # Load info on missingness across cells
    nonmissing_file = glob.glob(os.path.join(dir_matrix2d, '*num_nonmissing.npy'))
    if len(nonmissing_file) != 1:
        raise ValueError("Couldn't find unique file for nonmissingness single cells")
    nonmissing_file = nonmissing_file[0]
    nonmissing_matrix = np.load(nonmissing_file).astype(int)

    # Get single-cell distances: intra-molecular
    sc_dis_intramol_file = glob.glob(os.path.join(dir_matrix2d, '*distances.intramol.tsv.gz'))
    if len(sc_dis_intramol_file) != 1:
        raise ValueError("Couldn't find unique file for intra-mol single cell distances")
    sc_dis_intramol_file = sc_dis_intramol_file[0]
    sc_dis = pd.read_csv(
        sc_dis_intramol_file, sep='\t', header=None, index_col=0,
        converters={0: ast.literal_eval})
    sc_dis.index.name = None

    # Get single-cell distances: intra-chromosomal, inter-homolog
    sc_dis_diffH_file = glob.glob(os.path.join(dir_matrix2d, '*.distances.sameC-diffH.tsv.gz'))
    if len(sc_dis_diffH_file) != 1:
        raise ValueError("Couldn't find unique file for intra-chrom, inter-hmlg single cell distances")
    sc_dis_diffH_file = sc_dis_diffH_file[0]
    sc_dis_diffH = pd.read_csv(
        sc_dis_diffH_file, sep='\t', header=None, index_col=0,
        converters={0: ast.literal_eval})
    sc_dis_diffH.index.name = None

    # Single-cell distances: combine intra-chromosomal data
    sc_dis = pd.concat([sc_dis, sc_dis_diffH])

    # Setup chromosome lengths data
    if lengths_file is None:
        lengths_file = os.path.join(os.path.dirname(dir_matrix2d), 'counts.bed')
    lengths_df = pd.read_csv(lengths_file, sep='\t')
    lengths_df.columns = [c.strip('#') for c in lengths_df.columns]
    for col in ['start', 'end', 'mid']:  # Round to nearest 10,000bp
        lengths_df[col] = lengths_df[col].round(-4)
    lengths_s = lengths_df.groupby('chrom').size().sort_values(
        ascending=False)

    # Compare to and merge with chromosome lengths/bins data from snm3c
    clr_bins = clr_bins.reset_index().rename({'index': 'idx_clr'}, axis=1)
    lengths_compare = pd.concat([
        lengths_s.to_frame().rename({0: 'merfish'}, axis=1),
        lengths_clr.to_frame().rename({0: 'snm3c'}, axis=1)], axis=1)
    lengths_compare['difference'] = lengths_compare.snm3c - lengths_compare.merfish
    assert (lengths_compare.difference >= 0).all()
    lengths_df = lengths_df.merge(clr_bins[['chrom', 'mid', 'idx_clr', 'snm3c_isnan']], on=['chrom', 'mid'], how='left')
    if verbose:
        print(f"{lengths_df.idx_clr.isnull().sum()} genomic loci are present in MERFISH but not snm3c", flush=True)
        print(f"{lengths_df.snm3c_isnan.sum()} genomic loci are masked out from snm3c", flush=True)
    lengths_df.index = lengths_df.idx_genome
    lengths_df.index.name = None


    # Setup matrix-related data for intra-chromosomal (with inter- and intra-homolog)
    matrix_df = make_matrix_df(lengths_df, matrix_dict={'nonmissing': nonmissing_matrix})
    matrix_df['nonmissing'] = matrix_df['nonmissing'].astype(int)
    matrix_df = matrix_df[matrix_df['mask.sameC-sameH'] | matrix_df['mask.sameC-diffH']].drop(
        ['mask.sameC-sameH', 'mask.sameC-diffH', 'mask.diffC-sameH', 'mask.diffC-diffH', 'j.chrom'], axis=1).rename(
        {'i.chrom': 'chrom'}, axis=1)
    matrix_df['genomic_dis_ambig'] = (matrix_df['i.idx_ambig'] - matrix_df['j.idx_ambig']).abs()

    # Add snm3c-seq data to matrix_df
    matrix_df['i.idx_clr'] = lengths_df.loc[matrix_df['i.idx_ambig'], 'idx_clr'].values.ravel()
    matrix_df['j.idx_clr'] = lengths_df.loc[matrix_df['j.idx_ambig'], 'idx_clr'].values
    mask = matrix_df['i.idx_clr'].notnull() & matrix_df['j.idx_clr'].notnull()
    matrix_df.loc[mask, 'snm3c'] = snm3c[matrix_df.loc[mask, 'i.idx_clr'].astype(int), matrix_df.loc[mask, 'j.idx_clr'].astype(int)]
    matrix_df.loc[matrix_df['i.idx_ambig'] == matrix_df['j.idx_ambig'], 'snm3c'] = np.nan
    
    # Filter for locus pairs present in single-cell distances
    matrix_df = matrix_df.loc[matrix_df.index.isin(sc_dis.index)]

    return matrix_df, sc_dis, lengths_df



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir_matrix2d", type=str)
    parser.add_argument("--mcool_file", type=str)
    parser.add_argument("--resolution", default=2.5, type=float)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--name", type=str)

    parser.add_argument("--normalize", default=True, action='store_true')
    parser.add_argument("--dont-normalize", default=True, action='store_false')

    parser.add_argument('--intramol_only', default=False, action='store_true')
    parser.add_argument('--infer_nu', default=False, action='store_true')
    parser.add_argument('--infer_q', default=False, action='store_true')
    parser.add_argument('--infer_a', default=False, action='store_true')
    parser.add_argument("--obj_type", type=str, default='rmse')
    parser.add_argument("--weights_exponent", default=None, type=int)
    parser.add_argument("--constraint_opt", default=0, type=float)
    parser.add_argument("--constraint_penalty", default=0, type=float)
    parser.add_argument("--constraint_min_n", type=int)
    parser.add_argument("--scale_snm3c_by", default=1, type=float)
    

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--init", default=None, nargs='+')

    parser.add_argument("--max_iter", default=1e20, type=float)
    parser.add_argument("--factr", default=1e7, type=float)
    parser.add_argument("--maxls", default=20, type=int)
    parser.add_argument("--pgtol", default=1e-05, type=float)
    parser.add_argument('--no-jit', dest='jitted', default=True, action='store_false')

    # parser.add_argument('--verbose', default=True, action='store_true')
    # parser.add_argument('--silent', dest='verbose', default=True, action='store_false')
    args = parser.parse_args()

    if args.dir_matrix2d.lower() in ('test', 'load'):
        return

    init = args.init
    if init is not None:
        if len(init) == 1 and ',' in init[0] or ' ' in init[0]:
            init = re.sub(r',,+', ',', init[0].replace(' ', ',').strip(',"\'')).split(',')
        init = [float(x) for x in init]

    load_and_infer(
        args.dir_matrix2d, mcool_file=args.mcool_file, resolution=args.resolution,
        normalize_snm3c=args.normalize, outdir=args.outdir, name=args.name, intramol_only=args.intramol_only,
        infer_nu=args.infer_nu, infer_q=args.infer_q, infer_a=args.infer_a,
        obj_type=args.obj_type, weights_exponent=args.weights_exponent,
        constraint_opt=args.constraint_opt, constraint_penalty=args.constraint_penalty,
        constraint_min_n=args.constraint_min_n, scale_snm3c_by=args.scale_snm3c_by, init=init,
        seed=args.seed, max_iter=args.max_iter, factr=args.factr, maxls=args.maxls,
        pgtol=args.pgtol, jitted=args.jitted)


if __name__ == "__main__":
    main()
