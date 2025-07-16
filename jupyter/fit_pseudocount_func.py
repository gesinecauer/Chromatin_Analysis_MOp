import pandas as pd
import glob
import os
import ast
import numpy as np
import re
from functools import partial

import cooler

# import matplotlib.pyplot as plt
# from matplotlib import colors
# plt.style.use('seaborn-v0_8-paper')
# import seaborn as sns
# sns.set_theme('paper', style='white')

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='', category=UserWarning)
    warnings.filterwarnings('ignore', message='', category=FutureWarning)
    from topsy.analysis.compare_distances import make_matrix_df
    from pastis.optimization.utils_poisson import _dict_is_equal, _dict_to_hash, _setup_jax

_setup_jax(traceback=False, debug_nan_inf=False)
import jax.numpy as jnp
from jax import grad, jit
from jax.nn import relu
from scipy import optimize


class InferArgs(object):
    def __init__(self, snm3c, sc_dis_arr):
        self.snm3c = snm3c
        self.dis = sc_dis_arr

    def __eq__(self, other):
        if type(other) is type(self):
            if not _dict_is_equal(self.__dict__, other.__dict__):
                return False
            return True
        return NotImplemented

    def __hash__(self):
        return _dict_to_hash(self.__dict__)


def get_unambig_idx_per_ambig(df):
    s = ('h' + df['i.hmlg'].astype(str) + '_h' + df['j.hmlg'].astype(
        str)).to_frame().reset_index().set_index(0)['index']
    s['snm3c'] = np.nan
    if df.snm3c.notnull().any():
        s['snm3c'] = df.snm3c.mean()
    return s


def ambiguate_matrix_df_for_inference(matrix_df):
    grp_cols = ['i.idx_ambig', 'j.idx_ambig', 'chrom', 'i.idx_chrom', 'j.idx_chrom',
                'i.idx_clr', 'j.idx_clr']
    matrix_df_ambig = matrix_df.copy()
    mask_swap = matrix_df['i.idx_ambig'] > matrix_df['j.idx_ambig']
    if mask_swap.any():  # Ambig index pair should fall in upper triangular
        for col in ('idx_ambig', 'idx_chrom', 'hmlg', 'idx_clr'):
            matrix_df_ambig.loc[mask_swap, f'i.{col}'] = matrix_df.loc[mask_swap, f'j.{col}']
            matrix_df_ambig.loc[mask_swap, f'j.{col}'] = matrix_df.loc[mask_swap, f'i.{col}']
    matrix_df_ambig = matrix_df_ambig[  # Remove main diagonal
        matrix_df_ambig['i.idx_ambig'] != matrix_df_ambig['j.idx_ambig']]
    matrix_df_ambig = matrix_df_ambig[grp_cols + ['i.hmlg', 'j.hmlg', 'snm3c']].groupby(
        grp_cols).apply(get_unambig_idx_per_ambig, include_groups=False).reset_index(
        level=np.arange(len(grp_cols)).tolist())
    matrix_df_ambig = matrix_df_ambig[matrix_df_ambig.snm3c.notnull()]

    snm3c = np.asarray(matrix_df_ambig.snm3c.values, order='C')

    quadrants = ['h1_h1', 'h2_h2', 'h1_h2', 'h2_h1']
    dis_idx = np.concatenate([matrix_df_ambig[q].values for q in quadrants])
    dis_idx = np.asarray(dis_idx, order='C')

    return matrix_df_ambig, snm3c, dis_idx


def load(dir_matrix2d, mcool_file, lengths_file, resolution=2.5e6, verbose=True):
    # Get single-cell distances: intra-molecular
    sc_dis_intramol_file = glob.glob(os.path.join(dir_matrix2d, '*.distances.intramol.tsv.gz'))
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
    
    # Load snm3c-seq data via cooler
    resolution = int(resolution)
    uri = f'{mcool_file}::/resolutions/{resolution:d}'
    clr = cooler.Cooler(uri)
    snm3c = np.triu(clr.matrix(balance=True if 'Raw.' in mcool_file else False)[:, :], 1)
    
    clr_bins = clr.bins()[:].reset_index().rename({'index': 'idx_clr'}, axis=1)
    clr_bins['mid'] = clr_bins[['start', 'end']].mean(axis=1).round().astype(int)
    assert len(clr_bins) == snm3c.shape[0]
    print(snm3c.shape)
    
    nbins_clr = clr_bins.groupby('chrom', observed=True).size().sort_values(
        ascending=False).drop('chrX')
    lengths_clr = clr_bins.groupby('chrom', observed=True).apply(
        lambda x: int(round(x.end.max() / resolution)), include_groups=False).drop('chrX')
    assert (np.round((clr.chromsizes / resolution)).drop('chrX') - lengths_clr == 0).all()
    
    
    # Setup chromosome lengths data
    lengths_df = pd.read_csv(lengths_file, sep='\t')
    lengths_df.columns = [c.strip('#') for c in lengths_df.columns]
    for col in ['start', 'end', 'mid']:  # Round to nearest 10,000bp
        lengths_df[col] = lengths_df[col].round(-4)
    
    n = len(lengths_df)
    lengths_s = lengths_df.groupby('chrom').size().sort_values(
        ascending=False)
    
    lengths_compare = pd.concat([lengths_s.to_frame().rename({0: 'merfish'}, axis=1), lengths_clr.to_frame().rename({0: 'snm3c'}, axis=1)], axis=1)
    lengths_compare['difference'] = lengths_compare.snm3c - lengths_compare.merfish
    assert (lengths_compare.difference >= 0).all()
    lengths_df = lengths_df.merge(clr_bins[['chrom', 'mid', 'idx_clr']], on=['chrom', 'mid'], how='left')
    if verbose:
        print(f"{lengths_df.idx_clr.isnull().sum()} genomic bins are present in MERFISH but not snm3c", flush=True)
    lengths_df.index = lengths_df.idx_genome
    lengths_df.index.name = None
    
    # Setup matrix-related data
    matrix_df = make_matrix_df(lengths_df)
    matrix_df = matrix_df[matrix_df['mask.sameC-sameH'] | matrix_df['mask.sameC-diffH']].drop(
        ['mask.sameC-sameH', 'mask.sameC-diffH', 'mask.diffC-sameH', 'mask.diffC-diffH', 'j.chrom'], axis=1).rename(
        {'i.chrom': 'chrom'}, axis=1)
    matrix_df['i.idx_clr'] = lengths_df.loc[matrix_df['i.idx_ambig'], 'idx_clr'].values.ravel()
    matrix_df['j.idx_clr'] = lengths_df.loc[matrix_df['j.idx_ambig'], 'idx_clr'].values
    
    mask = matrix_df['i.idx_clr'].notnull() & matrix_df['j.idx_clr'].notnull()
    matrix_df.loc[mask, 'snm3c'] = snm3c[matrix_df.loc[mask, 'i.idx_clr'].astype(int), matrix_df.loc[mask, 'j.idx_clr'].astype(int)]
    matrix_df.loc[matrix_df['i.idx_ambig'] == matrix_df['j.idx_ambig'], 'snm3c'] = np.nan
    
    # Filter for locus pairs present in single-cell distances
    matrix_df = matrix_df.loc[matrix_df.index.isin(sc_dis.index)]

    return matrix_df, sc_dis, lengths_df


# ===================================================================================================================
# ===================================================================================================================


