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


from pastis.optimization.utils_poisson import _setup_jax
_setup_jax(traceback=False, debug_nan_inf=False)

from scipy import optimize
import jax.numpy as jnp
from jax import grad, jit
from jax.nn import relu


def prep_data_for_inference(dir_matrix2d, mcool_file, lengths_file=None, resolution=2.5e6, verbose=True):
    matrix_df, sc_dis, lengths_df = load(dir_matrix2d=dir_matrix2d, mcool_file=mcool_file, lengths_file=lengths_file, resolution=resolution, verbose=verbose)

    # Setup data for inference
    _, snm3c, dis_idx = ambiguate_matrix_df_for_inference(matrix_df)
    sc_dis_arr = sc_dis.loc[dis_idx].values
    sc_dis_arr[np.isnan(sc_dis_arr)] = 0
    sc_dis_arr = np.asarray(sc_dis_arr, order='C')
    data = InferArgs(snm3c=snm3c, sc_dis_arr=sc_dis_arr)

    return data

# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================

def logistic_jax(d, X, infer_nu=False, infer_q=False):
    q = nu = 1
    if infer_nu and infer_q:
        k, d0, nu, q = X
    elif infer_nu:
        k, d0, nu = X
    elif infer_q:
        k, d0, q = X
    else:
        k, d0 = X

    tmp = -k * (d - d0)
    bar = 1 + q * jnp.exp(tmp)
    baz = jnp.power(relu(bar), 1 / nu)
    counts = 1 / baz

    # counts = 1 / jnp.power(relu(1 + q * jnp.exp(tmp)), 1 / nu)
    # counts = 1 / jnp.power(1 + q * jnp.exp(tmp), 1 / nu)
    return counts


def get_pseudocounts_scaling_factor(pseudocounts, snm3c):
    scaling_factor = jnp.sum(pseudocounts * snm3c) / jnp.sum(jnp.square(pseudocounts))
    return scaling_factor


def get_bulk_pseudocounts(X, data, intramol_only=False, infer_nu=False,
                          infer_q=False):
    nbins = data.snm3c.size

    if intramol_only:
        dis_sc = data.dis[:nbins * 2]
    else:
        dis_sc = data.dis
    
    mask_sc = dis_sc > 0
    counts_sc = jnp.where(
        mask_sc,
        logistic_jax(dis_sc, X, infer_nu=infer_nu, infer_q=infer_q),
        jnp.zeros_like(dis_sc))
    mask_bulk = mask_sc.sum(axis=1)
    counts = jnp.where(
        mask_bulk > 0, counts_sc.sum(axis=1) / mask_bulk,
        jnp.zeros(counts_sc.shape[0])).reshape(-1, nbins).sum(axis=0)

    return counts


def obj_eval_pseudocounts(X, data, intramol_only=False, infer_nu=False,
                          infer_q=False, obj_type='pearson', verbose=False):
    if verbose:
        if type(X).__name__ == "DynamicJaxprTracer":
            print("Compiling objective function via Jax JIT...", flush=True)
        elif type(X).__name__ == "JVPTracer":
            print("Compiling gradient function via Jax JIT...", flush=True)

    if isinstance(obj_type, str):
        obj_type = (obj_type.lower(),)
    else:
        obj_type = tuple([x.lower() for x in obj_type])

    counts = get_bulk_pseudocounts(
        X, data=data, intramol_only=intramol_only, infer_nu=infer_nu,
        infer_q=infer_q)

    obj = 0
    if 'pearson' in obj_type:
        pearson_r = jnp.corrcoef(data.snm3c, counts)[0, 1]
        obj += -pearson_r
    if ('rmse' in obj_type) or ('mse' in obj_type):
        scaling_factor = get_pseudocounts_scaling_factor(counts, snm3c=data.snm3c)
        mse = jnp.mean(jnp.square(counts - data.snm3c))
        if 'mse' in obj_type:
            obj += mse
        elif 'rmse' in obj_type:
            obj += jnp.sqrt(mse)
        else:
            raise ValueError(f"{obj_type=}")
    else:
        raise ValueError(f"{obj_type=}")

    return obj


grad_eval_pseudocounts = grad(obj_eval_pseudocounts)
obj_eval_pseudocounts_jit = jit(
    obj_eval_pseudocounts, static_argnames=['data', 'intramol_only', 'infer_nu', 'infer_q', 'obj_type'])
grad_eval_pseudocounts_jit = grad(obj_eval_pseudocounts_jit)


def obj_wrap(X, data, intramol_only=False, infer_nu=False, infer_q=False,
             obj_type='pearson', jitted=False, verbose=False):
    if jitted:
        objective_func = obj_eval_pseudocounts_jit
    else:
        objective_func = obj_eval_pseudocounts
    
    obj = objective_func(
        X, data=data, intramol_only=intramol_only, infer_nu=infer_nu,
        infer_q=infer_q, obj_type=obj_type)
    if not isinstance(obj, float) and obj.size > 1:
        obj = sum(obj)
    if verbose:
        print_iter(X, infer_nu=infer_nu, infer_q=infer_q, note=obj)
    return obj


def grad_wrap(X, data, intramol_only=False, infer_nu=False, infer_q=False,
              obj_type='pearson', jitted=False, verbose=False):
    if jitted:
        gradient_func = grad_eval_pseudocounts_jit
    else:
        gradient_func = grad_eval_pseudocounts
    
    return np.array(gradient_func(
        X, data=data, intramol_only=intramol_only, infer_nu=infer_nu,
        infer_q=infer_q, obj_type=obj_type)).ravel()


def estimate_logistic_param(data, intramol_only=False, infer_nu=False, infer_q=False,
                            obj_type='pearson', init=None, bounds=None, seed=0, max_iter=1e20,
                            factr=1e7, maxls=20, pgtol=1e-05,
                            jitted=False, verbose=True):

    if isinstance(obj_type, str):
        obj_type = (obj_type.lower(),)
    else:
        obj_type = tuple([x.lower() for x in obj_type])

    # Get bounds
    buffer = 0.1
    if bounds is None:
        bounds = np.array([
            [-np.inf, 0 - buffer],  # k < 0
            [0, 1]])  # 0 <= d0 <= 1
        if infer_nu:  # 0 < nu <= 1
            bounds = np.concatenate(bounds, np.array([[0 + buffer, 1]]))
        if infer_q:  # q > 0
            bounds = np.concatenate(bounds, np.array([[0 + buffer, np.inf]]))
        bounds = np.nan_to_num(bounds, posinf=50, neginf=-50)  # XXX
    else:
        bounds = np.array(bounds, ndmin=2, copy=None).reshape(-1, 2)
    
    # Define initial value for X
    if init is None:
        rng = np.random.default_rng(seed=seed)
        tmp = np.nan_to_num(bounds, posinf=20, neginf=-20)
        init = rng.uniform(low=tmp[:, 0], high=tmp[:, 1])
        if verbose:
            print_iter(X=init, infer_nu=infer_nu, infer_q=infer_q, note='INIT')
    init = np.array(init, ndmin=1, copy=None)

    # Optimize
    args = [data, intramol_only, infer_nu, infer_q, obj_type, jitted, verbose > 1]
    if verbose:
        print("OPTIMIZING:", flush=True)
    obj = obj_wrap(
        init, data=data, intramol_only=intramol_only, infer_nu=infer_nu, infer_q=infer_q,
        obj_type=obj_type, jitted=False, verbose=verbose > 1)
    if max_iter == 0:
        X = init
        d = {'converged': 'N/A', 'obj': obj._value}
    else:
        results = optimize.fmin_l_bfgs_b(
            obj_wrap, x0=init, fprime=grad_wrap, bounds=bounds, args=args,
            factr=factr, maxls=maxls, pgtol=pgtol, maxiter=int(max_iter)) #, maxfun=max_fun)
        X, obj, d = results
        d['converged'] = d['warnflag'] == 0
        conv_desc = d['task']
        d['obj'] = obj
    if X.size == 1:
        d['params'] = X[0]
    else:
        d['params'] = X
    counts = get_bulk_pseudocounts(
        X, data=data, intramol_only=intramol_only, infer_nu=infer_nu,
        infer_q=infer_q)
    d['scaling_factor'] = float(get_pseudocounts_scaling_factor(counts, snm3c=data.snm3c)._value)

    if verbose:
        print("\nRESULTS:", flush=True)
        print_infer_results_pseudocounts(d, infer_nu=infer_nu, infer_q=infer_q)

    return d

# ===================================================================================================================

def print_iter(X, infer_nu=False, infer_q=False, note=None):
    q = nu = 1
    if infer_nu and infer_q:
        k, d0, nu, q = X
    elif infer_nu:
        k, d0, nu = X
    elif infer_q:
        k, d0, q = X
    else:
        k, d0 = X

    to_print = f"k={k:<8.6g}  d₀={d0:<8.6g}"
    if infer_nu:
        to_print += f"  𝜈={nu:<8.6g}"
    if infer_q:
        to_print += f"  q={q:<8.6g}"
    if isinstance(note, (str, int)):
        to_print += f" ... {note}"
    elif note is not None:
        to_print += f" ... obj={note:.8g}"
    print(to_print, flush=True)


def print_infer_results_pseudocounts(d, infer_nu=False, infer_q=False):
    for k, v in d.items():
        if k == 'params':
            v = np.array(v, ndmin=1, copy=None)
            print_iter(v, infer_nu=infer_nu, infer_q=infer_q)
            continue

        if k == 'grad':
            v = f"{v.mean():.3g}"
        elif isinstance(v, float):
            v = f"{v:.3g}"
        elif isinstance(v, np.ndarray) and v.size > 1:
            v = ', '.join([f"{x:.3g}" for x in v])
        print(f"{k.ljust(10)}   {v}", flush=True)

# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================

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


def load(dir_matrix2d, mcool_file, lengths_file=None, resolution=2.5e6, verbose=True):
    if lengths_file is None:
        lengths_file = os.path.join(os.path.dirname(dir_matrix2d), 'counts.bed')

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
    # print(snm3c.shape)

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

    # Setup matrix-related data for intra-chromosomal (withinter- and intra-homolog)
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

    # Genomic distances, including for inter-homolog
    matrix_df['genomic_dis_ambig'] = (matrix_df['i.idx_ambig'] - matrix_df['j.idx_ambig']).abs()

    return matrix_df, sc_dis, lengths_df
