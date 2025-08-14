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
from jax import config as jax_config
jax_config.update("jax_captured_constants_warn_bytes", -1)

from scipy import optimize
import jax.numpy as jnp
from jax import grad, jit
from jax.nn import relu



def prep_data_for_inference(dir_matrix2d, mcool_file, scale_snm3c_by=1,
                            resolution=2.5, normalize_snm3c=True, weights_exponent=None,
                            constraint_opt=0, constraint_penalty=0, constraint_min_n=None,
                            data_outdir=None, name=None, remake_npzfile=False, verbose=True):
    snm3c_type = f"{'raw' if '.raw.' in mcool_file.lower() else 'q'}_snm3c"
    if name is None:
        name = snm3c_type
    elif not ('raw' in name.lower() or 'q' in name.lower()):
        name = f"{name}.{snm3c_type}"
    if not 'norm' in name.lower():
        name = f"{name}_" + ('norm' if normalize_snm3c else 'orig')
    desc = f".{name}.{resolution * 1e3:g}kb"
    
    npzfile = npz_open = None
    saved_data_avail = False
    if data_outdir is not None:
        npzfile = os.path.join(data_outdir, f'pseudocounts_infer_logistic.data{desc}.npz')
        if os.path.isfile(npzfile) and not remake_npzfile:
            if verbose:
                print(f'Loading: {npzfile}', flush=True)
            npz_open = np.load(npzfile, allow_pickle=True)
            if set(npz_open.files) == {'snm3c', 'sc_dis_arr', 'genomic_dis', 'nghbr_dis_quantiles', 'chrom'}:
                saved_data_avail = True
                snm3c = npz_open['snm3c']
                sc_dis_arr = npz_open['sc_dis_arr']
                genomic_dis = npz_open['genomic_dis']
                chrom = npz_open['chrom']
                nghbr_dis_quantiles = npz_open['nghbr_dis_quantiles']

    if not saved_data_avail:
        # Load snm3c and single-cell distances (not scaling snm3c here...)
        matrix_df, sc_dis, lengths_df = load(
            dir_matrix2d=dir_matrix2d, mcool_file=mcool_file, resolution=resolution,
            normalize_snm3c=normalize_snm3c, verbose=verbose)
    
        # Describe 3D distances between neighboring loci
        nghbr_dis = sc_dis.loc[(matrix_df['genomic_dis'] == 1) & (
            ~matrix_df['mask.diffM']) & matrix_df['snm3c'].notnull()].values
        nghbr_dis_quantiles = {}
        for q in [0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 1]:
            nghbr_dis_quantiles[q] = np.nanquantile(nghbr_dis, q)
            nghbr_dis_quantiles[round(1 - q, 10)] = np.nanquantile(nghbr_dis, 1 - q)
        nghbr_dis_quantiles = pd.Series(nghbr_dis_quantiles).sort_values()
    
        # Setup data for inference
        _, snm3c, dis_idx, genomic_dis, chrom = ambiguate_matrix_df_for_inference(
            matrix_df, snm3c_are_normed=normalize_snm3c)
        sc_dis_arr = sc_dis.loc[dis_idx].values
        sc_dis_arr[np.isnan(sc_dis_arr)] = 0
        sc_dis_arr = np.asarray(sc_dis_arr, order='C')
    
        if data_outdir is not None:
            os.makedirs(data_outdir, exist_ok=True)
            np.savez(npzfile, snm3c=snm3c, sc_dis_arr=sc_dis_arr, genomic_dis=genomic_dis,
                     chrom=chrom, nghbr_dis_quantiles=np.stack(
                         [nghbr_dis_quantiles.index.values, nghbr_dis_quantiles.values]))

    data = InferArgs(
        snm3c=snm3c, sc_dis_arr=sc_dis_arr, genomic_dis=genomic_dis, chrom=chrom,
        nghbr_dis_quantiles=nghbr_dis_quantiles, snm3c_are_normed=normalize_snm3c,
        constraint_opt=constraint_opt, constraint_penalty=constraint_penalty,
        constraint_min_n=constraint_min_n, weights_exponent=weights_exponent,
        scale_snm3c_by=scale_snm3c_by)

    return data


# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================

def parse_X(X, infer_nu=False, infer_q=False, infer_a=False):
    q = nu = 1
    # q = 1; nu = 1e-8
    a = 0
    if infer_a:
        a = X[-1]
        X = X[:-1]
    if infer_q:
        q = X[-1]
        X = X[:-1]
    if infer_nu:
        nu = X[-1]
        X = X[:-1]
    k, d0 = X
    return k, d0, nu, q, a


def logistic_jax(d, X, infer_nu=False, infer_q=False, infer_a=False, ln_scale_snm3c_by=1):
    k, d0, nu, q, a = parse_X(X, infer_nu=infer_nu, infer_q=infer_q, infer_a=infer_a)

    tmp = -k * (d - d0)

    log_bar = jnp.log1p(q * jnp.exp(tmp))
    # bar = 1 + q * jnp.exp(tmp)

    # HERE
    # for k<0,  tmp >> 0 when d >> d0 (small d0)... especially when magnitude of k is big
    # for large postive tmp (>>0), log1p(exp(tmp)) approaches tmp (is approx equal to tmp, slightly larger)
    #         if tmp is 36, log1p(exp(tmp)) is also approx 36 → log(counts) approaches -36/nu...
    #         ...for nu=1, counts=2.3e16, as nu approaches 0, counts approaches 0
    # for large negative tmp (<<0), log1p(exp(tmp)) approaches 0
    # for near-0 tmp, log1p(exp(tmp)) approaches ln(2)=0.693 → log(counts)= -ln(2)/nu, counts = e**(-ln(2)/nu) = (e**(-ln2))**(1/nu) = (0.5)**(1/nu)
    #         aka... def y_when_x_is_x0(v, k=-1):   return 2 ** (-1/v)

    # NOTES: 
    # If bar<0 and nu>1:  baz=bar^(1/nu)=nan
    # If bar=0:           baz=bar^(1/nu)=0 → counts=1/baz=Inf
    #                     log(baz)=log(bar)/nu=-Inf
    # ...But bar isn't <=0 unless q<=0 and q<-1/e^tmp... so it doesn't matter, given my bound of q>0
    # # invalid value (inf) encountered in pow
    # if type(X).__name__ not in ("JVPTracer", "DynamicJaxprTracer"):
    #     print_iter(X, infer_nu=infer_nu, infer_q=infer_q, infer_a=infer_a, note='AGH')
    #     print(f"tmp... mean={tmp.mean():<.4g}  min={tmp[jnp.isfinite(tmp)].min():<.4g}  max={tmp[jnp.isfinite(tmp)].max():<.4g}", flush=True)
    #     print(f"bar... mean={bar.mean():<.4g}  min={bar[jnp.isfinite(bar)].min():<.4g}  max={bar[jnp.isfinite(bar)].max():<.4g}", flush=True)
    
    log_baz = log_bar / nu
    log_counts = -log_baz
    if ln_scale_snm3c_by is not None and ln_scale_snm3c_by != 0:
        log_counts = log_counts + ln_scale_snm3c_by
    
    counts = jnp.exp(log_counts)
    # baz = jnp.power(bar, 1 / nu)
    # counts = 1 / baz

    if infer_a:
        counts = a + (1 - a) * counts

    # mask_sc = d > 0 # **** XXX TEMP
    # tmp = {'tmp': tmp, 'log_bar': log_bar, 'log_counts': log_counts, 'counts': counts}
    # for key, val in tmp.items():
    #     toprint = f"*** {key:.<10} max={val[mask_sc].max():<6.3g} min={val[mask_sc].min():.<12.3g}  "
    #     if not jnp.all(jnp.isfinite(val[mask_sc])):
    #         toprint = f"{toprint}NaN/Inf, "
    #     if jnp.any(val[mask_sc] == 0):
    #         toprint = f"{toprint}0, "
    #     if jnp.any(val[mask_sc] < 0):
    #         toprint = f"{toprint}<0"
    #     print(toprint, flush=True)
    # exit(0)

    # counts = 1 / jnp.power(1 + q * jnp.exp(tmp), 1 / nu)
    return counts


def _get_pseudocounts_scaling_factor(pseudocounts, snm3c):
    scaling_factor = jnp.sum(pseudocounts * snm3c) / jnp.where(
        pseudocounts.sum() == 0, 1, jnp.sum(jnp.square(pseudocounts)))
    return scaling_factor


def get_pseudocounts_scaling_factors(pseudocounts, snm3c, chrom):
    if chrom is None:
        return float(_get_pseudocounts_scaling_factor(
            pseudocounts=pseudocounts, snm3c=snm3c)._value)
    scaling_factors = {}
    for x in np.unique(chrom):
        mask = chrom == x
        scaling_factors[x] = float(_get_pseudocounts_scaling_factor(
            pseudocounts=pseudocounts[mask], snm3c=snm3c[mask])._value)
    return pd.Series(scaling_factors)


def rescale_pseudocounts(pseudocounts, snm3c, chrom, snm3c_are_normed):
    if not snm3c_are_normed:
        scaling_factor = _get_pseudocounts_scaling_factor(
            pseudocounts=pseudocounts, snm3c=snm3c)
        return pseudocounts * scaling_factor

    for x in np.unique(chrom):
        mask = chrom == x
        scaling_factor = _get_pseudocounts_scaling_factor(
            pseudocounts=pseudocounts[mask], snm3c=snm3c[mask])
        pseudocounts = jnp.where(mask, pseudocounts * scaling_factor, pseudocounts)
    return pseudocounts


def get_bulk_pseudocounts(X, data, intramol_only=False, infer_nu=False,
                          infer_q=False, infer_a=False):
    nbins = data.snm3c.size

    if intramol_only:
        dis_sc = data.dis[:nbins * 2] # HERE
    else:
        dis_sc = data.dis
    
    mask_sc = dis_sc > 0
    counts_sc = jnp.where(
        mask_sc,
        logistic_jax(
            dis_sc, X, infer_nu=infer_nu, infer_q=infer_q, infer_a=infer_a,
            ln_scale_snm3c_by=data.ln_scale_snm3c_by),
        jnp.zeros_like(dis_sc))
    mask_bulk = mask_sc.sum(axis=1)
    counts = jnp.where(
        mask_bulk > 0, counts_sc.sum(axis=1) / mask_bulk,
        jnp.zeros(counts_sc.shape[0])).reshape(-1, nbins).sum(axis=0)

    return counts


def weighted_cov(x, y, weights=1):
    if isinstance(weights, (int, float)):
        weights = jnp.full_like(x, weights)
    x_weighted = jnp.average(x, weights=weights)
    y_weighted = jnp.average(y, weights=weights)
    # return jnp.sum(weights * (x - x_weighted) * (y - y_weighted)) / jnp.sum(weights)
    return jnp.average((x - x_weighted) * (y - y_weighted), weights=weights)


def weighted_pearsons_corr(x, y, weights=1):
    if isinstance(weights, (int, float)):
        weights = jnp.full_like(x, weights)
    # return weighted_cov(x, y, weights) / jnp.sqrt(
    #     weighted_cov(x, x, weights) * weighted_cov(y, y, weights))
    return weighted_cov(x, y, weights) / jnp.sqrt(jnp.where(
        x.sum() * y.sum() == 0, 1,
        weighted_cov(x, x, weights) * weighted_cov(y, y, weights)))


def _pearson_obj(pseudocounts, snm3c, weights):
    if weights is None:
        return jnp.corrcoef(snm3c, pseudocounts)[0, 1]
    else:
        return weighted_pearsons_corr(snm3c, pseudocounts, weights=weights)


def pearson_obj(pseudocounts, snm3c, weights, chrom, snm3c_are_normed):
    if not snm3c_are_normed:
        return _pearson_obj(pseudocounts, snm3c=snm3c, weights=weights)

    unique_chrom = np.unique(chrom)
    tmp = 0
    for x in unique_chrom:
        mask = chrom == x
        if weights is not None and weights != 1:
            weights_tmp = weights[mask]
        else:
            weights_tmp = None
        r = _pearson_obj(
            pseudocounts=pseudocounts[mask], snm3c=snm3c[mask], weights=weights_tmp)
        tmp = tmp + r * mask.sum()
    return tmp / snm3c.size


def obj_eval_pseudocounts(X, data, intramol_only=False, infer_nu=False,
                          infer_q=False, infer_a=False, obj_type='pearson',
                          verbose=False):
    if verbose:
        if type(X).__name__ == "DynamicJaxprTracer":
            print("Compiling objective function via Jax JIT...", flush=True)
        elif type(X).__name__ == "JVPTracer":
            print("Compiling gradient function via Jax JIT...", flush=True)

    if isinstance(obj_type, str):
        obj_type = (obj_type.lower(),)
    else:
        obj_type = tuple([x.lower() for x in obj_type])
    valid_obj_type = ['pearson', 'rmse', 'mse']
    if len([x for x in obj_type if x not in valid_obj_type]):
        raise ValueError(f"{obj_type=}")

    counts = get_bulk_pseudocounts(
        X, data=data, intramol_only=intramol_only, infer_nu=infer_nu,
        infer_q=infer_q, infer_a=infer_a)

    obj = 0
    counts_are_rescaled = False
    if 'pearson' in obj_type:
        pearson_r = pearson_obj(
            counts, snm3c=data.snm3c, weights=data.weights, chrom=data.chrom,
            snm3c_are_normed=data.snm3c_are_normed)
        obj = obj - pearson_r
    if ('rmse' in obj_type) or ('mse' in obj_type):
        counts = rescale_pseudocounts(
            counts, snm3c=data.snm3c, chrom=data.chrom, snm3c_are_normed=data.snm3c_are_normed)
        counts_are_rescaled = True
        if data.weights is None:
            mse = jnp.mean(jnp.square(counts - data.snm3c))
        else:
            mse = jnp.average(jnp.square(counts - data.snm3c), weights=data.weights)
        if 'mse' in obj_type:
            obj = obj + mse
        elif 'rmse' in obj_type:
            obj = obj + jnp.sqrt(mse)

    if data.agg_penalty:  # Apply constraint (based on mean counts per genomic distance bin)
        if not counts_are_rescaled:
            counts = rescale_pseudocounts(
                counts, snm3c=data.snm3c, chrom=data.chrom, snm3c_are_normed=data.snm3c_are_normed)
        agg_counts = jnp.histogram(
            data.agg_category, weights=counts[data.agg_mask], bins=data.agg_n.size,
            density=False)[0] / data.agg_n

        if data.constraint_v3:
            agg_counts_nghbr = agg_counts[data.agg_nghbr_idx]

            # agg_counts_ratio = agg_counts / agg_counts_nghbr
            mask = agg_counts_nghbr != 0
            agg_counts_ratio = jnp.where(mask, agg_counts, 0) / jnp.where(mask, agg_counts_nghbr, 1)

            agg_diff = agg_counts_ratio - data.agg_snm3c_ratio
        else:
            agg_diff = agg_counts - data.agg_snm3c
        
        if data.agg_weight is None:
            agg_mse = jnp.mean(jnp.square(agg_diff))
        else:
            agg_mse = jnp.average(jnp.square(agg_diff), weights=data.agg_weight)
        aux = {'main': obj}
        obj = obj + data.agg_penalty * agg_mse
        aux['const'] = agg_mse
        aux['obj'] = obj
    else:
        aux = obj

    return obj, aux


grad_eval_pseudocounts = grad(obj_eval_pseudocounts, has_aux=True)
obj_eval_pseudocounts_jit = jit(
    obj_eval_pseudocounts, static_argnames=['data', 'intramol_only', 'infer_nu', 'infer_q', 'infer_a', 'obj_type'])
grad_eval_pseudocounts_jit = grad(obj_eval_pseudocounts_jit, has_aux=True)


def obj_wrap(X, data, intramol_only=False, infer_nu=False, infer_q=False,
             infer_a=False, obj_type='pearson', jitted=False, verbose=False):
    if jitted:
        objective_func = obj_eval_pseudocounts_jit
    else:
        objective_func = obj_eval_pseudocounts
    
    obj, aux = objective_func(
        X, data=data, intramol_only=intramol_only, infer_nu=infer_nu,
        infer_q=infer_q, infer_a=infer_a, obj_type=obj_type)
    if not isinstance(obj, float) and obj.size > 1:
        obj = sum(obj)
    if verbose:
        print_iter(X, infer_nu=infer_nu, infer_q=infer_q, infer_a=infer_a, note=aux)
    return obj


def grad_wrap(X, data, intramol_only=False, infer_nu=False, infer_q=False,
              infer_a=False, obj_type='pearson', jitted=False, verbose=False):
    if jitted:
        gradient_func = grad_eval_pseudocounts_jit
    else:
        gradient_func = grad_eval_pseudocounts

    res, _ = gradient_func(
        X, data=data, intramol_only=intramol_only, infer_nu=infer_nu,
        infer_q=infer_q, infer_a=infer_a, obj_type=obj_type)
    return np.array(res).ravel()


def estimate_logistic_param(data, intramol_only=False, infer_nu=False, infer_q=False,
                            infer_a=False, obj_type='pearson', init=None, bounds=None, seed=0, max_iter=1e20,
                            factr=1e7, maxls=20, pgtol=1e-05, jitted=True, verbose=True):

    if isinstance(obj_type, str):
        obj_type = (obj_type.lower(),)
    else:
        obj_type = tuple([x.lower() for x in obj_type])

    # Get bounds
    if bounds is None:
        buffer = 1e-8
        bound_k = [-np.inf, 0 - buffer]  # k < 0
        bound_d0 = [0, data.nghbr_dis_quantiles[1]]  # 0 <= d0 <= max(neighbor bead dist)
        bound_nu = [0 + buffer, 5]  # 0 < nu <= 5  # XXX ************************
        # bound_nu = [0 + buffer, np.inf]  # nu > 0  # XXX ************************
        bound_q = [0 + buffer, np.inf]  # q > 0
        bound_a = [0, 1]  # 0 <= a <= 1
        bounds = [np.array(bound_k, ndmin=2), np.array(bound_d0, ndmin=2)]
        if infer_nu:
            bounds.append(np.array(bound_nu, ndmin=2))
        if infer_q:
            bounds.append(np.array(bound_q, ndmin=2))
        if infer_a:
            bounds.append(np.array(bound_a, ndmin=2))
        bounds = np.concatenate(bounds)
    else:
        bounds = np.array(bounds, ndmin=2, copy=None).reshape(-1, 2)
    
    # Define initial value for X
    if init is None:
        rng = np.random.default_rng(seed=seed)
        tmp = np.nan_to_num(bounds, posinf=10, neginf=-10) # 10, 25, 100  # XXX ************************
        init = rng.uniform(low=tmp[:, 0], high=tmp[:, 1])
    init = np.array(init, ndmin=1, copy=None)
    if verbose:
        print_iter(X=init, infer_nu=infer_nu, infer_q=infer_q, infer_a=infer_a, note='INIT')

    # Optimize
    args = [data, intramol_only, infer_nu, infer_q, infer_a, obj_type, jitted, verbose > 1]
    if verbose:
        print("OPTIMIZING:", flush=True)
    obj = obj_wrap(
        init, data=data, intramol_only=intramol_only, infer_nu=infer_nu, infer_q=infer_q,
        infer_a=infer_a, obj_type=obj_type, jitted=False, verbose=verbose > 1)
    if max_iter == 0:
        X = init
        d = {'converged': 'N/A', 'obj': obj._value, 'grad': np.zeros(1)}
    else:
        results = optimize.fmin_l_bfgs_b(
            obj_wrap, x0=init, fprime=grad_wrap, bounds=bounds, args=args,
            factr=factr, maxls=maxls, pgtol=pgtol, maxiter=int(max_iter)) #, maxfun=max_fun)
        X, obj, d = results
        d['converged'] = d['warnflag'] == 0
        conv_desc = d['task']
        d['obj'] = obj

    # Add results to dict
    d['k'], d['d0'], d['nu'], d['q'], d['a'] = parse_X(
        X, infer_nu=infer_nu, infer_q=infer_q, infer_a=infer_a)
    d['params'] = X[0] if X.size == 1 else X
    counts = get_bulk_pseudocounts(
        X, data=data, intramol_only=intramol_only, infer_nu=infer_nu,
        infer_q=infer_q, infer_a=infer_a)
    scaling_factors = get_pseudocounts_scaling_factors(
        counts, snm3c=data.snm3c, chrom=data.chrom)
    d['scaling_factor'] = scaling_factors.values

    if verbose > 1:
        print("\nRESULTS:", flush=True)
        print_infer_results_pseudocounts(d, infer_nu=infer_nu, infer_q=infer_q, infer_a=infer_a)

    return d

# ===================================================================================================================

def print_iter(X, infer_nu=False, infer_q=False, infer_a=False, note=None):
    k, d0, nu, q, a = parse_X(X, infer_nu=infer_nu, infer_q=infer_q, infer_a=infer_a)

    to_print = f"k={k:<11.6g}  d₀={d0:<11.6g}"
    if infer_nu:
        to_print += f"  𝜈={nu:<11.6g}"
    if infer_q:
        to_print += f"  q={q:<11.6g}"
    if infer_a:
        to_print += f"  a={a:<11.6g}"
    if isinstance(note, (str, int)):
        to_print += f" ... {note}"
    elif isinstance(note, dict):
        to_print += f" ... obj={note['obj']:<15.10g}  M={note['main']:<10.5g}  C={note['const']:<10.5g}"
    elif note is not None:
        to_print += f" ... obj={note:<15.10g}"
    print(to_print, flush=True)


def print_infer_results_pseudocounts(d, infer_nu=False, infer_q=False, infer_a=False):
    for k, v in d.items():
        if k in ('k', 'd0', 'nu', 'q'):
            continue
        if k == 'params':
            v = np.array(v, ndmin=1, copy=None)
            print_iter(v, infer_nu=infer_nu, infer_q=infer_q, infer_a=infer_a)
            continue
        if k == 'grad':
            v = f"{v.max():.3g}"
        elif isinstance(v, float):
            v = f"{v:.3g}"
        elif isinstance(v, np.ndarray) and v.size > 1:
            v = ', '.join([f"{x:.3g}" for x in v])
        print(f"{k.ljust(10)}   {v}", flush=True)

# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================


def compare_to_genomic_nghbr(df):
    df['nghbr_idx'] = df.loc[df.index.get_level_values(1) == 1, 'category'].values.item()
    df['snm3c_ratio'] = df.snm3c / df.loc[df.index.get_level_values(1) == 1, 'snm3c'].values.item()
    return df
    

def setup_dist_decay_obj(genomic_dis, chrom, snm3c, constraint_opt=0, min_n=None):
    tmp = pd.DataFrame.from_dict(
        dict(genomic_dis=genomic_dis, chrom=chrom, snm3c=snm3c))

    tmp['weight'] = 1
    if constraint_opt != 0:
        tmp['weight'] = tmp.genomic_dis.pow(float(constraint_opt))

    tmp['mask'] = True
    tmp['n'] = 1
    tmp['sort_order'] = np.arange(len(tmp), dtype=int)
    tmp.set_index(['chrom', 'genomic_dis'], inplace=True)

    agg = tmp.groupby(level=[0, 1]).agg({
        'snm3c': 'mean', 'n': 'sum', 'sort_order': 'min', 'mask': lambda x: x.iloc[0],
        'weight': lambda x: x.iloc[0]}).sort_values('sort_order').drop('sort_order', axis=1)
    tmp['mask'] = True
    if min_n is not None:
        agg.loc[agg.n < min_n, 'mask'] = False
    agg['category'] = -1
    agg.loc[agg['mask'], 'category'] = np.arange(agg['mask'].sum(), dtype=int)
    agg['weight'] *= agg['n']  # Also weight by nbins (to make it proportional)
    agg = agg.groupby(level=0).apply(
        compare_to_genomic_nghbr, include_groups=False).reset_index(level=0, drop=True)

    df = tmp[['sort_order']].join(agg).sort_values('sort_order')
    
    agg_snm3c = np.asarray(agg.loc[agg['mask'], 'snm3c'].values, order='C')
    agg_n = np.asarray(agg.loc[agg['mask'], 'n'].values, order='C')
    if not (agg.loc[agg['mask'], 'weight'] == 1).all():
        agg_weight = np.asarray(agg.loc[agg['mask'], 'weight'].values, order='C')
    else:
        agg_weight = None
    agg_snm3c_ratio = np.asarray(agg.loc[agg['mask'], 'snm3c_ratio'].values, order='C')
    agg_nghbr_idx = np.asarray(agg.loc[agg['mask'], 'nghbr_idx'].values, order='C')
        
    agg_category = np.asarray(df.loc[df['mask'], 'category'].values, order='C')
    agg_mask = np.asarray(df['mask'].values, order='C')

    return agg_snm3c, agg_n, agg_weight, agg_category, agg_mask, agg_snm3c_ratio, agg_nghbr_idx


class InferArgs(object):
    def __init__(self, snm3c, sc_dis_arr, genomic_dis, chrom, nghbr_dis_quantiles,
                 snm3c_are_normed=True, constraint_opt=0, constraint_penalty=0,
                 constraint_min_n=None, constraint_v3=True, weights_exponent=None, scale_snm3c_by=1):        
        self.snm3c = snm3c
        self.snm3c_are_normed = snm3c_are_normed
        if snm3c_are_normed:
            self.chrom = chrom
        else:
            self.chrom = None
        self.dis = sc_dis_arr

        if scale_snm3c_by is None or scale_snm3c_by == 1:
            self.scale_snm3c_by = 1.
            self.ln_scale_snm3c_by = 0.
        else:
            self.scale_snm3c_by = float(scale_snm3c_by)
            self.ln_scale_snm3c_by = np.log(scale_snm3c_by)
            self.snm3c *= scale_snm3c_by
            print(f'\tSCALE SNM3C BY ~~~~~~~~~~~~~~~~~~~~~~~~~~ {scale_snm3c_by=}\n', flush=True)

        if constraint_penalty is None or constraint_penalty == 0:
            self.agg_snm3c = None
            self.agg_penalty = 0
        else:
            if constraint_min_n is None and constraint_v3:
                constraint_min_n = 4
            if constraint_v3:
                print(f'\tCONSTRAINT: DIST DECAY **V3** ~~~~~~~~~~~~~~~~~~ penalty={constraint_penalty:g}, opt={constraint_opt:g}, min_n={constraint_min_n}\n', flush=True)
            else:
                print(f'\tCONSTRAINT: genomic dist (V2) ~~~~~~~~~~~~~~~~~~ penalty={constraint_penalty:g}, opt={constraint_opt:g}, min_n={constraint_min_n}\n', flush=True)
            self.constraint_v3 = constraint_v3
            self.agg_penalty = constraint_penalty
            
            tmp = setup_dist_decay_obj(
                genomic_dis=genomic_dis, chrom=chrom, snm3c=snm3c,
                constraint_opt=constraint_opt, min_n=constraint_min_n)
            (self.agg_snm3c, self.agg_n, self.agg_weight, self.agg_category,
             self.agg_mask, self.agg_snm3c_ratio, self.agg_nghbr_idx) = tmp
            

        if isinstance(nghbr_dis_quantiles, (np.ndarray, tuple, list)):
            if isinstance(nghbr_dis_quantiles, np.ndarray) and nghbr_dis_quantiles.shape[1] == 2:
                nghbr_dis_quantiles = nghbr_dis_quantiles.T
            if len(nghbr_dis_quantiles) != 2:
                raise ValueError(f"Input not understood for nghbr_dis_quantiles:\n{nghbr_dis_quantiles}")
            nghbr_dis_quantiles = pd.Series(nghbr_dis_quantiles[1], index=nghbr_dis_quantiles[0])
        if isinstance(nghbr_dis_quantiles, pd.Series):
            nghbr_dis_quantiles = nghbr_dis_quantiles.to_dict()
        elif not isinstance(nghbr_dis_quantiles, dict):
            raise ValueError(f"Input not understood for nghbr_dis_quantiles:\n{nghbr_dis_quantiles}")
        self.nghbr_dis_quantiles = nghbr_dis_quantiles
        # print('3D distance between neighboring loci, quantiles:\n\t' + pd.Series(
        #     nghbr_dis_quantiles).sort_values().to_frame().reset_index().to_string(
        #     header=False, index=False, float_format=lambda x: f"{x:.4g}").replace('\n', '\n\t'), flush=True)
        # print(nghbr_dis_quantiles)
        
        if weights_exponent is None or not weights_exponent:
            self.weights = None
        else:
            weights = np.power(genomic_dis.astype(float), weights_exponent)
            print(f'\nWEIGHTED ~~~~~~~~~~~~~~~~~~~~~~~~~~{weights_exponent=}\n', flush=True)
            self.weights = np.asarray(weights, order='C')

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
    # if 'snm3c' in df.columns:
    #     s['snm3c'] = df.snm3c.mean()
    # if 'genomic_dis_ambig' in df.columns:
    #     s['genomic_dis_ambig'] = df.genomic_dis_ambig.mean()
    return s


def ambiguate_matrix_df_for_inference(matrix_df, snm3c_are_normed, extra_grouping_cols=None):
    grp_cols = ['i.idx_ambig', 'j.idx_ambig', 'chrom', 'i.idx_chrom', 'j.idx_chrom',
                'i.idx_clr', 'j.idx_clr', 'genomic_dis_ambig', 'snm3c']
    if extra_grouping_cols is not None:
        if isinstance(extra_grouping_cols, str):
            extra_grouping_cols = [extra_grouping_cols]
        grp_cols.extend(extra_grouping_cols)
    matrix_df_ambig = matrix_df.copy()
    mask_swap = matrix_df['i.idx_ambig'] > matrix_df['j.idx_ambig']
    if mask_swap.any():  # Ambig index pair should fall in upper triangular
        for col in ('idx_ambig', 'idx_chrom', 'hmlg', 'idx_clr'):
            matrix_df_ambig.loc[mask_swap, f'i.{col}'] = matrix_df.loc[mask_swap, f'j.{col}']
            matrix_df_ambig.loc[mask_swap, f'j.{col}'] = matrix_df.loc[mask_swap, f'i.{col}']
    matrix_df_ambig = matrix_df_ambig[  # Remove main diagonal
        matrix_df_ambig['i.idx_ambig'] != matrix_df_ambig['j.idx_ambig']]
    matrix_df_ambig = matrix_df_ambig[grp_cols + ['i.hmlg', 'j.hmlg']].groupby(
        grp_cols).apply(get_unambig_idx_per_ambig, include_groups=False).reset_index(
        level=np.arange(len(grp_cols)).tolist())

    # Rescale snm3c so that mean of bins between neighboring loci is 1
    if snm3c_are_normed:
        snm3c_s1_mean = matrix_df_ambig[matrix_df_ambig.genomic_dis_ambig == 1].groupby(
            'chrom').snm3c.mean()[matrix_df_ambig.chrom].values
    else:
        snm3c_s1_mean = matrix_df_ambig[matrix_df_ambig.genomic_dis_ambig == 1, 'snm3c'].mean()
    matrix_df_ambig['snm3c'] /= snm3c_s1_mean

    matrix_df_ambig = matrix_df_ambig[matrix_df_ambig.snm3c.notnull()]
    # matrix_df_ambig.groupby('chrom').snm3c.mean().reset_index()  # FIXME
    matrix_df_ambig['chromnum'] = matrix_df_ambig.chrom.str.replace('chr', '', regex=False).astype(int)
    # matrix_df_ambig.sort_values(['chromnum', 'i.idx_chrom', 'j.idx_chrom'], inplace=True)
    # matrix_df_ambig.sort_values(['genomic_dis_ambig', 'chromnum'], inplace=True)
    matrix_df_ambig.sort_values(
        ['chromnum', 'genomic_dis_ambig', 'i.idx_chrom', 'j.idx_chrom'], inplace=True)

    snm3c = np.asarray(matrix_df_ambig.snm3c.values, order='C')
    genomic_dis = np.asarray(matrix_df_ambig.genomic_dis_ambig.values, order='C')
    chrom = np.asarray(matrix_df_ambig.chromnum.values, order='C')

    quadrants = ['h1_h1', 'h2_h2', 'h1_h2', 'h2_h1']
    dis_idx = np.concatenate([matrix_df_ambig[q].values for q in quadrants]) # HERE
    dis_idx = np.asarray(dis_idx, order='C')

    return matrix_df_ambig, snm3c, dis_idx, genomic_dis, chrom


# def load_cooler(mcool_file, resolution):
#     resolution_bp = int(resolution * 1e6)
#     uri = f'{mcool_file}::/resolutions/{resolution_bp:d}'
#     clr = cooler.Cooler(uri)
#     return clr
#     # clr = load_cooler(mcool_file, resolution=resolution)


def load_snm3c(mcool_file, resolution=2.5, normalize=True, mask_last_locus_in_chrom=True, verbose=True):
    if isinstance(normalize, str) and normalize.lower() == 'auto':
        normalize = True if 'Raw.' in mcool_file else False
    if verbose:
        if normalize:
            print("Loading and normalizing snm3c-seq data...", flush=True)
        else:
            print("Loading snm3c-seq data, skipping normalization...", flush=True)

    # Load snm3c-seq data via cooler
    resolution_bp = int(resolution * 1e6)
    uri = f'{mcool_file}::/resolutions/{resolution_bp:d}'
    clr = cooler.Cooler(uri)
    snm3c = clr.matrix(balance=normalize)[:, :]
    if clr.storage_mode != 'symmetric-upper':
        if not np.all(np.isnan(snm3c[np.tril_indices(snm3c.shape[0], -1)]) | (snm3c[np.tril_indices(snm3c.shape[0], -1)] == 0)):
            warnings.warn("snm3C-seq cooler matrix is not symmetric, has different values below diagonal")
        snm3c[np.tril_indices(snm3c.shape[0])] = 0
        snm3c += snm3c.T
    np.fill_diagonal(snm3c, np.nan)
    nan_loci_clr = np.nansum(snm3c, axis=0) == 0
    snm3c[nan_loci_clr, :] = np.nan
    snm3c[:, nan_loci_clr] = np.nan
    
        
    clr_bins = clr.bins()[:]
    assert len(clr_bins) == snm3c.shape[0]
    clr_bins['mid'] = clr_bins.start + int(round(resolution / 2 * 1e6))
    clr_bins['snm3c_isnan'] = False
    clr_bins.loc[nan_loci_clr, 'snm3c_isnan'] = True

    lengths_clr = clr_bins.groupby('chrom', observed=True).apply(
        lambda x: int(np.ceil(x.end.max() / resolution_bp)), include_groups=False)
    assert (np.ceil((clr.chromsizes / resolution_bp)) == lengths_clr).all()
    assert (clr_bins.groupby('chrom', observed=True).size() == lengths_clr).all()
    assert lengths_clr.sum() == snm3c.shape[0]

    # Drop chrX
    not_chrX = clr_bins.chrom != 'chrX'
    snm3c = snm3c[not_chrX, :][:, not_chrX]
    clr_bins = clr_bins[not_chrX]
    lengths_clr = lengths_clr.drop('chrX')

    # Remove data from last bin per chromosome, which is associated with fewer bp than other bins
    if mask_last_locus_in_chrom:
        small_bins_check = (clr_bins.groupby('chrom', observed=True).apply(
            lambda x: x.tail(1), include_groups=False)[['end']] % resolution_bp / resolution_bp).reset_index(
            level=1).rename({'level_1': 'idx', 'end': 'ratio'}, axis=1)
        # small_bins_check = small_bins_check[small_bins_check.ratio <= mask_last_locus_in_chrom]
        snm3c[small_bins_check.idx.values, :] = np.nan
        snm3c[:, small_bins_check.idx.values] = np.nan

    return snm3c, clr_bins, lengths_clr


def load(dir_matrix2d, mcool_file, lengths_file=None, resolution=2.5, normalize_snm3c=True, verbose=True):
    # Load snm3c-seq data via cooler
    snm3c, clr_bins, lengths_clr = load_snm3c(
        mcool_file, resolution=resolution, normalize=normalize_snm3c, verbose=verbose)
    clr_bins = clr_bins.reset_index().rename({'index': 'idx_clr'}, axis=1)
    # if scale_snm3c_by is not None and scale_snm3c_by != 1:
    #     if verbose:
    #         print(f"Scaling snm3c-seq data by {scale_snm3c_by:.3g}", flush=True)
    #     snm3c *= scale_snm3c_by

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
    n = len(lengths_df)
    lengths_s = lengths_df.groupby('chrom').size().sort_values(
        ascending=False)

    # Compare to and merge with chromosome lengths/bins data from snm3c
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

# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================



# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================

def load_and_infer(dir_matrix2d, mcool_file, resolution, normalize_snm3c=True, outdir=None, name=None,
                   intramol_only=False, scale_snm3c_by=1, infer_nu=False, infer_q=False, infer_a=False,
                   obj_type='pearson', constraint_opt=0, constraint_penalty=0, constraint_min_n=None,
                   weights_exponent=None, init=None, bounds=None, seed=0,
                   max_iter=1e20, factr=1e7, maxls=20, pgtol=1e-05, jitted=True):
    df_input = pd.Series(dict(
        resolution=resolution, normalize_snm3c=normalize_snm3c,
        seed=seed, obj_type=obj_type, infer_nu=infer_nu, weights_exponent=weights_exponent,
        constraint_penalty=constraint_penalty, constraint_opt=constraint_opt,
        constraint_min_n=constraint_min_n, intramol_only=intramol_only, factr=factr, pgtol=pgtol,
        maxls=maxls, max_iter=max_iter, init=init, bounds=bounds)).rename('value').to_frame()

    data_outdir = outdir
    if data_outdir is not None and (not re.search(r'(.+/|^)nobackup(/.*|$)', data_outdir)):
        data_outdir = os.path.join(data_outdir, 'nobackup')
    if name is None:
        name = re.sub(
            r'(^|.*/)cluster(?:\.LINK){0,1}/([^/]+)(/.*|$)',
            r'\2', os.path.dirname(dir_matrix2d))
    snm3c_type = f"{'raw' if '.raw.' in mcool_file.lower() else 'q'}_snm3c"
    name = f"{name}.{snm3c_type}"

    data = prep_data_for_inference(
        dir_matrix2d, mcool_file=mcool_file, resolution=resolution,
        normalize_snm3c=normalize_snm3c, scale_snm3c_by=scale_snm3c_by,
        constraint_opt=constraint_opt, constraint_penalty=constraint_penalty,
        constraint_min_n=constraint_min_n, weights_exponent=weights_exponent,
        data_outdir=data_outdir, name=name, verbose=True)
    assert isinstance(data.snm3c, np.ndarray) and data.snm3c.size
    assert isinstance(data.dis, np.ndarray) and data.dis.size

    if not jitted:
        _setup_jax(traceback=False, debug_nan_inf=True)

    df_input['type'] = 'input'
    print("\nINFERRING WITH:\n" + df_input.value.to_string(float_format=lambda x: f"{x:g}") + "\n", flush=True)
    d = estimate_logistic_param(
        data=data, intramol_only=intramol_only, infer_nu=infer_nu, infer_q=infer_q,
        infer_a=infer_a, obj_type=obj_type, init=init, bounds=bounds, seed=seed, max_iter=max_iter,
        jitted=jitted, factr=factr, maxls=maxls, pgtol=pgtol, verbose=2)
    print(d['params'], flush=True)

    d['grad'] = d['grad'].max()
    d['params'] = d['params'].tolist()
    d['scaling_fact_mean'] = d['scaling_factor'].mean()
    df_res = pd.Series(d).rename('value').drop(['scaling_factor', 'params']).to_frame()
    df_res['type'] = 'results'

    df = pd.concat([df_input, df_res]).reset_index().rename(
        {'index': 'key'}, axis=1)[['type', 'key', 'value']]
    print("\nRESULTS:\n" + df.to_string(
        index=False, header=False, float_format=lambda x: f"{x:g}"), flush=True)


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
