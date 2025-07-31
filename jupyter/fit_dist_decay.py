import pandas as pd
import glob
import os
import ast
import numpy as np
from cooltools.sandbox.expected_smoothing import log_smooth

import matplotlib.pyplot as plt
from matplotlib import colors
# %matplotlib inline
plt.style.use('seaborn-v0_8-poster')
import seaborn as sns

import jax.numpy as jnp
from jax import grad
from scipy import optimize
from estimate_alpha import print_alpha_infer_results



# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================

def agg_across_genomic_dist(df, min_n=None):
    data_sum = df.sum(axis=0)
    data_sum.name = 'data'
    n_detected_sum = df.notnull().sum(axis=0)
    n_detected_sum.name = 'n'
    if min_n is not None and min_n > 1:
        nan_mask = n_detected_sum < min_n
        data_sum[nan_mask] = 0
        n_detected_sum[nan_mask] = 0
    res = pd.concat([
        data_sum.to_frame().T, n_detected_sum.to_frame().T])
    return res


def get_dist3d_by_genomic_dist(sc_dis_intramol, min_n=None, dis_exp=-1):
    sc_invdis_intramol = sc_dis_intramol.pow(dis_exp)
    sc_invdis_intramol.index = [j - i for i, j in sc_invdis_intramol.index]
    sc_invdis_intramol.columns.name = 'cell_num'
    sc_invdis_intramol.sort_index(inplace=True)
    s_by_cell = sc_invdis_intramol.groupby(level=0).apply(agg_across_genomic_dist, min_n=min_n)
    s_by_cell.index.names = ['genomic_dis', 'desc']
    return s_by_cell


def get_counts_by_genomic_dist(matrix_df_ambig, min_n=None, counts_col='pc_mean'):
    counts_intramol = matrix_df_ambig.rename(
        {'genomic_dis_ambig': 'genomic_dis'}, axis=1, errors='ignore').set_index(
        'genomic_dis')[[counts_col]]
    counts_intramol.columns.name = 'cell_num'
    counts_intramol.sort_index(inplace=True)
    counts_by_s = counts_intramol.groupby(level=0).apply(agg_across_genomic_dist, min_n=min_n)
    counts_by_s.index.names = ['genomic_dis', 'desc']
    return counts_by_s


def smooth_frequency(df_freq, sigma_log10=0.1):
    df_freq_smooth = []
    
    for col in df_freq.columns:
        cell_data = df_freq[col].unstack()
        cell_data.columns.name = None

        mask = cell_data.n > 0
        if sigma_log10 is None or not sigma_log10:
            cell_data['smoothed_data'] = cell_data.data
            cell_data['smoothed_n'] = cell_data.n
        else:
            smoothed_data, smoothed_n = log_smooth(
                cell_data[mask].index.values.astype(np.float64),
                cell_data.loc[mask, ['data', 'n']].values.T,
                sigma_log10=sigma_log10,
                window_sigma=5, points_per_sigma=10)
            cell_data.loc[mask, 'smoothed_data'] = smoothed_data
            cell_data.loc[mask, 'smoothed_n'] = smoothed_n

        cell_data['smoothed_freq'] = np.nan
        mask_sm = mask & cell_data.smoothed_n != 0
        cell_data.loc[mask_sm, 'smoothed_freq'] = cell_data.loc[mask_sm, 'smoothed_data'] / cell_data.loc[mask_sm, 'smoothed_n']
        cell_data['cell_num'] = col
        cell_data['n'] = cell_data['n'].astype(int)
    
        df_freq_smooth.append(cell_data)
    
    df_freq_smooth = pd.concat(df_freq_smooth)
    return df_freq_smooth


def estimate_exponent_manual(s_by_cell_smooth_agg, x0=None, bounds=(-10, 10), seed=0,
                             min_genomic_dis=None, max_genomic_dis='default',
                             max_iter=1e20, resolution_bp=2.5e6, verbose=True, plot=True, as_counts=False, dis_exp=-1):

    dist_decay = s_by_cell_smooth_agg.smoothed_freq.copy()
    dist_decay.index *= int(resolution_bp)
    
    # Build args
    if min_genomic_dis is None:
        min_genomic_dis = dist_decay.index.min()
    if max_genomic_dis is None:
        max_genomic_dis = dist_decay.index.max()
    elif isinstance(max_genomic_dis, str) and max_genomic_dis.lower() == 'default':
        max_genomic_dis = 6e7
    mask = (dist_decay.index <= max_genomic_dis) & (dist_decay.index >= min_genomic_dis)
    genomic_dis = dist_decay[mask].index.values
    freq_smooth = dist_decay[mask].values.ravel()

    d = estimate_exponent(
        genomic_dis, freq_smooth=freq_smooth, x0=x0, bounds=bounds, seed=seed,
        max_iter=max_iter, verbose=verbose > 1)

    x = genomic_dis
    y = d['scaling_factor'] * np.power(x, d['exponent'])

    exponent = d['exponent']
    est_alpha = None
    if not as_counts:
        exponent = exponent / dis_exp
        est_alpha = -0.8 / exponent
        if verbose:
            if verbose > 1:
                print(flush=True)
            print(f"v = {exponent:.3g}")
            print(f"1/v = {1 / exponent:.3g}")
            print(f"est alpha = B/v = {est_alpha:.3g}")

    if plot:
        if as_counts:
            ylabel = 'Counts'
            line_label = r"$c \sim s^{" + f"{exponent:.3g}" + r"}$"
        else:
            ylabel = 'Inverse of 3D\n' + r'distances $(d^{' + f"{dis_exp:g}" '})$'
            line_label = r"$d \sim s^{" + f"{exponent:.3g}" + r"}$"
        
        f, ax = plt.subplots(figsize=[6.4, 4.8])
        ax.loglog(
            s_by_cell_smooth_agg.index.values * 2.5e6,
            s_by_cell_smooth_agg.smoothed_freq.values,
            label='MERFISH data')
        ax.set(xlabel='Separation, bp ($s$)',  ylabel=ylabel)
        ax.set_aspect(1.0)
        ax.grid(lw=0.5)
        ax.loglog(x, y, color='black', label=line_label)

        # ax.set_ylim()
        ax.set_title(f"")
        plt.legend()
        f.tight_layout()
        plt.show()

    return d, (x, y), est_alpha

# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================

def estimate_scaling_factor(exponent, genomic_dis, freq_smooth):
    y = jnp.power(genomic_dis, exponent)
    scaling_factor = jnp.sum(y * freq_smooth) / jnp.sum(jnp.square(y))
    return scaling_factor


def fit_exponent_obj(exponent, genomic_dis, freq_smooth, scaling_factor):
    y = scaling_factor * jnp.power(genomic_dis, exponent)
    mse = jnp.mean(jnp.square(freq_smooth - y))
    return mse


fit_exponent_grad = grad(fit_exponent_obj)


def obj_wrap(exponent, genomic_dis, freq_smooth):
    scaling_factor = estimate_scaling_factor(exponent, genomic_dis=genomic_dis, freq_smooth=freq_smooth)
    obj = fit_exponent_obj(exponent, genomic_dis=genomic_dis, freq_smooth=freq_smooth, scaling_factor=scaling_factor)
    if not isinstance(obj, float) and obj.size > 1:
        obj = sum(obj)
    return obj


def grad_wrap(exponent, genomic_dis, freq_smooth):
    scaling_factor = estimate_scaling_factor(exponent, genomic_dis=genomic_dis, freq_smooth=freq_smooth)
    return np.array(fit_exponent_grad(exponent, genomic_dis=genomic_dis, freq_smooth=freq_smooth, scaling_factor=scaling_factor)).ravel()


def estimate_exponent(genomic_dis, freq_smooth, x0=None, bounds=(-10, 10), seed=0,
                      max_iter=1e20, verbose=True):
    # Define initial value (x0) and bounds on X
    if x0 is None:
        rng = np.random.default_rng(seed=seed)
        x0 = rng.uniform(low=bounds[0], high=bounds[1])
    x0 = np.array(x0, ndmin=1, copy=None)
    bounds = np.array(bounds, ndmin=2, copy=None).reshape(1, -1)

    # Build args
    args = [genomic_dis, freq_smooth]

    # Optimize
    if verbose:
        print("OPTIMIZING:", flush=True)
    obj = obj_wrap(x0, *args)
    if max_iter == 0:
        X = x0
        d = {'converged': 'N/A', 'obj': obj._value}
    else:
        results = optimize.fmin_l_bfgs_b(
            obj_wrap, x0=x0, fprime=grad_wrap, bounds=bounds, args=args,
            factr=1e-20, maxls=40, pgtol=1e-10, maxiter=int(max_iter)) #, maxfun=max_fun)
        X, obj, d = results
        d['converged'] = d['warnflag'] == 0
        conv_desc = d['task']
        d['obj'] = obj
    if X.size == 1:
        d['exponent'] = X[0]
    else:
        d['exponent'] = X
    d['scaling_factor'] = float(estimate_scaling_factor(X, *args)._value)

    if verbose:
        print("\nRESULTS:", flush=True)
        print_alpha_infer_results(d)

    return d
