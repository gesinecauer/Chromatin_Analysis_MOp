import numpy as np
import pandas as pd
import os
import re

import seaborn as sns
import matplotlib.pyplot as plt

import jax.numpy as ag_np
from jax import grad
from scipy import optimize
from pastis.optimization.likelihoods import poisson_nll
from topsy.analysis.compare_distances import make_matrix_df



def plot_counts_vs_dis(matrix_df, mask, alpha, beta=None, title=None, scatter_opacity=0.1, outfile=None):
    x = np.linspace(matrix_df.loc[mask, 'dis_inverse'].min(), matrix_df.loc[mask, 'dis_inverse'].max(), 100)
    if beta is None:
        beta = matrix_df.counts.sum() / np.sum(np.power(matrix_df.dis, alpha))
    if verbose:
        print(f"β={beta:.3g}", flush=True)
    y = beta * np.power(x, alpha)  # Pseudo-counts

    # if integer_counts_nreads is not None:
    #     matrix_df["counts_int"] = (matrix_df.counts * integer_counts_nreads / matrix_df.counts.sum()).round()
    #     y_col = "counts_int"
    # else:
    #     y_col = "counts"

    sns.jointplot(data=matrix_df[mask], x='dis_inverse', y=y_col, kind='hex')
    sns.scatterplot(
        data=matrix_df[mask & (matrix_df.counts > matrix_df.counts.quantile(0.999))],
        x='dis_inverse', y=y_col, linewidths=0, alpha=scatter_opacity, s=10)
    plt.plot(x, y, color='red')
    plt.xlabel("Inverse distances, $d_{ij}^{-1}$")
    plt.ylabel("Pseudo-counts, $c_{ij}$")
    if title is not None:
        plt.suptitle(title)

    if outfile is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile)
    plt.clf()


def estimate_alphas_from_true_dis(lengths_df, dis_matrix, counts_matrix, nonmissing_matrix, nonmissing_percentile=0.5):
    matrix_df = make_matrix_df(
        lengths_df=lengths_df,
        matrix_dict={'dis': dis_matrix, 'counts': counts_matrix, 'nonmissing': nonmissing_matrix})

    matrix_df['genomic_dis'] = None
    matrix_df.loc[matrix_df['mask.sameC-sameH'], 'genomic_dis'] = (matrix_df['i.idx'] - matrix_df['j.idx']).abs()
    matrix_df['mask.nghbr'] = matrix_df['mask.sameC-sameH'] & (matrix_df.genomic_dis == 1)

    matrix_df['dis'] /= matrix_df.loc[matrix_df['mask.nghbr'], 'dis'].mean()  # Scale distances
    matrix_df['dis_inverse'] = matrix_df.dis.pow(-1)

    # Remove loci that are entirely missing in sc dataset
    n = len(lengths_df)  # n = nbeads / ploidy
    excluded_loci = np.where((nonmissing_matrix == 0).all(axis=0))[0]
    excluded_loci[excluded_loci >= n] -= n
    excluded_loci = np.unique(excluded_loci)  # If excluded in one hmlg, exclude in both
    mask = (~matrix_df['i.idx_ambig'].isin(excluded_loci)) & (~matrix_df['j.idx_ambig'].isin(excluded_loci))
    matrix_df = matrix_df[mask]

    # Filter locus pairs
    # matrix_df = matrix_df[matrix_df.counts != 0]
    # matrix_df = matrix_df[matrix_df.nonmissing > 0]
    if nonmissing_percentile is not None:
        matrix_df = matrix_df[matrix_df.nonmissing > matrix_df.nonmissing.quantile(nonmissing_percentile)]

    return matrix_df


# ===================================================================================================================
# ===================================================================================================================


def estimate_beta(alphas, counts, dis, intramol_mask, use_poisson=False, mods=[], verbose=False):
    if alphas.size == 1:
        beta = ag_np.sum(counts) / ag_np.sum(ag_np.power(dis, alphas))
        return beta
    alpha_intra, alpha_inter = alphas
    dis_alpha_intra = ag_np.sum(ag_np.power(dis[intramol_mask], alpha_intra))
    if 'beta_from_intra_only' in mods:
        beta = ag_np.sum(counts[intramol_mask]) / dis_alpha_intra
    else:
        dis_alpha_inter = ag_np.sum(ag_np.power(dis[~intramol_mask], alpha_inter))
        beta = ag_np.sum(counts) / (dis_alpha_intra + dis_alpha_inter)
    if verbose:
        if verbose > 1 or not np.isclose(alpha_intra, -2, atol=1e-3):
            print(f"\nβ={beta:g}\tINTRA={dis_alpha_intra:.3g}\tinter={dis_alpha_inter:.3g}")
    return beta


def fit_alpha_obj(alphas, counts, dis, intramol_mask, use_poisson=False, mods=[], verbose=False):
    beta = estimate_beta(alphas, counts=counts, dis=dis, intramol_mask=intramol_mask, mods=mods)
    
    if alphas.size == 1:
        lambda_pois = beta * ag_np.power(dis, alphas)
        if use_poisson:
            obj = poisson_nll(counts, lambda_pois=lambda_pois)
        else:
            obj = ag_np.mean(ag_np.square(lambda_pois - counts))
        if verbose:
            print(f"α={alphas[0]:.3f}\tβ={beta.round():.3g}\tobj={obj.round():g}", flush=True)
        return obj
        
    alpha_intra, alpha_inter = alphas

    lambda_pois_intra = beta * ag_np.power(dis[intramol_mask], alpha_intra)
    if use_poisson:
        obj_intra = poisson_nll(counts[intramol_mask], lambda_pois=lambda_pois_intra)
    else:
        obj_intra = ag_np.square(lambda_pois_intra - counts[intramol_mask])

    if 'alpha_from_intra_only' in mods:
        obj = ag_np.mean(obj_intra)
    else:
        lambda_pois_inter = beta * ag_np.power(dis[~intramol_mask], alpha_inter)
        if use_poisson:
            obj_inter = poisson_nll(counts[~intramol_mask], lambda_pois=lambda_pois_inter)
        else:
            obj_inter = ag_np.square(lambda_pois_inter - counts[~intramol_mask])

        if 'weight_equally' in mods:
            obj = ag_np.mean(obj_intra) + ag_np.mean(obj_inter)
        elif any([isinstance(x, (float, int)) for x in mods]):
            obj = ag_np.mean(obj_intra) + ag_np.mean(obj_inter) * [x for x in mods if isinstance(x, (float, int))][0]
        else:
            obj = (ag_np.sum(obj_intra) + ag_np.sum(obj_inter)) / counts.size

    if verbose:
        if verbose == 1 and np.isclose(alpha_intra, -2, atol=1e-3):
            print('.', flush=True)
        else:
            print(f"αINTRA={alpha_intra:.3f}\tαinter={alpha_inter:.3f}\tINTRA={obj_intra.sum().round():g}\tinter={obj_inter.sum().round():g}", flush=True)

    return obj


fit_alpha_grad = grad(fit_alpha_obj)


def obj_wrap(alphas, counts, dis, intramol_mask, use_poisson=False, mods=[], verbose=True):
    obj = fit_alpha_obj(alphas, counts=counts, dis=dis, intramol_mask=intramol_mask, use_poisson=use_poisson, mods=mods, verbose=verbose)
    if not isinstance(obj, float) and obj.size > 1:
        obj = sum(obj)
    return obj * 1


def grad_wrap(alphas, counts, dis, intramol_mask, use_poisson=False, mods=[], verbose=False):
    return np.array(fit_alpha_obj(alphas, counts=counts, dis=dis, intramol_mask=intramol_mask, use_poisson=use_poisson, mods=mods)).ravel()


def estimate_alpha(matrix_df, x0=None, bounds=(-6, -1), use_poisson=False, mods=[], seed=0, verbose=False):

    if isinstance(mods, str):
        mods = [mods]
    if 'alpha_from_intra_only' in mods:
        num_alphas = 1
    else:
        num_alphas = len(matrix_df['mask.sameC-sameH'].drop_duplicates())

    if x0 is None:
        rng = np.random.default_rng(seed=seed)
        x0 = rng.uniform(low=bounds[0], high=bounds[1], size=num_alphas)
    if isinstance(x0, (int, float)):
        x0 = np.repeat(x0, num_alphas)
    x0 = np.asarray(x0)
    bounds = np.repeat(np.asarray(bounds).reshape(1, -1), num_alphas, axis=0)

    counts = matrix_df.counts.values
    args = [counts, matrix_df[f"dis"].values, matrix_df['mask.sameC-sameH'].values, use_poisson, mods]

    if verbose:
        print("OPTIMIZING:", flush=True)
    obj_wrap(x0, *args, verbose=2 if verbose else 0)
    results = optimize.fmin_l_bfgs_b(
        obj_wrap, x0=x0, fprime=grad_wrap, bounds=bounds, args=[*args, verbose],
        factr=1e-60, maxls=10000, pgtol=1e-10) #, maxiter=max_iter, maxfun=max_fun, pgtol=pgtol, factr=factr)
    X, obj, d = results
    d['converged'] = d['warnflag'] == 0
    conv_desc = d['task']
    d['obj'] = obj
    if X.size == 1:
        d['alphas'] = X[0]
    else:
        d['alphas'] = X
    d['beta'] = estimate_beta(X, *args)._value
    if verbose:
        print("\nRESULTS:", flush=True)
        print_est_alpha_results(d)
    return d


def print_est_alpha_results(d):
    for k, v in d.items():
        if k == 'grad':
            v = f"{v.mean():.3g}"
        if k == 'alphas' and v.size == 2:
            v = f"INTRA={v[0]:.3g}, inter={v[1]:.3g}"
        elif isinstance(v, float):
            v = f"{v:.3g}"
        elif isinstance(v, np.ndarray):
            v = ', '.join([f"{x:.3g}" for x in v])
        print(f"{k.ljust(10)}   {v}", flush=True)
