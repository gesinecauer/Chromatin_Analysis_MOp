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


def infer_alpha_float_vs_int(matrix_df, ambiguity='ua', dis_agg_func='mean', infer_alpha_mask=None,
                             infer_alpha_mods=None, plot=True, outdir=None, num_infer=10, verbose=True):
    from process_counts import filter_matrix_df, ambiguate_matrix_df

    if outdir is None:
        outdir_fig = None
        outfile = None
    else:
        outdir_fig = os.path.join(outdir, "images")
        outfile = os.path.join(outdir, "true_distances.alpha_inference.tsv")  # FIXME

    # Prepare data
    if ambiguity.lower() == 'ambig':
        matrix_df = ambiguate_matrix_df(matrix_df)
    elif ambiguity.lower() == 'pa':
        raise NotImplemetedError("Alpha inference for partially ambig")
    matrix_df = filter_matrix_df(matrix_df, mask=infer_alpha_mask)

    if verbose:
        print("\nALPHA INFERENCE WITH 'TRUE' SINGLE-CELL DISTANCES:", flush=True)
    results = []

    # Infer alpha with float-counts
    alpha, beta, obj = estimate_alphas_from_true_dis(
        matrix_df, num_infer=num_infer, use_poisson=False, integer_counts=False,
        dis_agg_func=dis_agg_func, infer_alpha_mask=infer_alpha_mask,
        infer_alpha_mods=infer_alpha_mods, plot=plot,
        outdir_fig=outdir_fig, verbose=False)
    alpha_intra, alpha_inter = np.tile(alpha, int(2 / np.array(alpha, ndmin=1).size))
    results.append({
        'desc': 'infer_alpha.non-integer.mse', 'obj': obj, 'beta': beta,
        'alpha_intra': alpha_intra, 'alpha_inter': alpha_inter})

    if 'counts_int' not in matrix_df.columns:
        results = pd.DataFrame(results)
        if verbose:
            print(results.to_string(index=False), flush=True)
        if outfile is not None:
            results.to_csv(outfile, sep='\t', index=False)

        return alpha, beta, results

    # Eval objective & estimate beta for integer-counts, using alpha obtained above 
    res = estimate_alpha(
        matrix_df, counts_col='counts_int', x0=alpha, use_poisson=True,
        mods=infer_alpha_mods, max_iter=0, verbose=False)
    alpha_intra, alpha_inter = np.tile(res['alphas'], int(2 / np.array(res['alphas'], ndmin=1).size))
    results.append({
        'desc': 'eval_at_alpha.poisson', 'obj': res['obj'], 'beta': res['beta'],
        'alpha_intra': alpha_intra, 'alpha_inter': alpha_inter})

    # Infer alpha with integer-counts, with designated nreads
    alpha, beta, obj = estimate_alphas_from_true_dis(
        matrix_df, num_infer=num_infer, use_poisson=True, integer_counts=True,
        dis_agg_func=dis_agg_func, infer_alpha_mask=infer_alpha_mask,
        infer_alpha_mods=infer_alpha_mods, plot=plot,
        outdir_fig=outdir_fig, verbose=False)
    alpha_intra, alpha_inter = np.tile(alpha, int(2 / np.array(alpha, ndmin=1).size))
    results.append({
        'desc': 'infer_alpha.poisson', 'obj': obj, 'beta': beta,
        'alpha_intra': alpha_intra, 'alpha_inter': alpha_inter})

    results = pd.DataFrame(results)
    if verbose:
        print(results.to_string(index=False), flush=True)
    if outfile is not None:
        results.to_csv(outfile, sep='\t', index=False)

    tmp = results[results.desc.isin(['eval_at_alpha.poisson', 'infer_alpha.poisson'])]
    alpha_intra, alpha_inter, beta = tmp.loc[
        tmp.obj == tmp.obj.min(), ['alpha_intra', 'alpha_inter', 'beta']].values.ravel().tolist()

    return alpha_intra, alpha_inter, beta, results



def plot_counts_vs_dis(matrix_df, mask, alpha, counts_col, dis_col='dis_mean', beta=None,
                       title=None, scatter_opacity=0.1, outfile=None, counts_max_percentile=0.999,
                       counts_scatter_percentile=0.95):
    matrix_df['dis_inverse'] = matrix_df[dis_col].pow(-1)

    mask = mask & (matrix_df[counts_col] < matrix_df.loc[mask, counts_col].quantile(
        counts_max_percentile))
    
    x = np.linspace(matrix_df.loc[mask, 'dis_inverse'].min(), matrix_df.loc[mask, 'dis_inverse'].max(), 100)
    if beta is None:
        beta = matrix_df[counts_col].sum() / np.sum(np.power(matrix_df[dis_col], alpha))
    y = beta * np.power(x, -alpha)  # beta * dis^alpha

    sns.jointplot(data=matrix_df[mask], x='dis_inverse', y=counts_col, kind='hex')
    cutoff = matrix_df.loc[mask, counts_col].quantile(counts_scatter_percentile)
    sns.scatterplot(
        data=matrix_df[mask & (matrix_df[counts_col] > cutoff)],
        x='dis_inverse', y=counts_col, linewidths=0, alpha=scatter_opacity, s=10)
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

    matrix_df.drop('dis_inverse', axis=1, inplace=True)


def estimate_alphas_from_true_dis(matrix_df, num_infer=10, use_poisson=True, integer_counts=True,
                                  dis_agg_func='mean', infer_alpha_mask=None, infer_alpha_mods=None,
                                  plot=True, outdir_fig=None, verbose=True):
    from process_counts import filter_matrix_df

    dis_col = f'dis_{dis_agg_func}'
    if integer_counts:
        counts_col = 'counts_int'
        if verbose:
            print(f"Inferring alpha from true {dis_agg_func} distances & integer"
                  f" pseudo-counts (nreads={matrix_df[counts_col].sum()})", flush=True)
    else:
        counts_col = 'counts'
        if verbose:
            print(f"Inferring alpha from true {dis_agg_func} distances & original"
                  " (non-integer) pseudo-counts", flush=True)

    # Optinally mask data to be used for alpha inference
    data = filter_matrix_df(matrix_df, mask=infer_alpha_mask)

    results_per_infer = []
    for seed in range(num_infer): # beta_from_intra_only, alpha_from_intra_only
        results_per_infer.append(estimate_alpha(
            data, counts_col=counts_col, seed=seed, use_poisson=use_poisson,
            mods=infer_alpha_mods))
    results_per_infer = pd.DataFrame(results_per_infer)
    results_per_infer = results_per_infer[results_per_infer.converged]
    if len(results_per_infer) == 0:
        raise ValueError(f"None of the optimizations to infer alpha converged.")
    results_best = list(
        results_per_infer[results_per_infer.obj == results_per_infer.obj.min()].T.to_dict().values())[0]
    if verbose:
        print_alpha_infer_results(results_best)

    alpha_ = results_best['alphas']
    alpha_intra_, alpha_inter_ = np.tile(alpha_, int(2 / np.array(alpha_, ndmin=1).size))
    beta_ = results_best['beta']

    if plot:
        outfile_intra = outfile_inter = None
        if outdir_fig is not None:
            tmp = f"alpha_infer.dis-{dis_agg_func}_vs_{counts_col.replace('_', '-')}"
            outfile_intra = os.path.join(outdir_fig, f"{tmp}.intra-mol.png")
            outfile_inter = os.path.join(outdir_fig, f"{tmp}.inter-mol.png")
        plot_counts_vs_dis(
            matrix_df, mask=~matrix_df['mask.diffM'], counts_col=counts_col, alpha=alpha_intra_,
            beta=beta_, title=f"INTRA-molecular\nα={alpha_intra_:.3g}, β={beta_:.3g}",
            outfile=outfile_intra)
        plot_counts_vs_dis(
            matrix_df, mask=matrix_df['mask.diffM'], counts_col=counts_col, alpha=alpha_inter_,
            beta=beta_, title=f"Inter-molecular\nα={alpha_inter_:.3g}, β={beta_:.3g}",
            outfile=outfile_inter)

    return alpha_, beta_, results_best['obj']


# ===================================================================================================================
# ===================================================================================================================


def estimate_beta(alphas, counts, dis, mask_intra, use_poisson=False, mods=[], verbose=False):
    if alphas.size == 1:
        beta = ag_np.sum(counts) / ag_np.sum(ag_np.power(dis, alphas))
        return beta
    alpha_intra, alpha_inter = alphas
    dis_alpha_intra = ag_np.sum(ag_np.power(dis[mask_intra], alpha_intra))
    if 'beta_from_intra_only' in mods:
        beta = ag_np.sum(counts[mask_intra]) / dis_alpha_intra
    else:
        dis_alpha_inter = ag_np.sum(ag_np.power(dis[~mask_intra], alpha_inter))
        beta = ag_np.sum(counts) / (dis_alpha_intra + dis_alpha_inter)
    if verbose:
        if verbose > 1:
            print(f"\nβ={beta:g}\tINTRA={dis_alpha_intra:.3g}\tinter={dis_alpha_inter:.3g}")
    return beta


def fit_alpha_obj(alphas, counts, dis, mask_intra, use_poisson=False, mods=[], verbose=False):
    beta = estimate_beta(alphas, counts=counts, dis=dis, mask_intra=mask_intra, mods=mods)

    if alphas.size == 1:
        lambda_pois = beta * ag_np.power(dis, alphas)
        if len(lambda_pois.shape) > 1:
            lambda_pois = ag_np.sum(lambda_pois, axis=1)
        if use_poisson:
            obj = poisson_nll(counts, lambda_pois=lambda_pois)
        else:
            obj = ag_np.mean(ag_np.square(lambda_pois - counts))
        if verbose:
            print(f"α={alphas[0]:.3f}\tβ={beta.round():.3g}\tobj={obj.round():g}", flush=True)
        return obj

    alpha_intra, alpha_inter = alphas

    lambda_pois_intra = beta * ag_np.power(dis[mask_intra], alpha_intra)
    if len(lambda_pois_intra.shape) > 1:
        lambda_pois_intra = ag_np.sum(lambda_pois_intra, axis=1)
    if use_poisson:
        obj_intra = poisson_nll(counts[mask_intra], lambda_pois=lambda_pois_intra)
    else:
        obj_intra = ag_np.square(lambda_pois_intra - counts[mask_intra])

    if 'alpha_from_intra_only' in mods:
        obj = ag_np.mean(obj_intra)
    else:
        lambda_pois_inter = beta * ag_np.power(dis[~mask_intra], alpha_inter)
        if len(lambda_pois_inter.shape) > 1:
            lambda_pois_inter = ag_np.sum(lambda_pois_inter, axis=1)
        if use_poisson:
            obj_inter = poisson_nll(counts[~mask_intra], lambda_pois=lambda_pois_inter)
        else:
            obj_inter = ag_np.square(lambda_pois_inter - counts[~mask_intra])

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


def obj_wrap(alphas, counts, dis, mask_intra, use_poisson=False, mods=[], verbose=True):
    obj = fit_alpha_obj(alphas, counts=counts, dis=dis, mask_intra=mask_intra, use_poisson=use_poisson, mods=mods, verbose=verbose)
    if not isinstance(obj, float) and obj.size > 1:
        obj = sum(obj)
    return obj * 1


def grad_wrap(alphas, counts, dis, mask_intra, use_poisson=False, mods=[], verbose=False):
    return np.array(fit_alpha_obj(alphas, counts=counts, dis=dis, mask_intra=mask_intra, use_poisson=use_poisson, mods=mods)).ravel()


def estimate_alpha(matrix_df, counts_col, dis_col='dis_mean', x0=None, bounds=(-6, -1),
                   use_poisson=False, mods=[], seed=0, max_iter=1e20, verbose=False):
    if 'mask.sameC' in matrix_df.columns:
        mask_intra_col = 'mask.sameC'
    else:
        mask_intra_col = 'mask.sameC-sameH'
    if not (matrix_df[counts_col].astype(int) == matrix_df[counts_col]).all():
        use_poisson = False  # Can't use Poisson-based obj for non-integer data

    # Set up any specified modifications
    if mods is None:
        mods = []
    if isinstance(mods, str):
        mods = [mods]
    if 'mask_intra' in mods:
        matrix_df = matrix_df[matrix_df[mask_intra_col]]
    if 'mask_inter' in mods:
        matrix_df = matrix_df[~matrix_df[mask_intra_col]]
    if 'alpha_from_intra_only' in mods:
        num_alphas = 1
    else:
        num_alphas = len(matrix_df[mask_intra_col].drop_duplicates())

    # Define initial value (x0) and bounds on X
    if x0 is None:
        rng = np.random.default_rng(seed=seed)
        x0 = rng.uniform(low=bounds[0], high=bounds[1], size=num_alphas)
    if isinstance(x0, (int, float)):
        x0 = np.repeat(x0, num_alphas)
    x0 = np.asarray(x0)
    bounds = np.repeat(np.asarray(bounds).reshape(1, -1), num_alphas, axis=0)

    # Build args
    counts = matrix_df[counts_col].values
    mask_intra = matrix_df[mask_intra_col].values
    dis = matrix_df[dis_col].values
    if isinstance(dis[0], (np.ndarray, list)):
        dis = np.stack(dis.tolist())
    args = [counts, dis, mask_intra, use_poisson, mods]

    # Optimize
    if verbose:
        print("OPTIMIZING:", flush=True)
    obj = obj_wrap(x0, *args, verbose=2 if verbose else 0)
    if max_iter == 0:
        X = x0
        d = {'converged': 'N/A', 'obj': obj._value}
    else:
        results = optimize.fmin_l_bfgs_b(
            obj_wrap, x0=x0, fprime=grad_wrap, bounds=bounds, args=[*args, verbose],
            factr=1e-60, maxls=10000, pgtol=1e-10, maxiter=int(max_iter)) #, maxfun=max_fun, pgtol=pgtol, factr=factr)
        X, obj, d = results
        d['converged'] = d['warnflag'] == 0
        conv_desc = d['task']
        d['obj'] = obj
    if X.size == 1:
        d['alphas'] = X[0]
    else:
        d['alphas'] = X
    d['beta'] = float(estimate_beta(X, *args)._value)
    

    if verbose:
        print("\nRESULTS:", flush=True)
        print_alpha_infer_results(d)

    return d


def print_alpha_infer_results(d):
    for k, v in d.items():
        if k == 'grad':
            v = f"{v.mean():.3g}"
        if k == 'alphas' and v.size == 2:
            v = f"INTRA={v[0]:.3g}, inter={v[1]:.3g}"
        elif isinstance(v, float):
            v = f"{v:.3g}"
        elif isinstance(v, np.ndarray) and v.size > 1:
            v = ', '.join([f"{x:.3g}" for x in v])
        print(f"{k.ljust(10)}   {v}", flush=True)
