import pandas as pd
import glob
import os
import ast
import numpy as np
import re
from functools import partial
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
from matplotlib import colors
plt.style.use('seaborn-v0_8-paper')
import seaborn as sns
sns.set_theme('paper', style='white')


def _get_pseudocounts_scaling_factor(pseudocounts, snm3c):
    scaling_factor = np.sum(pseudocounts * snm3c) / np.where(
        pseudocounts.sum() == 0, 1, np.sum(np.square(pseudocounts)))
    return scaling_factor


def rescale_snm3c(pseudocounts, snm3c, chrom, copy=True, verbose=False):
    if chrom is None:
        scaling_factor = _get_pseudocounts_scaling_factor(
            pseudocounts=pseudocounts, snm3c=snm3c)
        return snm3c / scaling_factor

    scaling_factors = pd.Series(index=[f"chr{x}" for x in np.unique(chrom)])
    if copy:
        snm3c = snm3c.copy()
    for x in np.unique(chrom):
        mask = chrom == x
        scaling_factor = _get_pseudocounts_scaling_factor(
            pseudocounts=pseudocounts[mask], snm3c=snm3c[mask])
        scaling_factors[f"chr{x}"] = scaling_factor
        snm3c[mask] /= scaling_factor
    if verbose:
        print(f"Scaling factors: (mean={scaling_factors.mean():.3g}, var={scaling_factors.var():.3g})", flush=True)
        print(scaling_factors.to_string(float_format=lambda x: f"{x:.3g}"), flush=True)
    return snm3c


def logistic(d, k, d0=0, nu=1, q=1, a=0, scale_snm3c_by=1):
    assert k < 0
    tmp = -k * (d - d0)
    log_bar = np.log1p(q * np.exp(tmp))
    log_baz = log_bar / nu
    log_counts = -log_baz
    if scale_snm3c_by is not None and scale_snm3c_by != 1:
        log_counts = log_counts + scale_snm3c_by
    counts = np.exp(log_counts)
    # counts = 1 / np.power(1 + q * np.exp(tmp), 1 / nu)
    if a != 0:
        counts = a + (1 - a) * counts
    return counts


def power(x, alpha, scale_snm3c_by=1):
    if isinstance(x, np.ndarray):
        y = np.full_like(x, np.nan)
        mask = (~np.isnan(x)) & (x > 0)
        y[mask] = np.power(x[mask], alpha)
        counts = y
    else:
        counts = x.pow(alpha)
    if scale_snm3c_by is not None and scale_snm3c_by != 1:
        counts *= scale_snm3c_by
    return counts


def thresholded(x, cutoff, scale_snm3c_by=1):
    counts = x <= cutoff
    if scale_snm3c_by is not None and scale_snm3c_by != 1:
        counts *= scale_snm3c_by
    return counts

# ===================================================================================================================


def get_pseudocounts(matrix_df, sc_dis, transfer_func, transfer_func_kwargs=None, snm3c_are_normalized=True,
                     scale_snm3c_by=1, agg_func='mean', remove_null_snm3c=True, verbose=False):
    pc_col = f"pc_{agg_func}"
    if transfer_func_kwargs is None:
        transfer_func_kwargs = {}
    if scale_snm3c_by is not None and scale_snm3c_by != 1:
        transfer_func_kwargs['scale_snm3c_by'] = scale_snm3c_by

    counts_sc = transfer_func(sc_dis, **transfer_func_kwargs)
    if agg_func.lower() == 'mean':
        counts = counts_sc.mean(axis=1).to_frame().rename({0: 'pc_mean'}, axis=1)
    else:
        counts = counts_sc.median(axis=1).to_frame().rename({0: 'pc_med'}, axis=1)
    # counts = pd.concat([
    #     counts_sc.mean(axis=1).to_frame().rename({0: 'pc_mean'}, axis=1),
    #     counts_sc.median(axis=1).to_frame().rename({0: 'pc_med'}, axis=1)], axis=1)
    matrix_df = matrix_df.join(counts, how='inner')
    if scale_snm3c_by is not None and scale_snm3c_by != 1:
        matrix_df['snm3c'] *= scale_snm3c_by
    if remove_null_snm3c:
        matrix_df = matrix_df[matrix_df.snm3c.notnull()]

    # Ambiguate matrix_df
    grp_cols = ['i.idx_ambig', 'j.idx_ambig', 'chrom', 'i.idx_chrom', 'j.idx_chrom',
                'i.idx_clr', 'j.idx_clr', 'genomic_dis_ambig', 'snm3c']
    excluded_cols = ['i.idx', 'j.idx', 'i.hmlg', 'j.hmlg', 'mask.diffM', 'mask.nghbr', 'genomic_dis']
    matrix_df_ambig = matrix_df.copy().drop(excluded_cols, axis=1)

    mask_swap = matrix_df['i.idx_ambig'] > matrix_df['j.idx_ambig']
    if mask_swap.any():  # Ambig index pair should fall in upper triangular
        for col in ('idx_ambig', 'idx_chrom', 'idx_clr'):
            matrix_df_ambig.loc[mask_swap, f'i.{col}'] = matrix_df.loc[mask_swap, f'j.{col}']
            matrix_df_ambig.loc[mask_swap, f'j.{col}'] = matrix_df.loc[mask_swap, f'i.{col}']
    matrix_df_ambig = matrix_df_ambig[  # Remove diagonal of ambig
        matrix_df_ambig['i.idx_ambig'] != matrix_df_ambig['j.idx_ambig']]
    assert (matrix_df_ambig.groupby(grp_cols).pc_mean.count() == 4).all()
    matrix_df_ambig = matrix_df_ambig.groupby(grp_cols, dropna=False).mean()
    matrix_df_ambig.reset_index(level=np.arange(len(grp_cols)).tolist(), inplace=True)

    # Optimally rescale snm3c (MLE of MSE)
    matrix_df_ambig['chromnum'] = matrix_df_ambig.chrom.str.replace('chr', '', regex=False).astype(int)
    if snm3c_are_normalized:
        # for col in ('pc_mean', 'pc_med'):  # FIXME select mean/median as arg when running the func...
        matrix_df_ambig['snm3c'] = rescale_snm3c(
            pseudocounts=matrix_df_ambig[pc_col].values,
            snm3c=matrix_df_ambig.snm3c.values, chrom=matrix_df_ambig.chromnum.values, copy=False)

    return matrix_df_ambig, matrix_df


def pearson_corr_pseudocounts(matrix_df_ambig, transfer_func, transfer_func_kwargs=None,
                              agg_func='mean', verbose=False):
    pc_col = f"pc_{agg_func}"
    if transfer_func_kwargs is None:
        transfer_func_kwargs = {}

    if transfer_func.__name__ == 'sum_logistic':
        func_kwargs_str = 'sum'
    else:
        func_kwargs_str = ', '.join([f"{k}={v:.4g}" for k, v in transfer_func_kwargs.items()])
    pearson_r = {
        'func': transfer_func.__name__,
        'func_kwargs': func_kwargs_str,
        agg_func: float(matrix_df_ambig[["snm3c", pc_col]].corr(method='pearson').values[0, 1])}
    if verbose:
        print(f"Pearson R:  {agg_func}={pearson_r[agg_func]:.3g}", flush=True)
        # print(f"Pearson R:  mean={pearson_r['mean']:.3g}\n             "
        #       f"med={pearson_r['med']:.3g}", flush=True)
    return pearson_r


def assess_pseudocounts(matrix_df, sc_dis, transfer_func, transfer_func_kwargs=None, plot=False, snm3c_are_normalized=True,
                        scale_snm3c_by=1, perc_cutoff=0.95, agg_func='mean', plot_hue=None, remove_null_snm3c=True, verbose=False):
    pc_col = f"pc_{agg_func}"
    if transfer_func_kwargs is None:
        transfer_func_kwargs = {}

    matrix_df_ambig, _ = get_pseudocounts(
        matrix_df, sc_dis=sc_dis, transfer_func=transfer_func, transfer_func_kwargs=transfer_func_kwargs,
        snm3c_are_normalized=snm3c_are_normalized, scale_snm3c_by=scale_snm3c_by, agg_func=agg_func,
        remove_null_snm3c=remove_null_snm3c, verbose=verbose)

    pearson_r = pearson_corr_pseudocounts(
        matrix_df_ambig, transfer_func=transfer_func, transfer_func_kwargs=transfer_func_kwargs,
        agg_func=agg_func, verbose=verbose)

    if plot:
        plot_counts_corr(
            matrix_df_ambig, pearson_r=pearson_r, perc_cutoff=perc_cutoff, agg_func=agg_func,
            hue=plot_hue)

    return pearson_r, (matrix_df, matrix_df_ambig)


def assess_via_genomic_dis(matrix_df, sc_dis, transfer_func, transfer_func_kwargs=None, genomic_dis_list=None,
                           snm3c_are_normalized=True, scale_snm3c_by=1, plot=False, perc_cutoff=0.95, agg_func='mean', plot_hue=None,
                           remove_null_snm3c=True, results_file=None, verbose=True):
    pc_col = f"pc_{agg_func}"
    if transfer_func_kwargs is None:
        transfer_func_kwargs = {}
    if genomic_dis_list is None:
        genomic_dis_list = [(0, np.inf)]

    results_df = pd.DataFrame()
    results = []
    if results_file is not None and os.path.exists(results_file):
        # print(results_file, flush=True)
        results_df = pd.read_csv(results_file, sep='\t')
        results = results_df.to_dict(orient='records')
    # print(f"{len(results_df)} entries:")
    # display(results_df.head())

    matrix_df_ambig, _ = get_pseudocounts(
        matrix_df, sc_dis=sc_dis, transfer_func=transfer_func, transfer_func_kwargs=transfer_func_kwargs,
        snm3c_are_normalized=snm3c_are_normalized, scale_snm3c_by=scale_snm3c_by, agg_func=agg_func,
        remove_null_snm3c=remove_null_snm3c, verbose=verbose)

    for genomic_dis in genomic_dis_list:
        if isinstance(genomic_dis, (int, float)):
            genomic_dis_min = genomic_dis_max = genomic_dis
            desc=f's={genomic_dis}'.ljust(10)
        else:
            genomic_dis_min, genomic_dis_max = genomic_dis
            desc=f'{genomic_dis_min}≤s≤{genomic_dis_max}'.ljust(10)

        mask = (matrix_df_ambig.genomic_dis_ambig >= genomic_dis_min) & (
            matrix_df_ambig.genomic_dis_ambig <= genomic_dis_max)
        res = pearson_corr_pseudocounts(
            matrix_df_ambig.loc[mask], transfer_func=transfer_func,
            transfer_func_kwargs=transfer_func_kwargs, agg_func=agg_func, verbose=verbose)
        res['genomic_dis_min'] = genomic_dis_min
        res['genomic_dis_max'] = genomic_dis_max
        if verbose:
            print(f"GENOMIC DIS: {desc}  R={res['mean']:.3g}", flush=True)

        if plot:
            plot_counts_corr(
                matrix_df_ambig.loc[mask], pearson_r=res, perc_cutoff=perc_cutoff,
                agg_func=agg_func, hue=plot_hue)

        results.append(res)
        results_df = pd.DataFrame(results).drop_duplicates().sort_values(
            'mean', ascending=False).reset_index(drop=True)
        if results_file is not None:
            results_df.to_csv(results_file, sep='\t', index=False)

    if verbose:
        print(flush=True)
        results_df_func_single = results_df[(results_df.func == res['func']) & (
            results_df.func_kwargs == res['func_kwargs']) & (
            results_df.genomic_dis_min == results_df.genomic_dis_max)]
        results_df_func_range = results_df[(results_df.func == res['func']) & (
            results_df.func_kwargs == res['func_kwargs']) & (
            results_df.genomic_dis_min != results_df.genomic_dis_max)]
        if len(results_df_func_single):
            display(results_df_func_single.head())
        if len(results_df_func_range):
            display(results_df_func_range.head())

    return results_df


def plot_counts_corr(matrix_df_ambig, perc_cutoff=0.95, agg_func='mean', pearson_r=None, title=None,
                     hue=None, alpha=0.5, verbose=False):
    xmax = matrix_df_ambig.snm3c.quantile(perc_cutoff)
    ymax = matrix_df_ambig[f"pc_{agg_func}"].quantile(perc_cutoff)
    mask = (matrix_df_ambig.snm3c <= xmax) & (matrix_df_ambig[f"pc_{agg_func}"] <= ymax)

    nhue = 0
    if hue is not None:
        nhue = matrix_df_ambig.loc[mask, hue].drop_duplicates().size
        if nhue == 1:
            hue = None

    if hue is None:
        g = sns.jointplot(
            data=matrix_df_ambig[mask], x="snm3c", y=f"pc_{agg_func}", kind='hex', height=4)
    else:
        palette = None
        if nhue > 2:
            palette = sns.diverging_palette(220, 20, s=100, l=50, center="dark", as_cmap=True)
        g = sns.jointplot(
            data=matrix_df_ambig[mask], x="snm3c", y=f"pc_{agg_func}", hue=hue, height=4,
            joint_kws={'edgecolor': None, 'alpha': alpha, 'size': 10}, xlim=(0, xmax), ylim=(0, ymax),
            palette=palette)
        sns.move_legend(g.ax_joint, "upper left", bbox_to_anchor=(1.25, 1))
    plt.ylabel(f"Pseudo-counts\n({agg_func} across cells)")
    plt.xlabel(f"snm3C-seq counts\n(ambiguous, normalized)")
    if pearson_r is not None:
        if title is None:
            title = f"{pearson_r['func']}\n({pearson_r['func_kwargs']})".replace('nu=', r'$v$=') #r'$v$=') #𝜈
        title = f"{title}\nR={pearson_r[agg_func]:.3g}"
    if title is not None:
        g.fig.suptitle(title, y=1.07)
    plt.show()

    if verbose:
        xmin = matrix_df_ambig.snm3c.quantile(1 - perc_cutoff)
        ymin = matrix_df_ambig[f"pc_{agg_func}"].quantile(1 - perc_cutoff)
        mask = (matrix_df_ambig.snm3c <= xmin) | (matrix_df_ambig[f"pc_{agg_func}"] <= ymin)
        print(matrix_df_ambig.loc[mask])

# ===================================================================================================================

def get_best_func(results_df):
    func_dict = {'logistic': logistic, 'power': power, 'thresholded': thresholded}
    func = func_dict[results_df.loc[0, 'func']]
    func_kwargs = ast.literal_eval("{'" + results_df.loc[0, 'func_kwargs'].replace(
        '=', "': ").replace(', ', ", '") + "}")
    return func, func_kwargs


def plot_best_result(matrix_df, sc_dis, results_df, snm3c_are_normalized=True, scale_snm3c_by=1, plot=True):
    func, func_kwargs = get_best_func(results_df)
    pearson_r, (matrix_df, matrix_df_ambig) = assess_pseudocounts(
        matrix_df, sc_dis, transfer_func=func, transfer_func_kwargs=func_kwargs,
        snm3c_are_normalized=snm3c_are_normalized, scale_snm3c_by=scale_snm3c_by, plot=plot)
    return pearson_r, (matrix_df, matrix_df_ambig)


def plot_transfer_func(sc_dis, func, func_kwargs=None, title=None, xmin=0, xmax=None, make_fig=True,
                       show=True, figsize=(3, 2), dpi=None, alpha=1, linewidth=None, color=None,
                       linestyle=None):
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = np.quantile(sc_dis.values[~np.isnan(sc_dis.values)], 0.99)
    x = np.linspace(xmin, xmax, num=100)
    if func_kwargs is None:
        y = func(x)
    else:
        y = func(x, **func_kwargs)

    if func.__name__ == 'sum_logistic':
        info = func.__name__ 
    else:
        info = [func.__name__, '(' + ', '.join([f"{k}={v:.4g}".replace(
            'nu=', r'$v$=') for k, v in func_kwargs.items()]) + ')']

    if make_fig:
        fig = plt.figure()
        if dpi is not None:
            fig.set_dpi(dpi)
        if figsize is not None:
            fig.set_size_inches(figsize)
    plt.plot(x, y, label=' '.join(info), alpha=alpha, linewidth=linewidth,
             color=color, linestyle=linestyle)
    plt.xlim(xmin, xmax)
    if title is None:
        title = '\n'.join(info)
    plt.title(title)
    plt.xlabel('Single-cell 3D distances')
    plt.ylabel('Pseudo-counts')
    if show:
        plt.show()
    elif make_fig:
        return fig


def plot_transfer_func_multiple(sc_dis, kwargs_per_line, title=None, figsize=(6, 4), dpi=None,
                                linewidth=2, xmin=0, xmax=2.5):
    make_fig = True
    for kwargs in kwargs_per_line:
        if 'color' not in kwargs:
            kwargs['color'] = None
        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = None
        plot_transfer_func(
            sc_dis, func=kwargs['func'], func_kwargs=kwargs['func_kwargs'], title="Transfer functions",
            xmin=xmin, xmax=xmax, make_fig=make_fig, show=False, figsize=figsize, dpi=dpi, alpha=0.5,
            linewidth=linewidth, color=kwargs['color'], linestyle=kwargs['linestyle'])
        make_fig = False
    plt.legend()
    plt.show()


def plot_best_transfer_func(results_df, sc_dis):
    func, func_kwargs = get_best_func(results_df)
    plot_transfer_func(sc_dis=sc_dis, func=func, func_kwargs=func_kwargs)

# ===================================================================================================================

def get_lower_outliers_loci(matrix_df_ambig, perc_cutoff=0.01, xmax=None, ymax=None, agg_func='mean'):
    if xmax is None:
        xmax = matrix_df_ambig.snm3c.quantile(perc_cutoff)
    if ymax is None:
        ymax = matrix_df_ambig[f"pc_{agg_func}"].quantile(perc_cutoff)
    mask = (matrix_df_ambig.snm3c <= xmax) | (matrix_df_ambig[f"pc_{agg_func}"] <= ymax)
    return matrix_df_ambig.loc[mask]


def assess_lower_outliers_loci(matrix_df_ambig, lengths_df, perc_cutoff=0.01, ymax=None, agg_func='mean',
                              lengths_df_outliers=None, verbose=False, plot=True):
    outliers = get_lower_outliers_loci(
        matrix_df_ambig, perc_cutoff=perc_cutoff, xmax=0, ymax=ymax,
        agg_func=agg_func).sort_values(f'pc_{agg_func}')

    if lengths_df_outliers is None:
        lengths_df_outliers = lengths_df[['idx_genome', 'chrom', 'idx_chrom', 'has_data', 'cell_cov_min', 'cell_cov_h1', 'cell_cov_h2']].copy()
        lengths_df_outliers['cell_cov_mean'] = (lengths_df_outliers['cell_cov_h1'] + lengths_df_outliers['cell_cov_h2']) / 2
    
    lengths_df_outliers[perc_cutoff] = 0
    lengths_df_outliers.loc[lengths_df_outliers.idx_genome.isin(outliers['i.idx_ambig']), perc_cutoff] += 1
    lengths_df_outliers.loc[lengths_df_outliers.idx_genome.isin(outliers['j.idx_ambig']), perc_cutoff] += 1
    assert (lengths_df_outliers.loc[lengths_df_outliers[perc_cutoff] > 0, 'has_data']).all()
    mask = lengths_df_outliers[perc_cutoff] > 0

    pvals = {}
    
    for col in ['cell_cov_min', 'cell_cov_mean']:
        mask0 = lengths_df_outliers[col].notnull() & (lengths_df_outliers[perc_cutoff] == 0)
        tmp = []
        nsamp_per_cat = lengths_df_outliers.loc[
            lengths_df_outliers[col].notnull(), perc_cutoff].value_counts().drop(0)
        nsamp_per_cat = nsamp_per_cat[nsamp_per_cat > 2]
        for i in nsamp_per_cat.index:
            mask = lengths_df_outliers[col].notnull() & (lengths_df_outliers[perc_cutoff] == i)
            pval = ttest_ind(
                lengths_df_outliers.loc[mask0, col].values,
                lengths_df_outliers.loc[mask, col].values, equal_var=False).pvalue
            tmp.append(f"0vs{i}: {pval * nsamp_per_cat.size:.3g}")
        pvals[col] = '\n'.join(tmp)
    
    if verbose:
        outlier_mean = lengths_df_outliers.loc[mask, ['cell_cov_min', 'cell_cov_mean']].describe()
        other_mean = lengths_df_outliers.loc[~mask, ['cell_cov_min', 'cell_cov_mean']].describe()
        desc_df = outlier_mean.join(other_mean, lsuffix='.LOW', rsuffix='.OTHER')
        desc_df = desc_df[['cell_cov_min.LOW', 'cell_cov_min.OTHER', 'cell_cov_mean.LOW', 'cell_cov_mean.OTHER']].rename(
            {c: c.replace('cell_cov_', '') for c in desc_df.columns}, axis=1)
        desc_df.rename({c: f"{c.split('.')[1]}_{c.split('.')[0]}" for c in desc_df.columns}, axis=1, inplace=True)
        display(desc_df.round(2))

    if plot:
        fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(8, 4), sharey=True)
        for i, col in enumerate(['cell_cov_min', 'cell_cov_mean']):
            sns.histplot(lengths_df_outliers, x=col, hue=perc_cutoff, ax=ax[i])
            ax[i].set_title(pvals[col])
        if ymax is None:
            suptitle = f"{perc_cutoff * 100:g}%"
            ymax = matrix_df_ambig[f"pc_{agg_func}"].quantile(perc_cutoff)
            suptitle += f" ({ymax:.2g})"
        else:
            suptitle = f"{ymax:.2g}"
        fig.suptitle("pseudo-counts ≤ " + suptitle)
        plt.show()

    return outliers, lengths_df_outliers

# ===================================================================================================================

