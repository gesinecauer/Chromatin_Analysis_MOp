import numpy as np
import pandas as pd
from scipy import interpolate
from process_loci import get_index_of_loci


def interp_molecule(df, kind='linear'):    
    df.sort_values('idx_chrom', inplace=True)

    df_chosen_idx = df.loc[df.chosen_loci, 'idx_chrom']
    if len(df_chosen_idx) == df_chosen_idx.max() + 1 - df_chosen_idx.min():
        df = df.loc[df.chosen_loci, ['x', 'y', 'z', 'idx_chrom', 'idx_genome', 'chosen_loci']]
        return df

    idx_avail = df.idx_chrom.values - df.idx_chrom.min()
    length = idx_avail.max() + 1
    idx_full = np.arange(0, length)
    coords = np.full((length, 3), np.nan)
    coords[idx_avail] = df[['x', 'y', 'z']].values
    x, y, z = df[['x', 'y', 'z']].values.T
    if kind == 'rbf':
        f_x = interpolate.Rbf(idx_avail, x, smooth=0)
        f_y = interpolate.Rbf(idx_avail, y, smooth=0)
        f_z = interpolate.Rbf(idx_avail, z, smooth=0)
    else:
        f_x = interpolate.interp1d(idx_avail, x, kind=kind)
        f_y = interpolate.interp1d(idx_avail, y, kind=kind)
        f_z = interpolate.interp1d(idx_avail, z, kind=kind)
    mask = np.full(length, False)
    mask[idx_avail] = True
    idx_nan = idx_full[~mask]
    coords[idx_nan, 0] = f_x(idx_nan)
    coords[idx_nan, 1] = f_y(idx_nan)
    coords[idx_nan, 2] = f_z(idx_nan)
    df_new = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    df_new['idx_chrom'] = idx_full + df.idx_chrom.min()
    df_new['idx_genome'] = idx_full + df.idx_genome.min()
    df_new['chosen_loci'] = True
    return df_new


def interp_all(df, spacing=2.5, interp_kind='linear', verbose=True):
    if verbose:
        print("\nINTERPOLATING ALL MOLECULES:", flush=True)
        print(f"\tINITIAL number of eligible entries: {df.chosen_loci.sum():,}", flush=True)

    df = get_index_of_loci(df, spacing=spacing).sort_values(['trace_id', 'idx_genome'])
    cols = ['cell_id', 'chrom', 'trace_id', 'ntraces_per_cell', 'rep_id']
    if 'hmlg' in df.columns:
        cols.append('hmlg')

    df_interp = df.groupby(cols).apply(
        interp_molecule, kind=interp_kind, include_groups=False).reset_index(level=np.arange(len(cols)).tolist())

    if verbose:
        print(f"\tFINAL number of eligible entries: {len(df_interp):,}", flush=True)

    return df_interp

    