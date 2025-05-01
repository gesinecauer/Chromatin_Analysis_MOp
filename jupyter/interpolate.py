import numpy as np
import pandas as pd
from scipy import interpolate
from process_loci import get_index_of_loci


def interp_molecule(df, kind='linear', spacing=2.5):    
    df.sort_values('idx_chrom', inplace=True)

    df_chosen_idx = df.loc[df.chosen_loci, 'idx_chrom']
    if len(df_chosen_idx) == df_chosen_idx.max() + 1 - df_chosen_idx.min():
        df = df.loc[df.chosen_loci, ['x', 'y', 'z', 'idx_chrom', 'idx_genome', 'chosen_loci', 'chrom_start', 'chrom_end']]
        return df

    # Interpolate coords
    idx_avail = df.idx_chrom.values - df.idx_chrom.min()
    length = idx_avail.max() + 1
    idx_full = np.arange(0, length)
    x, y, z = df[['x', 'y', 'z']].values.T
    if kind == 'rbf':
        f_x = interpolate.Rbf(idx_avail, x, smooth=0)
        f_y = interpolate.Rbf(idx_avail, y, smooth=0)
        f_z = interpolate.Rbf(idx_avail, z, smooth=0)
    else:
        f_x = interpolate.interp1d(idx_avail, x, kind=kind)
        f_y = interpolate.interp1d(idx_avail, y, kind=kind)
        f_z = interpolate.interp1d(idx_avail, z, kind=kind)
    df_new = pd.DataFrame()
    df_new['x'] = f_x(idx_full)
    df_new['y'] = f_y(idx_full)
    df_new['z'] = f_z(idx_full)
    df_new['idx_chrom'] = idx_full + df.idx_chrom.min()
    df_new['idx_genome'] = idx_full + df.idx_genome.min()
    df_new['chosen_loci'] = True

    # Interpolate chrom start/mid/end positions
    f_mid = interpolate.interp1d(idx_avail, df.mid.values, kind='linear')
    mid_new = np.round(f_mid(idx_full)).astype(int)
    df_new['chrom_start'] = mid_new - int(spacing * 1e6 / 2)
    df_new['chrom_end'] = mid_new + int(spacing * 1e6 / 2)
    
    return df_new


def interp_all(df, spacing=2.5, interp_kind='linear', verbose=True):
    if verbose:
        print("\nINTERPOLATING ALL MOLECULES:", flush=True)
        print(f"\tINITIAL number of eligible entries: {df.chosen_loci.sum():,}", flush=True)

    # 'Correcting' chrom start/mid/end positions
    df['mid'] = df[['chrom_start', 'chrom_end']].mean(axis=1)
    df['offset'] = df.mid % (spacing * 1e6)
    offset_median = int(round(df.loc[df.chosen_loci, 'offset'].median()))  # Median offset is only based on chosen_loci
    df['mid'] = (df.mid - df.offset).astype(int) + offset_median
    df['chrom_start'] = df.mid - int(spacing * 1e6 / 2)
    df['chrom_end'] = df.mid + int(spacing * 1e6 / 2)

    df = get_index_of_loci(df, spacing=spacing).sort_values(['trace_id', 'idx_genome'])
    cols = ['cell_id', 'chrom', 'trace_id', 'ntraces_per_cell', 'rep_id']
    if 'hmlg' in df.columns:
        cols.append('hmlg')
    df_interp = df.groupby(cols).apply(
        interp_molecule, kind=interp_kind, spacing=spacing, include_groups=False).reset_index(
        level=np.arange(len(cols)).tolist())

    if verbose:
        print(f"\tFINAL number of eligible entries: {len(df_interp):,}", flush=True)

    return df_interp

    