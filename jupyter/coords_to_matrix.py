import numpy as np
import pandas as pd
import os
import re
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='', category=UserWarning)
    warnings.filterwarnings('ignore', message='', category=FutureWarning)
    from iced.io import write_counts, write_lengths
from parse_dna_merfish import filter_data_per_hmlg, restrict_to_equal_nmol_per_hmlg
from process_loci import get_index_of_loci
from topsy.analysis.compare_distances import make_matrix_df


def logistic(d, k, d0, nu=1, q=1):
    tmp = -k * (d - d0)
    if nu == 1:
        counts = 1 / (1 + q * np.exp(tmp))
    else:
        counts = 1 / np.power(1 + q * np.exp(tmp), 1 / nu)
    return counts


def get_transfer_func_params(contact_th=None, alpha=None, k=None, d0=None, sc_dist_vec=None):
    if (contact_th is None) + (alpha is None) + (k is None or d0 is None) != 2:
        raise ValueError("Must input either: \n- contact_th\n- alpha\n- k & d0."
                         f"\nInput: {contact_th=:g}, {alpha=:g}, {k=:g}, {d0=:g}")

    if sc_dist_vec is not None:
        if (contact_th is not None) and (
            contact_th > np.nanmax(sc_dist_vec) or contact_th < np.nanmin(sc_dist_vec)):
        raise ValueError(f"{contact_th=:g}μm is not appropriate for sc distances, which range from"
                     f" {np.nanmin(sc_dist_vec):g}μm to {np.nanmax(sc_dist_vec):g}μm")
    if alpha is not None:
        if alpha > -1 or alpha < -6:
            raise ValueError(f"Alpha should be in the range [-6, -1], inputted {alpha=:g}")
    if k is not None and d0 is not None:
        if d0 < 0 or d0 > 1:
            raise ValueError(f"d0 should be in the range [0, 1], inputted {d0=:g}")
        if k >= 0:
            raise ValueError(f"k should be < 0, inputted {k=:g}")
    
    return contact_th, alpha, k, d0


def generate_counts_from_sc_dist(sc_dist_vec, transfer_func_kwargs, idx=None, exclude_missing_loci=False):
    contact_th, alpha, k, d0 = get_transfer_func_params(**transfer_func_kwargs, sc_dist_vec=None)

    # In how many cells is each pair of loci detected?
    has_data = np.invert(np.isnan(sc_dist_vec)).astype(int).sum(axis=0)
    nonmissing = squareform(has_data)

    # Generate pseudo-counts
    if contact_th is not None:  # Use contact threshold approach
        pass_thresh = (sc_dist_vec < contact_th).astype(int).sum(axis=0)
        res = squareform(pass_thresh / has_data)
    elif alpha is not None:  # Generate sc counts via 'c = d^alpha' transfer function
        counts_vec = np.nanmean(np.power(sc_dist_vec, alpha), axis=0)
        print(f"{counts_vec.max()=:g}", flush=True)
        res = squareform(counts_vec)
    else:  # Generate counts via logistic transfer function
        counts_vec = np.nanmean(logistic(sc_dist_vec, k=k, d0=d0), axis=0)
        res = squareform(counts_vec)

    # Add missing data if desired
    if exclude_missing_loci:
        mean_counts_matrix = res
        nonmissing_per_locus_pair = nonmissing
    else:
        if idx is None:
            raise ValueError("Must supply idx to make complete counts matrix, including rows/cols of missing loci")
        n = idx.max() + 1
        mean_counts_matrix = np.zeros((n, n))
        mean_counts_matrix[idx, idx.reshape(-1, 1)] = res
        nonmissing_per_locus_pair = np.zeros((n, n))
        nonmissing_per_locus_pair[idx, idx.reshape(-1, 1)] = nonmissing
    return mean_counts_matrix, nonmissing_per_locus_pair


def save_sc_dis_intrachr(sc_dist_vec, idx, outfile_sameH, outfile_diffH, lengths_df):
    row, col = np.triu_indices(idx.size, 1)
    row = idx[row]
    col = idx[col]
    matrix_idx = list(map(tuple, np.stack([row, col], axis=1).tolist()))
    
    matrix_df = make_matrix_df(lengths_df)
    matrix_df = matrix_df.loc[matrix_idx]  # Sort/filter to match sc_dist_vec

    if not os.path.exists(outfile_sameH):
        print('                 ...intra-molecular', flush=True)
        mask = matrix_df['mask.sameC-sameH'].values
        df = pd.DataFrame(data=sc_dist_vec.T[mask], index=matrix_df.index[mask])
        df.to_csv(outfile_sameH, index=True, header=False, sep='\t')

    if not os.path.exists(outfile_diffH):
        print('                 ...intra-chromosomal, inter-homolog', flush=True)
        mask = matrix_df['mask.sameC-diffH'].values
        df = pd.DataFrame(data=sc_dist_vec.T[mask], index=matrix_df.index[mask])
        df.to_csv(outfile_diffH, index=True, header=False, sep='\t')
    

def save_sc_dis_per_locus(sc_dist_vec, idx, outfile, lengths_df):
    _, where_idx_nan = np.where(~np.isnan(sc_dist_vec))

    row, col = np.triu_indices(idx.size, 1)
    row = idx[row][where_idx_nan]
    col = idx[col][where_idx_nan]
    # matrix_idx = list(map(tuple, np.stack([row, col], axis=1)))
    # matrix_idx = [(int(row[i]), int(col[i])) for i in range(row.size)]
    matrix_idx = list(map(tuple, np.stack([row, col], axis=1).tolist()))

    df = pd.DataFrame(data={'dis': sc_dist_vec[~np.isnan(sc_dist_vec)]}, index=matrix_idx)

    df = df.groupby(level=0).apply(
        lambda x: x.dis.values.tolist(), include_groups=False).reset_index(level=0).rename(
        {0: 'dis', 'index': 'idx'}, axis=1).set_index('idx')

    matrix_df = make_matrix_df(lengths_df)
    sameM = (~matrix_df[['mask.diffM']]).rename({'mask.diffM': 'same_molecule'}, axis=1).astype(int)

    idx_mismatch = ~df.index.isin(matrix_df.index)
    if idx_mismatch.any():
        print(f"{len(df)=}, {len(matrix_df)=}, {idx_mismatch.sum()=}", flush=True)
        print(matrix_df.head().index, flush=True)
        print(df.head().index, flush=True)
        raise ValueError(f"Index mismatch!! {len(df)=}, {len(matrix_df)=}\n{idx_mismatch.sum()}"
                         f" indices in df but not in matrix_df:\n{idx_mismatch[idx_mismatch].index}")
    
    df = df.join(sameM).reset_index().rename({'index': 'idx'}, axis=1).sort_values(
        ['same_molecule', 'idx'], ascending=[False, True])

    df['dis_mean'] = df.dis.apply(np.mean)
    df['dis_med'] = df.dis.apply(np.median)

    df = df[['idx', 'same_molecule', 'dis_mean', 'dis_med', 'dis']]
    print('                 ...saving single-cell distances per locus', flush=True)
    df.to_csv(outfile, index=False, header=False, sep='\t')


def process_sc_distances(sc_dna_coords, idx, outdir, lengths_df, transfer_func_kwargs, name=None, redo=False):
    contact_th, alpha, k, d0 = get_transfer_func_params(**transfer_func_kwargs)  # Check params

    os.makedirs(outdir, exist_ok=True)
    if name is not None and name != "":
        name = f"{name}."
    n = idx.max() + 1

    sc_dist_vec_file = os.path.join(outdir, f'{name}distances.vector_per_cell.npy')
    if os.path.exists(sc_dist_vec_file + '.gz') and not os.path.exists(sc_dist_vec_file):
        sc_dist_vec_file = sc_dist_vec_file + '.gz'
    median_dist_matrix_file = os.path.join(outdir, f'{name}distances.median.npy')
    mean_dist_matrix_file = os.path.join(outdir, f'{name}distances.mean.npy')
    # sc_counts_vec_file = os.path.join(outdir, f'{name}counts.vector_per_cell.cutoff{contact_th:g}.npy')
    if contact_th is not None:
        mean_counts_matrix_file = os.path.join(outdir, f'{name}counts.mean.cutoff{contact_th:g}.npy')
    else:
        mean_counts_matrix_file = os.path.join(outdir, f'{name}counts.mean.alpha{alpha:g}.npy')
    nonmissing_per_locus_pair_file = os.path.join(outdir, f'{name}num_nonmissing.npy')
    sc_dist_per_locus_file = os.path.join(outdir, f'{name}distances.per_locus.tsv.gz')
    sc_dist_intramol_file = os.path.join(outdir, f'{name}distances.intramol.tsv.gz')
    sc_dist_sameC_diffH_file = os.path.join(outdir, f'{name}distances.sameC-diffH.tsv.gz')

    print(f"Counts: {mean_counts_matrix_file}", flush=True)
    
    all_files = [sc_dist_vec_file, median_dist_matrix_file, mean_dist_matrix_file, mean_counts_matrix_file,
                nonmissing_per_locus_pair_file, sc_dist_per_locus_file, sc_dist_intramol_file,
                 sc_dist_sameC_diffH_file]
    missing_files = [os.path.basename(f) for f in all_files if not os.path.exists(f)]
    if (not redo) and len(missing_files) == 0:
        return {'counts': np.load(mean_counts_matrix_file), 'dis_mean': np.load(mean_dist_matrix_file),
                'dis_median': np.load(median_dist_matrix_file), 'nonmissing': np.load(nonmissing_per_locus_pair_file)}
    print(f"Creating files in {outdir}:\n\t- " + '\n\t- '.join(missing_files), flush=True)
    
    if isinstance(sc_dna_coords, str):
        print('Loading sc DNA coords...', flush=True)
        sc_dna_coords = np.load(sc_dna_coords)

    print('Converting sc DNA coords to distance vectors...', flush=True)
    sc_dist_vec = np.stack([pdist(x) for x in tqdm(sc_dna_coords)])  # shape=(ncells, nloci)
    if redo or not os.path.exists(sc_dist_vec_file):
        print('                 ...saving...', flush=True)
        np.save(sc_dist_vec_file, sc_dist_vec)

    if redo or not os.path.exists(median_dist_matrix_file):
        print('Generate median distances across cells...', flush=True)
        median_dist_matrix = np.full((n, n), np.nan)
        median_dist_matrix[idx, idx.reshape(-1, 1)] = squareform(np.nanmedian(sc_dist_vec, axis=0))
        np.save(median_dist_matrix_file, median_dist_matrix)
    else:
        median_dist_matrix = np.load(median_dist_matrix_file)

    if redo or not os.path.exists(mean_dist_matrix_file):
        print('Generate mean distances across cells...', flush=True)
        mean_dist_matrix = np.full((n, n), np.nan)
        mean_dist_matrix[idx, idx.reshape(-1, 1)] = squareform(np.nanmean(sc_dist_vec, axis=0))
        np.save(mean_dist_matrix_file, mean_dist_matrix)
    else:
        mean_dist_matrix = np.load(mean_dist_matrix_file)

    # Check appropriateness of contact threshold
    if contact_th is not None and (contact_th > np.nanmax(sc_dist_vec) or contact_th < np.nanmin(sc_dist_vec)):
        raise ValueError(f"{contact_th=}μm is not appropriate for sc distances, which range from"
                         f" {np.nanmin(sc_dist_vec):g}μm to {np.nanmax(sc_dist_vec):g}μm")

    if redo or not (os.path.exists(mean_counts_matrix_file) and os.path.exists(nonmissing_per_locus_pair_file)):
        print('Generate counts...', flush=True)
        mean_counts_matrix, nonmissing_per_locus_pair = generate_counts_from_sc_dist(
            sc_dist_vec, transfer_func_kwargs=transfer_func_kwargs, idx=idx)
        print('                 ...saving mean contacts across cells', flush=True)
        if not os.path.exists(mean_counts_matrix_file):
            np.save(mean_counts_matrix_file, mean_counts_matrix)
        if not os.path.exists(nonmissing_per_locus_pair_file):
            np.save(nonmissing_per_locus_pair_file, nonmissing_per_locus_pair)
    else:
        mean_counts_matrix = np.load(mean_counts_matrix_file)
        nonmissing_per_locus_pair = np.load(nonmissing_per_locus_pair_file).astype(int)

    if redo or not os.path.exists(sc_dist_per_locus_file):
        print('Get single-cell distances per locus...', flush=True)
        save_sc_dis_per_locus(sc_dist_vec, idx=idx, outfile=sc_dist_per_locus_file, lengths_df=lengths_df)

    if redo or not (os.path.exists(sc_dist_intramol_file) and os.path.exists(sc_dist_sameC_diffH_file)):
        print('Saving intra-chromosomal single-cell distances...', flush=True)
        save_sc_dis_intrachr(
            sc_dist_vec, idx=idx, outfile_sameH=sc_dist_intramol_file,
            outfile_diffH=sc_dist_sameC_diffH_file, lengths_df=lengths_df)

    print('Done!', flush=True)
    return {'counts': mean_counts_matrix, 'dis_mean': mean_dist_matrix, 'dis_median': median_dist_matrix, 'nonmissing': nonmissing_per_locus_pair}


def add_missing_loci(df, spacing=2.5):
    assert df.idx_chrom.min() == 0  # (Shouldn't be including loci at beginning of chrom if they're always NaN)
    idx_chrom = np.arange(df.idx_chrom.min(), df.idx_chrom.max() + 1, dtype=int)
    if len(idx_chrom) == len(df):
        return df
    df_missing = pd.DataFrame()
    df_missing['idx_chrom'] = idx_chrom[~np.isin(idx_chrom, df.idx_chrom.values)]
    df_missing['idx_genome'] = df.idx_genome.min() + df_missing.idx_chrom
    df_missing['mid'] = df.mid.min() + (df_missing.idx_chrom * spacing * 1e6).astype(int)
    df_missing['has_data'] = 0
    return pd.concat([df, df_missing]).reset_index(drop=True)


def process_sc_dna_coords(input_file, outdir=None, min_nonmissing_per_phased_locus=0.05, nmol_per_hmlg_ratio=1,
                          spacing=2.5, contact_th=0.75, alpha=None, redo=False, name=None, verbose=False):

    transfer_func_kwargs = dict(contact_th=contact_th, alpha=alpha, k=k, d0=d0)
    get_transfer_func_params(**transfer_func_kwargs)  # Check params

    # Assign defaults
    if min_nonmissing_per_phased_locus is None:
        min_nonmissing_per_phased_locus = 0.05
    if nmol_per_hmlg_ratio is None:
        nmol_per_hmlg_ratio = 1

    if outdir is None:
        outdir = os.path.dirname(input_file)
    output_note = ''
    if nmol_per_hmlg_ratio is None:
        output_note += '.nmol_per_hmlg_unrestricted'
    elif nmol_per_hmlg_ratio != 1:
        output_note += f'.nmol_per_hmlg_within_{nmol_per_hmlg_ratio * 100:.3g}p'
    if min_nonmissing_per_phased_locus != 0.05:
        output_note += f'.min_nonmissing_per_phased_locus_{min_nonmissing_per_phased_locus * 100:.3g}p'
    outdir_matrix2d = os.path.join(outdir, f'matrix2d{output_note}')
    outfile_df = os.path.join(outdir, re.sub(r'\.csv(\.gz)*$', '', os.path.basename(input_file)) + f'.processed{output_note}.csv')

    # Load data, restrict each chromosome to have an equal (or roughly equal) number of molecules per homolog
    df = restrict_to_equal_nmol_per_hmlg(input_file, cutoff_ratio=nmol_per_hmlg_ratio, verbose=verbose)

    # Remove loci where <[cutoff]% of cells are non-missing from one or more of the traces
    df, cov_per_locus = filter_data_per_hmlg(df, min_nonmissing_per_phased_locus=min_nonmissing_per_phased_locus, verbose=verbose)
    
    # Get index of each locus in the final counts/distance matrices
    #if 'idx_chrom' not in df.columns or 'idx_genome' not in df.columns:
    df = get_index_of_loci(df, spacing=spacing)  # Redo indexing, even if it's been done before
    nbins_per_hmlg = df.idx_genome.max() + 1
    df['idx'] = nbins_per_hmlg * (df.hmlg - 1) + df.idx_genome
    # idx = np.arange(df.idx.min(), df.idx.max() + 1, dtype=int)
    
    # Create chromosome lengths for bed file
    lengths_df = df[['idx_genome', 'chrom', 'idx_chrom', 'chrom_start', 'chrom_end', 'chrom_order']].drop_duplicates().sort_values('idx_genome')
    lengths_df = lengths_df.merge(
        cov_per_locus.reset_index().drop('pass_cutoff', axis=1, errors='ignore'),
        on=['chrom', 'chrom_order'], how='left').drop('chrom_order', axis=1)
    lengths_df['mid'] = lengths_df[['chrom_start', 'chrom_end']].mean(axis=1)
    offset = lengths_df['mid'] % (spacing * 1e6)
    lengths_df['mid'] = (lengths_df['mid'] - offset).astype(int) + int(round(offset.median()))
    lengths_df['chrom'] = 'chr' + lengths_df.chrom.astype(str)
    lengths_df['has_data'] = 1
    lengths_df = lengths_df.groupby('chrom').apply(
        add_missing_loci, include_groups=False, spacing=spacing).reset_index(level=0).sort_values(
        'idx_genome').reset_index(drop=True)
    lengths_df['start'] = lengths_df['mid'] - int(spacing * 1e6 / 2)
    lengths_df['end'] = lengths_df['mid'] + int(spacing * 1e6 / 2)
    lengths_df = lengths_df[['chrom', 'start', 'end', 'idx_genome', 'idx_chrom', 'mid', 'has_data', 'cell_cov_min', 'cell_cov_h1', 'cell_cov_h2']]

    # Save bed file of chromosome lengths
    lengths_df_cp = lengths_df.copy()
    lengths_df_cp.columns = [f"#{c}" for c in lengths_df_cp.columns]
    lengths_df_cp.to_csv(os.path.join(outdir, "counts.bed"), index=False, header=True, sep="\t")
    
    # Get 3D coordinates for each cell (shape = ncells, nloci, ndim)
    df.sort_values(['idx', 'cell_id'], inplace=True)
    df.to_csv(outfile_df, index=False)
    idx = df.idx.drop_duplicates().sort_values().values

    df.set_index(['idx', 'cell_id'], inplace=True)
    df = df[['hmlg', 'chrom', 'idx_chrom', 'idx_genome', 'x', 'y', 'z']]
    sc_dna_coords = np.stack([df[[c]].unstack(level=0).values for c in ['x', 'y', 'z']], axis=2)

    matrices = process_sc_distances(
        sc_dna_coords, idx=idx, outdir=outdir_matrix2d, lengths_df=lengths_df,
        transfer_func_kwargs=transfer_func_kwargs, redo=redo, name=name)

    return matrices, lengths_df, outdir_matrix2d


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--spacing", default=2.5, type=float)
    parser.add_argument("--min_nonmissing_per_phased_locus", default=0.05, type=float)
    parser.add_argument("--nmol_per_hmlg_ratio", default=1, type=float)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--contact_th", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--d0", type=float)
    # parser.add_argument("--chrom", type=str, nargs='+')
    parser.add_argument('--verbose', default=True, action='store_true')
    parser.add_argument('--silent', dest='verbose', default=True, action='store_false')
    args = parser.parse_args()

    name = args.name
    if name is None:
        name = re.sub(r'(^|.*/)cluster(?:\.LINK){0,1}/([^/]+)(/.*|$)', r'\2', os.path.dirname(args.data))

    nmol_per_hmlg_ratio = args.nmol_per_hmlg_ratio
    if args.nmol_per_hmlg_ratio >= 1000:
        nmol_per_hmlg_ratio = None

    process_sc_dna_coords(
        input_file=args.data, min_nonmissing_per_phased_locus=args.min_nonmissing_per_phased_locus,
        nmol_per_hmlg_ratio=nmol_per_hmlg_ratio, spacing=args.spacing, outdir=args.outdir, name=name,
        contact_th=args.contact_th, alpha=args.alpha, k=args.k, d0=args.d0, verbose=args.verbose)


if __name__ == "__main__":
    main()