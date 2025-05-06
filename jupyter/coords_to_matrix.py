import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform
from iced.io import write_counts, write_lengths
from tqdm import tqdm
from parse_dna_merfish import filter_data_per_hmlg
from process_loci import get_index_of_loci


def generate_counts_from_sc_dist(sc_dist_vec, contact_th=500, idx=None, exclude_missing_loci=False):
    if contact_th > np.nanmax(sc_dist_vec) or contact_th < np.nanmin(sc_dist_vec):
        raise ValueError(f"{contact_th=}μm is not appropriate for sc distances, which range from"
                         f" {np.nanmin(sc_dist_vec):g}μm to {np.nanmax(sc_dist_vec):g}μm")
    # sc_counts_vec = (sc_dist_vec < contact_th).astype(float)
    # sc_counts_vec[np.isnan(sc_dist_vec)] = np.nan
    # res = squareform(np.nanmean(sc_counts_vec, axis=0))
    pass_thresh = (sc_dist_vec < contact_th).astype(int).sum(axis=0)
    has_data = np.invert(np.isnan(sc_dist_vec)).astype(int).sum(axis=0)
    res = squareform(pass_thresh / has_data)
    nonmissing = squareform(has_data)
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


def process_sc_distances(sc_dna_coords, idx, outdir, contact_th=500, name=None, redo=False):
    outdir_matrix2d = os.path.join(outdir, 'matrix2d')
    os.makedirs(outdir_matrix2d, exist_ok=True)
    if name is not None and name != "":
        name = f"{name}."
    n = idx.max() + 1

    sc_dist_vec_file = os.path.join(outdir_matrix2d, f'{name}distances.vector_per_cell.npy')
    median_dist_matrix_file = os.path.join(outdir_matrix2d, f'{name}distances.median.npy')
    mean_dist_matrix_file = os.path.join(outdir_matrix2d, f'{name}distances.mean.npy')
    # sc_counts_vec_file = os.path.join(outdir_matrix2d, f'{name}counts.vector_per_cell.cutoff{contact_th:g}.npy')
    mean_counts_matrix_file = os.path.join(outdir_matrix2d, f'{name}counts.mean.cutoff{contact_th:g}.npy')
    nonmissing_per_locus_pair_file = os.path.join(outdir_matrix2d, f'{name}num_nonmissing.npy')

    print(f"Counts: {mean_counts_matrix_file}", flush=True)
    
    all_files = [sc_dist_vec_file, median_dist_matrix_file, mean_dist_matrix_file, mean_counts_matrix_file,
                nonmissing_per_locus_pair_file]
    missing_files = [os.path.basename(f) for f in all_files if not os.path.exists(f)]
    if (not redo) and len(missing_files) > 0:
        return {'counts': np.load(mean_counts_matrix_file), 'dis_mean': np.load(mean_dist_matrix_file),
                'dis_median': np.load(median_dist_matrix_file), 'nonmissing': np.load(nonmissing_per_locus_pair_file)}
    print(f"Creating files in {outdir_matrix2d}:\n\t- " + '\n\t- '.join(missing_files), flush=True)
    
    if isinstance(sc_dna_coords, str):
        print('Loading sc DNA coords...', flush=True)
        sc_dna_coords = np.load(sc_dna_coords)

    print('Converting sc DNA coords to distance vectors...', flush=True)
    sc_dist_vec = np.stack([pdist(x) for x in tqdm(sc_dna_coords)])
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
    print(contact_th, np.nanmax(sc_dist_vec), np.nanmean(sc_dist_vec), np.nanmin(sc_dist_vec))
    if contact_th > np.nanmax(sc_dist_vec) or contact_th < np.nanmin(sc_dist_vec):
        raise ValueError(f"{contact_th=}μm is not appropriate for sc distances, which range from"
                         f" {np.nanmin(sc_dist_vec):g}μm to {np.nanmax(sc_dist_vec):g}μm")

    if redo or not (os.path.exists(mean_counts_matrix_file) and os.path.exists(nonmissing_per_locus_pair_file)):
        print('Generate counts...', flush=True)
        mean_counts_matrix, nonmissing_per_locus_pair = generate_counts_from_sc_dist(
            sc_dist_vec, contact_th=contact_th, idx=idx)
        print('                 ...saving mean contacts across cells', flush=True)
        if not os.path.exists(mean_counts_matrix_file):
            np.save(mean_counts_matrix_file, mean_counts_matrix)
        if not os.path.exists(nonmissing_per_locus_pair_file):
            np.save(nonmissing_per_locus_pair_file, nonmissing_per_locus_pair)
    else:
        mean_counts_matrix = np.load(mean_counts_matrix_file)

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


def process_sc_dna_coords(input_file, outdir=None, min_nonmissing_per_phased_locus=0.05, spacing=2.5,
                          contact_th=500, redo=False, name=None, verbose=False):

    if outdir is None:
        outdir = os.path.dirname(input_file)
    df = pd.read_csv(input_file)

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
    
    # Get 3D coordinates for each cell (shape = ncells, nloci, ndim)
    df.sort_values(['idx', 'cell_id'], inplace=True)
    idx = df.idx.drop_duplicates().sort_values().values
    df.set_index(['idx', 'cell_id'], inplace=True)
    df = df[['hmlg', 'chrom', 'idx_chrom', 'idx_genome', 'x', 'y', 'z']]
    sc_dna_coords = np.stack([df[[c]].unstack(level=0).values for c in ['x', 'y', 'z']], axis=2)

    matrices = process_sc_distances(sc_dna_coords, idx=idx, outdir=outdir, contact_th=contact_th, redo=redo, name=name)

    return matrices, lengths_df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--spacing", default=2.5, type=float)
    parser.add_argument("--min_nonmissing_per_phased_locus", default=0.05, type=float)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--contact_th", default=500, type=float)
    # parser.add_argument("--chrom", type=str, nargs='+')
    parser.add_argument('--verbose', default=True, action='store_true')
    parser.add_argument('--silent', dest='verbose', default=True, action='store_false')
    args = parser.parse_args()

    name = args.name
    if name is None:
        name = os.path.basename(os.path.dirname(args.data))
        if name.startswith('ntraces_chrom-cell_'):
            name = os.path.basename(os.path.dirname(os.path.dirname(args.data)))

    process_sc_dna_coords(
        input_file=args.data, min_nonmissing_per_phased_locus=args.min_nonmissing_per_phased_locus,
        spacing=args.spacing, outdir=args.outdir, name=name, contact_th=args.contact_th, verbose=args.verbose)


if __name__ == "__main__":
    main()