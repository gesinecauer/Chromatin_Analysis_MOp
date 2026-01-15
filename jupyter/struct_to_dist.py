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
    from iced.io import write_lengths
from process_loci import get_index_of_loci
from topsy.analysis.compare_distances import make_matrix_df


def get_n_detected_per_locus_pair(sc_dist_vec, idx=None, exclude_missing_loci=False):
    # In how many cells is each pair of loci detected?
    has_data = np.invert(np.isnan(sc_dist_vec)).astype(int).sum(axis=0)
    nonmissing = squareform(has_data).astype(int)

    # Add missing data if desired
    if exclude_missing_loci:
        nonmissing_per_locus_pair = nonmissing
    else:
        if idx is None:
            raise ValueError("Must supply idx to make complete matrix, including rows/cols of missing loci")
        n = idx.max() + 1
        nonmissing_per_locus_pair = np.zeros((n, n), dtype=int)
        nonmissing_per_locus_pair[idx, idx.reshape(-1, 1)] = nonmissing
    return nonmissing_per_locus_pair


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


def process_sc_distances(sc_dna_coords, idx, outdir, lengths_df, name=None, redo=False):
    os.makedirs(outdir, exist_ok=True)
    if name is not None and name != "":
        name = f"{name}."
    n = idx.max() + 1

    sc_dist_vec_file = os.path.join(outdir, f'{name}distances.vector_per_cell.npy')
    if os.path.exists(sc_dist_vec_file + '.gz') and not os.path.exists(sc_dist_vec_file):
        sc_dist_vec_file = sc_dist_vec_file + '.gz'
    median_dist_matrix_file = os.path.join(outdir, f'{name}distances.median.npy')
    mean_dist_matrix_file = os.path.join(outdir, f'{name}distances.mean.npy')
    nonmissing_per_locus_pair_file = os.path.join(outdir, f'{name}num_nonmissing.npy')
    sc_dist_per_locus_file = os.path.join(outdir, f'{name}distances.per_locus.tsv.gz')
    sc_dist_sameH_file = os.path.join(outdir, f'{name}distances.same-hmlg.tsv.gz')
    sc_dist_diffH_file = os.path.join(outdir, f'{name}distances.diff-hmlg.tsv.gz')

    print(f"Intra-mol sc distances: {sc_dist_sameH_file}", flush=True)
    
    all_files = [sc_dist_vec_file, median_dist_matrix_file, mean_dist_matrix_file,
                nonmissing_per_locus_pair_file, sc_dist_per_locus_file, sc_dist_sameH_file,
                 sc_dist_diffH_file]
    missing_files = [os.path.basename(f) for f in all_files if not os.path.exists(f)]
    if (not redo) and len(missing_files) == 0:
        return {'dis_mean': np.load(mean_dist_matrix_file), 'dis_median': np.load(median_dist_matrix_file),
                'nonmissing': np.load(nonmissing_per_locus_pair_file)}
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

    if redo or not os.path.exists(nonmissing_per_locus_pair_file):
        print('Generate num detected per locus pair...', flush=True)
        nonmissing_per_locus_pair = get_n_detected_per_locus_pair(sc_dist_vec, idx=idx)
        print('                 ...saving', flush=True)
        if not os.path.exists(nonmissing_per_locus_pair_file):
            np.save(nonmissing_per_locus_pair_file, nonmissing_per_locus_pair)
    else:
        nonmissing_per_locus_pair = np.load(nonmissing_per_locus_pair_file).astype(int)

    if redo or not os.path.exists(sc_dist_per_locus_file):
        print('Get single-cell distances per locus...', flush=True)
        save_sc_dis_per_locus(sc_dist_vec, idx=idx, outfile=sc_dist_per_locus_file, lengths_df=lengths_df)

    if redo or not (os.path.exists(sc_dist_sameH_file) and os.path.exists(sc_dist_diffH_file)):
        print('Saving intra-chromosomal single-cell distances...', flush=True)
        save_sc_dis_intrachr(
            sc_dist_vec, idx=idx, outfile_sameH=sc_dist_sameH_file,
            outfile_diffH=sc_dist_diffH_file, lengths_df=lengths_df)

    print('Done!', flush=True)
    return {'dis_mean': mean_dist_matrix, 'dis_median': median_dist_matrix, 'nonmissing': nonmissing_per_locus_pair}



def setup_dna_coords(df, phased=True, phasing_column='hmlg'):

    # Get unified diploid index, phased or unphased
    df['idx'] = df['idx_genome']
    if phased:
        n = len(lengths_df)
        df['idx'] *= (df[phasing_column] - 1) * n

    # If needed, filter for cells that contain two traces
    if phased and phasing_column == 'trace_id':
        pass
    
    # Get 3D coordinates for each cell (shape = ncells, nloci, ndim)
    df.sort_values(['idx', 'cell_id', 'chrom', 'chrom_order'], inplace=True)
    df.set_index(['idx', 'cell_id'], inplace=True)
    sc_dna_coords = np.stack([df[[c]].unstack(level=0).values for c in ['x', 'y', 'z']], axis=2)

    idx = df.idx.drop_duplicates().sort_values().values

    return sc_dna_coords, idx




def process_sc_dna_coords(coords_file, outdir=None, spacing=2.5, name=None, chrom=None, redo=False, verbose=False):
    outdir_matrix2d = os.path.join(os.path.dirname(coords_file), 'matrix2d')

    df = pd.read_csv(
        coords_file, sep='\t', index_col=0,
        converters={0: ast.literal_eval, 'merfish_id': ast.literal_eval})
    lengths_file = os.path.join(os.path.dirname(coords_file), 'counts.bed')


    if chrom is not None:
        raise NotImplementedError # FIXME

    sc_dna_coords, idx = setup_dna_coords(df)

    matrices = process_sc_distances(
        sc_dna_coords, idx=idx, outdir=outdir_matrix2d, lengths_df=lengths_df,
        redo=redo, name=name)

    return matrices, lengths_df, outdir_matrix2d


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--spacing", default=2.5, type=float)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--chrom", type=str, nargs='+')
    parser.add_argument('--redo', default=False, action='store_true')
    parser.add_argument('--verbose', default=True, action='store_true')
    parser.add_argument('--silent', dest='verbose', default=True, action='store_false')
    args = parser.parse_args()

    name = args.name
    if name is None:
        name = re.sub(r'(^|.*/)cluster(?:\.LINK){0,1}/([^/]+)(/.*|$)', r'\2', os.path.dirname(args.data))

    process_sc_dna_coords(
        coords_file=args.data, spacing=args.spacing, outdir=args.outdir, name=name,
        chrom=args.chrom, redo=args.redo, verbose=args.verbose)


if __name__ == "__main__":
    main()