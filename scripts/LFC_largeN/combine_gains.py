'''
Combines every per-graph gains CSV compute_gains.py wrote into one
'<degree>_gains.csv' per degree, for a given node count. Converted from an
interactive notebook (gain_combiner.ipynb) into a parameterized script so it
fits the reproducible pipeline instead of being hand-edited per node count.

Usage:
    python combine_gains.py --root nets/LFC --nodes 1000 --centrality 50
    python combine_gains.py --root nets/LFC --nodes 5000 --centrality 250 --degrees 2 4 6 8
'''

import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=str, default='nets/LFC', help='Root directory holding <nodes>/mhk/<degree>_gain/')
    parser.add_argument('--nodes', type=int, required=True, help='Node count (matches the <root>/<nodes>/mhk/ directory)')
    parser.add_argument('--centrality', type=int, required=True, help='Centrality value used when compute_gains.py ran (matches the _c<centrality>_gains.csv suffix)')
    parser.add_argument('--degrees', type=int, nargs='+', default=list(range(2, 33, 2)), help='Degree values to combine (default: 2..32 step 2)')
    return parser.parse_args()


def main():
    args = parse_args()
    base = f'{args.root}/{args.nodes}/mhk'
    suffix = f'_c{args.centrality}_gains.csv'

    for k in args.degrees:
        src_folder = f'{base}/{k}_gain'
        out_file = f'{base}/{k}_gains.csv'

        if not os.path.isdir(src_folder):
            print(f'  no such folder: {src_folder}')
            continue

        csv_files = sorted(f for f in os.listdir(src_folder) if f.endswith(suffix))

        if not csv_files:
            print(f'  no files found in {src_folder}')
            continue

        print(f'Processing {k}_gain/ ({len(csv_files)} files) -> {k}_gains.csv')

        dfs = [pd.read_csv(os.path.join(src_folder, f), sep='\t') for f in csv_files]
        combined = pd.concat(dfs, ignore_index=True)

        # Rename 'prob' -> 'p'
        if 'prob' in combined.columns:
            combined.rename(columns={'prob': 'p'}, inplace=True)

        combined.to_csv(out_file, sep='\t', index=False)
        print(f'  saved {len(combined)} rows, columns: {list(combined.columns)}')

    print('Done!')


if __name__ == '__main__':
    main()
