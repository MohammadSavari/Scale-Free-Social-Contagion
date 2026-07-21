'''
Scans nets/LTM/1000/<net>/<k>_seed<seed>/ directories (produced by
generate_ltm.py) and writes one props CSV per (net, k, seed) directory,
mirroring extract_lfc_csv.py but for LTM realizations.

Each .gt file already carries local_clustering, transitivity,
shortest_path, and eig_laplacian (unnormalized) as vertex/graph
properties, computed at generation time - so this is pure extraction,
no recomputation.

Output: nets/LTM/1000/<net>/<k>_seed<seed>_props.csv, one row per p-value,
columns CC, T, SP, Rg, indexed by (ID, p).

Usage:
    python extract_ltm_csv.py --root nets
'''

import argparse
from pathlib import Path

import numpy as np
import graph_tool as gt
import pandas as pd


def get_graph_props(G):
    eig = np.sort(G.vp.eig_laplacian.a)
    return {
        'CC': sum(G.vp.local_clustering.get_array()) / G.num_vertices(),
        'T': G.gp.transitivity,
        'SP': G.gp.get('shortest_path'),
        'Rg': float(G.num_vertices() * np.sum(1 / eig[1:])),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=str, default='nets', help='Root directory to recursively search for LTM .gt files')
    return parser.parse_args()


def main():
    args = parse_args()
    ltm_root = Path(args.root) / 'LTM'

    by_dir = {}
    for path in ltm_root.rglob('*.gt'):
        by_dir.setdefault(path.parent, []).append(path)

    for directory, paths in sorted(by_dir.items()):
        graphs = [gt.load_graph(str(p)) for p in paths]
        prop_rows = {(G.gp.ID, G.gp.probability): get_graph_props(G) for G in graphs}
        network_props = pd.DataFrame.from_dict(prop_rows, orient='index', columns=['CC', 'T', 'SP', 'Rg'])
        network_props.index.names = ['ID', 'p']

        spec_file = f'{directory}_props.csv'
        network_props.to_csv(spec_file, sep='\t', mode='w', header=True)
        print(spec_file)

    print('Done all!')


if __name__ == '__main__':
    main()
