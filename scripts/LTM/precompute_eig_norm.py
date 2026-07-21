'''
Pools normalized Laplacian eigenvalues across every realization seed's .gt
graph for one (network, k, p) combo, for the figure notebooks' bottom-row
histogram panels.

The realization .gt files (generate_ltm.py) only store the *unnormalized*
Laplacian spectrum (norm=False), but the histogram panels use the
normalized Laplacian (gt.laplacian(G, norm=True), range [0,2]) - so this
recomputes it per graph rather than reading a cached vertex property.

Output: nets/LTM/1000/<net>/<k>_p<value>_eig_norm.npy, the concatenated
normalized eigenvalues of every seed's graph at that (net, k, p) - one
combo per array task.

Usage (SLURM array index selects the combo - see submit_eig_norm.sh):
    python precompute_eig_norm.py --index 0
'''

import argparse
from pathlib import Path

import graph_tool as gt
import graph_tool.spectral
import numpy as np

N = 1000
PICK_PROPS = {'ws': [0, 15, 23, 30, 65], 'mhk': [99, 60, 10, 0]}
PROB_ARRAYS = {
    'ws': np.linspace(0, 1, 101),
    'mhk': np.append([0], np.linspace(0.9, 1, 100)),
}
K = 16

COMBOS = [(net, p) for net in ('ws', 'mhk') for p in PROB_ARRAYS[net][PICK_PROPS[net]]]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--index', type=int, required=True, help=f'Combo index, 0-{len(COMBOS) - 1}')
    parser.add_argument('--root', type=str, default='nets')
    return parser.parse_args()


def main():
    args = parse_args()
    net, p = COMBOS[args.index]

    root = Path(args.root) / 'LTM' / str(N) / net
    seed_dirs = sorted(d for d in root.glob(f'{K}_seed*') if d.is_dir())

    all_eigs = []
    for seed_dir in seed_dirs:
        graph_path = seed_dir / f'p{p:.6f}.gt'
        G = gt.load_graph(str(graph_path))
        eig = np.linalg.eigvalsh(gt.spectral.laplacian(G, norm=True).todense())
        all_eigs.append(eig)

    pooled = np.concatenate(all_eigs)
    out_path = root / f'{K}_p{p:.6f}_eig_norm.npy'
    np.save(out_path, pooled)
    print(f'{out_path} saved ({len(seed_dirs)} seeds, {pooled.size} eigenvalues)')


if __name__ == '__main__':
    main()
