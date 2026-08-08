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
import json
from pathlib import Path

import graph_tool as gt
import graph_tool.spectral
import numpy as np

N = 1000
KS = [16, 8]

# The p-values are CC-matched per k: mhk's are calibrated on clustering and
# ws's follow from the same matching. They are read from the same JSON
# generate_ltm.py uses, which keeps this stage in sync with what was
# actually generated.
#
# ws additionally carries its `ws_extra` p (the p=0 ring lattice), which has
# no mhk counterpart because its clustering sits above mhk's ceiling. It must
# be included here: the figure's eigenvalue panel loads one .npy for EVERY p
# present in pooled_props(), and p=0 is one of ws's. Omitting it makes
# Figure 1 die on a missing file rather than silently drop a curve.
CC_PROBS_PATH = Path(__file__).resolve().parent / 'mhk_cc_matched_probs.json'
with open(CC_PROBS_PATH) as _fh:
    _CAL = json.load(_fh)


def probs_for(net, k):
    entries = _CAL['probs'][str(k)]
    if net == 'mhk':
        return [float(e['p']) for e in entries]
    ps = [float(e['ws_p']) for e in entries]
    ps += [float(e['ws_p']) for e in _CAL.get('ws_extra', {}).get(str(k), [])]
    return sorted(set(ps))


COMBOS = [(net, k, p) for k in KS for net in ('ws', 'mhk')
          for p in probs_for(net, k)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--index', type=int, help=f'Combo index, 0-{len(COMBOS) - 1}')
    parser.add_argument('--root', type=str, default='nets')
    parser.add_argument('--n-combos', action='store_true',
                        help='print the number of combos and exit (for sizing the SLURM array)')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_combos:
        print(len(COMBOS))
        return
    if args.index is None:
        raise SystemExit('pass --index I or --n-combos')
    net, k, p = COMBOS[args.index]

    root = Path(args.root) / 'LTM' / str(N) / net
    seed_dirs = sorted(d for d in root.glob(f'{k}_seed*') if d.is_dir())

    all_eigs = []
    for seed_dir in seed_dirs:
        graph_path = seed_dir / f'p{p:.6f}.gt'
        G = gt.load_graph(str(graph_path))
        eig = np.linalg.eigvalsh(gt.spectral.laplacian(G, norm=True).todense())
        all_eigs.append(eig)

    pooled = np.concatenate(all_eigs)
    out_path = root / f'{k}_p{p:.6f}_eig_norm.npy'
    np.save(out_path, pooled)
    print(f'{out_path} saved ({len(seed_dirs)} seeds, {pooled.size} eigenvalues)')


if __name__ == '__main__':
    main()
