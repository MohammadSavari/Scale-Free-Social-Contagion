'''
Generates independent realizations of the LTM .gt networks used by the
figure notebook - just the (network_type, k, p) triples the figures need,
not a full dense p sweep.

The p-values are CLUSTERING-MATCHED. p is not comparable across the models
(a rewiring probability in ws, a triad-formation probability in mhk), so the
comparison is pinned to equal mean local clustering instead: four targets
spaced evenly across mhk's reachable CC range, with the ws p realizing each
target solved from ws's own CC(p) curve, plus an unmatched fifth ws value at
p=0 (the ring lattice, whose CC sits above mhk's ceiling). The values live in
mhk_cc_matched_probs.json - see the "p-values are matched on clustering"
section of README.md.

ke gets ONE graph per (k, seed): ke_network has no p-dependence, so
p-labelled duplicates would be repeated draws of a single ensemble.
Realizations, indexed by --seed, are the meaningful axis. ke is generated
for k in {8, 16} only.

Realizations are written to nets/LTM/1000/<net>/<k>_seed<n>/, mirroring
the nets/LFC/240/<net>/<k>_seed<n>/ realization layout used for LFC.
Idempotent (existing .gt files are skipped), so resuming after a partial
failure is safe.

Usage (one realization per call, for SLURM array parallelism - see
submit_ltm_generate.sh):
    python generate_ltm.py --seed 0
    python generate_ltm.py --seed 0 --nets mhk ke
'''

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Generators and graph-property helpers come from scripts/net_functions.py -
# the single source of truth for every pipeline stage.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from net_functions import (          # noqa: E402
    ws_network,
    mhk_network,
    ke_network,
    nx_to_gt,
    get_local_clutsering,
    get_transitivity,
    get_ave_shortest_path,
    get_laplacian_eigenvalues,
    get_kirchhoff_index,
)

N = 1000

# ke is restricted to k in {8, 16}. Its topology has no p-dependence, so
# p-labelled graphs would be redundant draws of one ensemble rather than a
# sweep; independent realizations are the meaningful axis, indexed by --seed.
COMBOS = [('ws', 16), ('ws', 8), ('mhk', 16), ('mhk', 8), ('ke', 16), ('ke', 8)]

# p is NOT comparable between the models: it is a rewiring probability in ws
# and a triad-formation probability in mhk, so equal p is not a controlled
# comparison. The p-values are therefore matched on CLUSTERING instead - four
# targets spaced evenly across mhk's reachable CC range, with the ws p that
# realizes each target solved from ws's own CC(p) curve. ws additionally
# keeps p=0 (the pure ring lattice) as an unmatched fifth value, since its CC
# sits above mhk's ceiling and has no mhk counterpart.
#
# The matched values are checked in as mhk_cc_matched_probs.json - a fixed
# constant of the study, so reproducing this data needs no calibration run.
CC_PROBS_PATH = Path(__file__).resolve().parent / 'mhk_cc_matched_probs.json'

cascades = np.round(np.linspace(0.1, 0.9, 9), 1)


def load_cc_probs(path=CC_PROBS_PATH):
    with open(path) as fh:
        return json.load(fh)


def probs_for(net, k, cal):
    '''p-values to generate for one (net, k).

    mhk -> the CC-matched set for THIS k
    ws  -> the p realizing the same CC values, plus the unmatched p=0 lattice
    ke  -> a single placeholder 0.0; ke ignores p entirely
    '''
    if net == 'ke':
        return [0.0]
    entries = cal['probs'][str(k)]
    if net == 'mhk':
        return [float(e['p']) for e in entries]
    ps = [float(e['ws_p']) for e in entries]
    ps += [float(e['ws_p']) for e in cal.get('ws_extra', {}).get(str(k), [])]
    return sorted(set(ps))


def generate_one(net, k, p, seed):
    # mhk/ke return networkx graphs, so they cross to graph_tool here;
    # expected_n catches a generator returning the wrong node count.
    if net == 'ws':
        G = ws_network(N, k, p, seed=seed)          # already graph_tool
    elif net == 'mhk':
        G = nx_to_gt(mhk_network(N, k, p, seed=seed), expected_n=N)
    elif net == 'ke':
        G = nx_to_gt(ke_network(N, k, seed=seed), expected_n=N)
    else:
        raise ValueError(net)

    G.graph_properties['ID'] = G.new_graph_property('int64_t', val=int(time.time() * 1000))
    G.graph_properties['ntype'] = G.new_graph_property('string', val=net)
    G.graph_properties['probability'] = G.new_graph_property('double', p)
    G.graph_properties['cascades'] = G.new_gp(value_type='vector<double>', val=cascades)
    G = get_local_clutsering(G)
    G = get_transitivity(G)
    G = get_ave_shortest_path(G)
    G = get_laplacian_eigenvalues(G)
    G = get_kirchhoff_index(G)
    return G


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, required=True,
                         help='Realization index (0-99) - also seeds ws/mhk/ke generation.')
    parser.add_argument('--root', type=str, default='nets',
                         help="Root directory (default: 'nets', matching the LFC realization layout).")
    parser.add_argument('--nets', type=str, nargs='+', default=None,
                         choices=['ws', 'mhk', 'ke'],
                         help='Restrict to these networks (default: all in COMBOS).')
    parser.add_argument('--cc-probs', type=str, default=str(CC_PROBS_PATH),
                         help='Path to mhk_cc_matched_probs.json (the CC-matched p-values).')
    return parser.parse_args()


def main():
    args = parse_args()

    combos = COMBOS if args.nets is None else [(n, k) for n, k in COMBOS if n in args.nets]
    cal = load_cc_probs(args.cc_probs) if any(n != 'ke' for n, _ in combos) else None

    for net, k in combos:
        out_dir = Path(args.root) / 'LTM' / str(N) / net / f'{k}_seed{args.seed}'
        out_dir.mkdir(parents=True, exist_ok=True)

        for p_index, p in enumerate(probs_for(net, k, cal)):
            out_path = out_dir / f'p{p:.6f}.gt'
            if out_path.exists():
                print(f'{out_path} exists, skipping', flush=True)
                continue
            # Offset by the p INDEX, not by p itself: indices are dense in
            # [0, len(probs)), so each realization's block of seeds stays
            # disjoint from its neighbours' and no two graphs in the whole
            # sweep can come out bit-identical.
            graph_seed = args.seed * 1000 + p_index
            G = generate_one(net, k, p, graph_seed)
            G.save(str(out_path))
            print(f'{out_path} saved  (E={G.num_edges()} '
                  f'kbar={2 * G.num_edges() / G.num_vertices():.4f})', flush=True)
    print('Done.')


if __name__ == '__main__':
    main()
