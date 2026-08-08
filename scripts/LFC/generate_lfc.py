'''
Generates the gt files (no CSVs - use extract_lfc_csv.py to derive CSVs
from the saved .gt files afterwards).

Random seed, network model type, rewiring/edge probability (p), mean degree
(k), and node count are all command line arguments, so one invocation is one
graph and the sweep lives in the submitting script. One output graph
directory per (nodes, k) is produced, tagged with model_type/seed so that
parallel runs (e.g. sbatch array jobs) do not clobber each other.

Also computes the frequency-domain gains and stores them as two vertex
properties (gains_top5, gains_bot5) on the same .gt file, so every property
(clustering, transitivity, shortest path, Laplacian eigenvalues, and gains)
is saved together at generation time.

Gains come from net_functions.gain_all_nodes, which recovers H^2 for every
node from ONE eigendecomposition rather than a dense complex solve per
(node, frequency). The top/bottom 5% sets are then slices of that single
result. This is what makes large N affordable: ~0.1 s per graph at N=240,
~5 s at N=1000, ~9.5 min at N=5000.

`ke` (Klemm-Eguiluz high-clustering scale-free) networks have no
rewiring-probability parameter - ke_network(n, k, seed) only depends on
node count and mean degree, so its topology is identical across the whole
`--p` sweep for a fixed (nodes, k, seed). `--p` is still accepted and stored
as a graph property for schema consistency with mhk/ws (so the same
extraction/pooling pipeline works across all three model types), it just
does not affect the generated topology when `--model_type ke`.

Note the `--k` argument is the TARGET AVERAGE DEGREE for every model,
including ke (whose active-set size is k/2). The generators in
net_functions.py pin the realized average degree to exactly k, which the
V/E/kbar columns written by extract_lfc_csv.py make auditable per graph.

Example:
    python generate_lfc.py --model_type mhk --p 0.1 --k 8 --seed 42
    python generate_lfc.py --model_type ke --p 0.1 --k 10 --nodes 240 --seed 1
'''

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import graph_tool as gt


# Generators and graph-property helpers live in scripts/net_functions.py -
# the single source of truth for every pipeline stage.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from net_functions import (          # noqa: E402
    ws_network,
    mhk_network,
    ke_network,
    nx_to_gt,
    gain_all_nodes,
    adjacency_array,
    select_nodes_by_degree,
    gains_property,
    get_local_clutsering,
    get_transitivity,
    get_ave_shortest_path,
    get_laplacian_eigenvalues,
    get_kirchhoff_index,
)


def get_recursive_graph_paths(root):
    paths = Path(Path(root)).rglob('*.gt')
    return paths

def parse_args():
    parser = argparse.ArgumentParser(description='Generate LFC networks for a given seed, model type, and probability.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for graph generation (default: random)')
    parser.add_argument('--model_type', type=str, choices=['mhk', 'ws', 'ke'], required=True, help='Network model type: mhk, ws, or ke')
    parser.add_argument('--p', type=float, required=True, help='Rewiring/edge probability value (accepted but unused for model_type=ke, whose topology has no p-dependence)')
    parser.add_argument('--k', type=int, required=True, help='Mean degree of nodes')
    parser.add_argument('--nodes', type=int, default=240, help='Number of nodes (default: 240)')
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else np.random.randint(2**63)
    n_type = args.model_type
    prob = args.p
    k = args.k

    model = 'LFC'
    nodes = [args.nodes]
    w = np.logspace(-4, 1, 100)

    for n in nodes:
        network_path = f"nets/{model}/{n}/{n_type}/{k}_seed{seed}"
        os.makedirs(network_path, exist_ok=True)

        # Explicit 3-way dispatch, so a future --model_type cannot be silently
        # swallowed by a catch-all branch. mhk and ke return networkx graphs,
        # so they cross to graph_tool here; expected_n catches a generator
        # returning the wrong node count.
        if n_type == "mhk":
            G = nx_to_gt(mhk_network(n, k, prob, seed=seed), expected_n=n)
        elif n_type == "ws":
            G = ws_network(n, k, prob, seed=seed)      # already graph_tool
        elif n_type == "ke":
            G = nx_to_gt(ke_network(n, k, seed=seed), expected_n=n)
        else:
            raise ValueError(f'unknown model_type {n_type!r}')

        G.graph_properties['ID'] = G.new_graph_property('int64_t', val=int(time.time() * 1000))
        G.graph_properties['ntype'] = G.new_graph_property('string', val=n_type)
        G.graph_properties['probability'] = G.new_graph_property('double', prob)
        G.graph_properties['seed'] = G.new_graph_property('int64_t', val=seed)
        G.graph_properties['frequencies'] = G.new_graph_property('vector<double>', val=w)

        # One eigendecomposition gives H^2 for EVERY node, so the top-5% and
        # bottom-5% sets are slices of a single computation rather than two
        # separate sweeps of dense complex solves. net_functions.get_gain is
        # the direct per-node reference implementation, and agrees with this
        # to ~1e-14.
        h2 = gain_all_nodes(adjacency_array(G), w)
        G.vertex_properties['gains_top5'] = gains_property(
            G, h2, select_nodes_by_degree(G, 'top', 5))
        G.vertex_properties['gains_bot5'] = gains_property(
            G, h2, select_nodes_by_degree(G, 'bot', 5))
        G = get_local_clutsering(G)
        G = get_transitivity(G)
        G = get_ave_shortest_path(G)
        G = get_laplacian_eigenvalues(G)
        # Deterministic name: the graph is fully identified by (k, p) within
        # its <k>_seed<seed> directory, so re-running a task overwrites in
        # place rather than leaving a second graph for the same parameter
        # point. That makes a resubmit or a backfill of a partially failed
        # array idempotent. Nothing reads .gt files by name - extract_lfc_csv.py
        # and lfc_data_loader both glob '*.gt' and key off gp.probability.
        out_path = f'{network_path}/k{k}_p{prob:.6f}.gt'
        G.save(out_path)
        print(out_path)

    print('Done all!')


if __name__ == '__main__':
    main()
