'''
Generates independent realizations of the LTM .gt networks actually used
by the figure notebooks - just the (network_type, k, p) triples each
notebook's pick_props dict selects, not a full dense p sweep.

Combos and picked p-values were derived by indexing each network's
generation-time probability array with the pick_props indices, which are
identical across every figure notebook found:
    pick_props = {'mhk': [99,60,10,0], 'ws': [0,15,23,30,65], 'ke': [0,10,30,20]}
  - ws  (k=16 and k=8): probabilities = linspace(0, 1, 101)
  - mhk (k=16 and k=8): probabilities = [0] + linspace(0.9, 1, 100)
  - ke  (k=16 and k=8): same array as mhk - ke_network doesn't structurally
                         depend on p (see ke_network below), but the
                         existing single-realization pipeline still labels
                         each graph with one of these values, so that's
                         preserved here for parity.

Realizations are written to nets/LTM/1000/<net>/<k>_seed<n>/, mirroring
the nets/LFC/240/<net>/<k>_seed<n>/ realization layout used for LFC.

Usage (one realization per call, for SLURM array parallelism - see
submit_ltm_generate.sh):
    python generate_ltm.py --seed 0
'''

import argparse
import random
import time
from pathlib import Path

import numpy as np
import networkx as nx
import scipy as sp
import graph_tool as gt
import graph_tool.clustering
import graph_tool.spectral
import graph_tool.topology


def ws_network(N, k, p, seed=None):
    if seed is None:
        seed = np.random.randint(2**63)
    G = gt.Graph(directed=False)
    G.add_edge_list(np.transpose(sp.sparse.tril(nx.adjacency_matrix(
        nx.connected_watts_strogatz_graph(N, k, p, tries=1000000, seed=seed))).nonzero()))
    return G


def mhk_network(N, k, p, seed=None):
    '''
    Includes an np.random.seed(seed) call so each of the realizations is
    actually reproducible/independent instead of depending on call order.
    '''
    if seed is not None:
        np.random.seed(seed)
    G = nx.connected_watts_strogatz_graph(k, 2, 0)

    for n in range(k, N):
        anchor = np.random.choice(list(G.nodes()))
        anchor_neigh = list(G.neighbors(anchor))

        for i in range(int(k)):
            if np.random.random() < p:
                G.add_edge(np.random.choice(anchor_neigh), n)
            else:
                try:
                    temp = np.random.choice(np.setdiff1d(G.nodes(), np.append(anchor_neigh, [anchor, n])))
                    G.add_edge(temp, n)
                except Exception:
                    temp = np.random.choice(anchor_neigh)
                    G.add_edge(temp, n)
        G.add_edge(anchor, n)

    T = gt.Graph(directed=False)
    T.add_edge_list(np.transpose(sp.sparse.tril(nx.adjacency_matrix(G)).nonzero()))
    return T


def ke_network(n, m, seed=None):
    '''
    Includes a random.seed(seed) call so each realization is reproducible.
    '''
    if seed is not None:
        random.seed(seed)
    G = nx.connected_watts_strogatz_graph(m, m, 0)
    active_nodes = list(G.nodes())
    for i in range(m, n):
        for k in active_nodes:
            G.add_edge(k, i)
        active_nodes.append(i)
        active_nodes.remove(random.choice(active_nodes))

    T = gt.Graph(directed=False)
    T.add_edge_list(np.transpose(sp.sparse.tril(nx.adjacency_matrix(G)).nonzero()))
    return T


def get_laplacian_eigenvalues(G):
    if not G.vertex_properties.get('eig_laplacian', False):
        eig_lap = np.linalg.eigvalsh(gt.spectral.laplacian(G, norm=False).todense())
        G.vp['eig_laplacian'] = G.new_vertex_property('double', vals=eig_lap)
    return G


def get_kirchhoff_index(G):
    G = get_laplacian_eigenvalues(G)
    G.graph_properties['kirchhoff'] = G.new_graph_property('int64_t', sum(1 / np.sort(G.vp.eig_laplacian.get_array())[1:]))
    return G


def get_local_clutsering(G):
    if not G.vertex_properties.get('local_clustering', False):
        G.vertex_properties['local_clustering'] = graph_tool.clustering.local_clustering(G)
    return G


def get_transitivity(G):
    if not G.gp.get('transitivity', False):
        G.graph_properties['transitivity'] = G.new_graph_property('double', val=graph_tool.clustering.global_clustering(G)[0])
    return G


def get_ave_shortest_path(G):
    if not G.gp.get('shortest_path', False):
        G.gp['shortest_path'] = G.new_graph_property(
            'double',
            val=np.sum(graph_tool.topology.shortest_distance(G).get_2d_array(range(G.num_vertices())))
            / (G.num_vertices() * (G.num_vertices() - 1)))
    return G


N = 1000
PICK_PROPS = {'ws': [0, 15, 23, 30, 65], 'mhk': [99, 60, 10, 0], 'ke': [0, 10, 30, 20]}
PROB_ARRAYS = {
    'ws': np.linspace(0, 1, 101),
    'mhk': np.append([0], np.linspace(0.9, 1, 100)),
    'ke': np.append([0], np.linspace(0.9, 1, 100)),
}
COMBOS = [('ws', 16), ('ws', 8), ('mhk', 16), ('ke', 16), ('mhk', 8), ('ke', 8)]

cascades = np.round(np.linspace(0.1, 0.9, 9), 1)


def generate_one(net, k, p, seed):
    if net == 'ws':
        G = ws_network(N, k, p, seed=seed)
    elif net == 'mhk':
        G = mhk_network(N, k, p, seed=seed)
    elif net == 'ke':
        G = ke_network(N, k, seed=seed)
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
    return parser.parse_args()


def main():
    args = parse_args()

    for net, k in COMBOS:
        out_dir = Path(args.root) / 'LTM' / str(N) / net / f'{k}_seed{args.seed}'
        out_dir.mkdir(parents=True, exist_ok=True)

        probs = PROB_ARRAYS[net][PICK_PROPS[net]]
        for p in probs:
            out_path = out_dir / f'p{p:.6f}.gt'
            if out_path.exists():
                print(f'\r{out_path} exists, skipping', end='', flush=True)
                continue
            # Offset the seed per p-value so the 4-5 graphs within one
            # realization/network don't share generation randomness.
            graph_seed = args.seed * 1000 + int(round(p * 1000))
            G = generate_one(net, k, p, graph_seed)
            G.save(str(out_path))
            print(f'\r{out_path} saved', end='', flush=True)
    print('\nDone.')


if __name__ == '__main__':
    main()
