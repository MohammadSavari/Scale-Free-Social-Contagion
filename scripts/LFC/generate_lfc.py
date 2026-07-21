'''
Generates the gt files (no CSVs - use extract_lfc_csv.py to derive CSVs
from the saved .gt files afterwards).

Extended version of the original (LFC_net_gen_args.py) reproducibility
script: random seed, network model type, rewiring/edge probability (p), mean
degree (k), and node count are all passed in as command line arguments
instead of being swept over inside the script. One output graph directory
per (nodes, k) is produced, tagged with model_type/seed so that parallel
runs (e.g. sbatch array jobs) do not clobber each other.

Also computes the frequency-domain gains (ported from the original
LFC_gains_gen.py's get_gain), restricted to the top 5% and bottom 5% of
nodes by degree, and stores them as two vertex properties (gains_top5,
gains_bot5) on the same .gt file, so every property (clustering,
transitivity, shortest path, Laplacian eigenvalues, and gains) is saved
together at generation time. Only computing these two 5% subsets instead of
every node cuts the expensive linear-solve cost substantially.

`ke` (Klemm-Eguiluz-style high-clustering scale-free) networks have no
rewiring-probability parameter - ke_network(n, m, seed) only depends on
node count and mean degree, so its topology is identical across the whole
`--p` sweep for a fixed (nodes, k, seed). `--p` is still accepted and stored
as a graph property for schema consistency with mhk/ws (so the same
extraction/pooling pipeline works across all three model types), it just
does not affect the generated topology when `--model_type ke`.

Example:
    python generate_lfc.py --model_type mhk --p 0.1 --k 8 --seed 42
    python generate_lfc.py --model_type ke --p 0.1 --k 10 --nodes 240 --seed 1
'''

import argparse
import random
import numpy as np
import networkx as nx
import scipy as sp
from pathlib import Path
from scipy import sparse
from scipy import linalg
import graph_tool.clustering
import graph_tool.spectral
import graph_tool.topology
import graph_tool as gt
import os
import time


def ws_network(N, k, p, seed=None):
    """
    Function for creating a Watts-Strogatz network
    Takes inputs:
       N: int, Number of nodes
       k: integer, The mean degree of nodes
       p: Probability of rewireing of edges on the graph
       seed: Integer for the random seed
    Returns:
       G: a graphtool graph

    """
    if seed is None:
        seed = np.random.randint(2**63)

    G = gt.Graph(directed=False)
    G.add_edge_list(np.transpose(sp.sparse.tril(nx.adjacency_matrix(nx.connected_watts_strogatz_graph(N, k, p, tries=1000000, seed=seed))).nonzero()))

    return G


def mhk_network(N, k, p, seed=None):
    """
    Function for creating a Modified Holme-Kim network
    Takes inputs:
       N: int, Number of nodes
       k: integer, The mean degree of nodes
       p: Probability of rewireing of edges on the graph
       seed: Integer for the random seed
    Returns:
       G: a graphtool graph

    """
    if seed is None:
        seed = np.random.randint(2**63)
    rng = np.random.default_rng(seed)

    G = nx.connected_watts_strogatz_graph(k, 2, 0, seed=seed)

    for n in range(k, N):
        anchor = rng.choice(list(G.nodes()))
        anchor_neigh = list(G.neighbors(anchor))

        for i in range(int(k)):

            if rng.random() < p:
                G.add_edge(rng.choice(anchor_neigh), n)

            else:
                try:
                    temp = rng.choice(np.setdiff1d(G.nodes(), np.append(anchor_neigh, [anchor, n])))
                    G.add_edge(temp, n)
                except ValueError:
                    temp = rng.choice(anchor_neigh)
                    G.add_edge(temp, n)
        G.add_edge(anchor, n)

    T = gt.Graph(directed=False)
    T.add_edge_list(np.transpose(sp.sparse.tril(nx.adjacency_matrix(G)).nonzero()))

    return T


def ke_network(n, m, seed=None):
    """
    Function for creating a Klemm-Eguiluz-style high-clustering scale-free
    network (https://rf.mokslasplius.lt/acchieving-high-clustering-in-scale-free-networks/).
    Takes inputs:
       n: int, Number of nodes
       m: integer, mean degree / initial core size
       seed: Integer for the random seed
    Returns:
       G: a networkx graph (converted to graph-tool by the caller, matching
          mhk_network/ws_network's return type)
    """
    if seed is None:
        seed = np.random.randint(2**63)
    rng = random.Random(seed)

    G = nx.connected_watts_strogatz_graph(m, m, 0, seed=seed)
    active_nodes = list(G.nodes())
    for i in range(m, n):
        for k in active_nodes:
            G.add_edge(k, i)
        active_nodes.append(i)
        active_nodes.remove(rng.choice(active_nodes))

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


def get_recursive_graph_paths(root):
    paths = Path(Path(root)).rglob('*.gt')
    return paths


def get_local_clutsering(G):
    if not G.vertex_properties.get('local_clustering', False):
        G.vertex_properties['local_clustering'] = graph_tool.clustering.local_clustering(G)
    return G


def get_transitivity(G):
    if not G.gp.get('transitivity', False):
        trans = G.new_graph_property('double', val=graph_tool.clustering.global_clustering(G)[0])
        G.graph_properties['transitivity'] = trans
    return G


def get_ave_shortest_path(G):
    if not G.gp.get('shortest_path', False):
        G.gp['shortest_path'] = G.new_graph_property('double', val=np.sum(graph_tool.topology.shortest_distance(G).get_2d_array(range(G.num_vertices()))) / (G.num_vertices() * (G.num_vertices() - 1)))
    return G


def get_gain(graph, w, N, centrality=240, base='degree'):
    '''
    graph : G is a graph tool network
    w : array of the frequencies
    N : number of nodes
    centrality : choosing the top (positive) or bottom (negative) nodes with selector
    base : choose the node selector 'degree' or 'clustering'
    '''
    L = gt.spectral.laplacian(graph, norm=False)  # the built-in normalized gives the symmetric normalized laplacian, but we want the random walk normalized laplacian

    L = (L / L.diagonal()).T  # Random walk normalization  D^-1 L = LD^-1 because L is symmetric

    L = L.toarray()
    h2 = graph.new_vertex_property('vector<double>')

    if base == 'degree':
        degrees = dict(nx.from_numpy_array(gt.spectral.adjacency(graph).T.toarray()).degree())

        if centrality == N:
            Nodes = graph.vertices()
        elif centrality > 0:
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            Nodes = sorted_nodes[:centrality]
        else:
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            Nodes = sorted_nodes[centrality:]
    elif base == 'clustering':
        clustering_coeffs = graph_tool.clustering.local_clustering(graph)
        sorted_nodes = sorted(graph.vertices(), key=lambda v: clustering_coeffs[v], reverse=True)
        if centrality == N:
            Nodes = graph.vertices()
        elif centrality > 0:
            Nodes = [int(v) for v in sorted_nodes[:centrality]]
        else:
            Nodes = [int(v) for v in sorted_nodes[centrality:]]

    for g in Nodes:
        ida = np.arange(N) != g
        idb = np.arange(N) == g
        A = L[np.ix_(ida, ida)].astype(complex)
        B = L[np.ix_(ida, idb)]

        H2 = []
        for f in w:
            np.fill_diagonal(A, 1.0 + 1j * f)
            h = linalg.solve(A, -B)

            H2.append(linalg.norm(h) ** 2)

        h2[g] = H2

    return h2


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

        if n_type == "mhk":
            G = mhk_network(n, k, prob, seed=seed)
        elif n_type == "ws":
            G = ws_network(n, k, prob, seed=seed)
        else:
            G = ke_network(n, k, seed=seed)

        G.graph_properties['ID'] = G.new_graph_property('int64_t', val=int(time.time() * 1000))
        G.graph_properties['ntype'] = G.new_graph_property('string', val=n_type)
        G.graph_properties['probability'] = G.new_graph_property('double', prob)
        G.graph_properties['seed'] = G.new_graph_property('int64_t', val=seed)
        G.graph_properties['frequencies'] = G.new_graph_property('vector<double>', val=w)

        G.vertex_properties['gains_top5'] = get_gain(G, w, n, centrality=round(n * 0.05), base='degree')
        G.vertex_properties['gains_bot5'] = get_gain(G, w, n, centrality=-round(n * 0.05), base='degree')
        G = get_local_clutsering(G)
        G = get_transitivity(G)
        G = get_ave_shortest_path(G)
        G = get_laplacian_eigenvalues(G)
        # G.gp.ID alone (millisecond timestamp) can collide between concurrent
        # SLURM array tasks writing into the same network_path, silently
        # overwriting each other's file. SLURM_ARRAY_JOB_ID+SLURM_ARRAY_TASK_ID
        # is unique per task cluster-wide; fall back to PID for manual runs.
        job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
        task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        unique_suffix = f'{job_id}_{task_id}' if job_id is not None else str(os.getpid())
        out_path = f'{network_path}/{G.gp.ID}_{unique_suffix}.gt'
        G.save(out_path)
        print(out_path)

    print('Done all!')


if __name__ == '__main__':
    main()
