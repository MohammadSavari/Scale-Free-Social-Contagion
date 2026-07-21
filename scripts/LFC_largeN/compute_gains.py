
'''
LFC processes a single .gt graph file and computes gains for a specific degree and a specific centrality
Saves CSV to {degree}_gain folder

Usage:
    python compute_gains.py n d c /path/to/graph.gt
    python compute_gains.py 1000 4 50 /nets/LFC/1000/mhk/4/graph_001.gt

Output:
    /nets/LFC/1000/mhk/4_gain/graph_001_c50_gains.csv
'''

import numpy as np
import networkx as nx
import scipy as sp
from pathlib import Path
from scipy import sparse
import graph_tool.clustering
import graph_tool.spectral
import graph_tool.topology
import matplotlib.pyplot as plt
import graph_tool as gt
import glob
import pandas as pd
import os
from scipy import linalg
from scipy.sparse import coo_array
import time
import random
from graph_tool.clustering import local_clustering
import argparse

# ==================== Argument Parsing ====================

parser = argparse.ArgumentParser(
    description='Process a single .gt graph file and compute LFC gains',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  python compute_gains.py 1000 4 50 /nets/LFC/1000/mhk/4/graph_001.gt
  python compute_gains.py 2000 6 100 /path/to/graph.gt
    '''
)

parser.add_argument('n', type=int, help='Number of nodes')
parser.add_argument('d', type=int, help='Average degree')
parser.add_argument('c', type=int, help='Centrality value (number of nodes to select)')
parser.add_argument('gt_file', type=str, help='Full path to the .gt graph file')

args = parser.parse_args()

# Validate arguments
if not os.path.exists(args.gt_file):
    print(f"Error: File not found: {args.gt_file}")
    exit(1)

# ==================== Helper Functions ====================

def get_laplacian_eigenvalues(G):
    """Compute and cache Laplacian eigenvalues"""
    if not G.vertex_properties.get('eig_laplacian', False):
        eig_lap = np.linalg.eigvalsh(gt.spectral.laplacian(G, norm=False).todense())
        G.vp['eig_laplacian'] = G.new_vertex_property('double', vals=eig_lap)
    return G

def get_kirchhoff_index(G):
    """Compute Kirchhoff index from Laplacian eigenvalues"""
    G = get_laplacian_eigenvalues(G)
    G.graph_properties['kirchhoff'] = G.new_graph_property(
        'int64_t',
        sum(1/np.sort(G.vp.eig_laplacian.get_array())[1:])
    )
    return G

def get_local_clustering(G):
    """Compute local clustering coefficient"""
    if not G.vertex_properties.get('local_clustering', False):
        G.vertex_properties['local_clustering'] = graph_tool.clustering.local_clustering(G)
    return G

def get_transitivity(G):
    """Compute global clustering coefficient (transitivity)"""
    if not G.gp.get('transitivity', False):
        trans = G.new_graph_property(
            'double',
            val=graph_tool.clustering.global_clustering(G)[0]
        )
        G.graph_properties['transitivity'] = trans
    return G

def get_ave_shortest_path(G):
    """Compute average shortest path length"""
    if not G.gp.get('shortest_path', False):
        G.gp['shortest_path'] = G.new_graph_property(
            'double',
            val=np.sum(graph_tool.topology.shortest_distance(G).get_2d_array(range(G.num_vertices())))
                / (G.num_vertices() * (G.num_vertices() - 1))
        )
    return G

def get_gain(graph, w, N, centrality=240, base='degree'):
    '''
    Compute H2 gains for selected nodes

    Parameters:
    -----------
    graph : graph_tool.Graph
        Graph to analyze
    w : np.array
        Array of frequencies
    N : int
        Number of nodes
    centrality : int
        Number of nodes to select (positive=top, negative=bottom)
    base : str
        Node selection method: 'degree' or 'clustering'

    Returns:
    --------
    h2 : VertexPropertyMap
        H2 gains for each selected node across frequencies
    '''
    L = gt.spectral.laplacian(graph, norm=False)
    L = (L / L.diagonal()).T  # Random walk normalization: D^-1 L
    L = L.toarray()

    h2 = graph.new_vertex_property('vector<double>')

    if base == 'degree':
        degrees = dict(nx.from_numpy_array(gt.spectral.adjacency(graph).T.toarray()).degree())

        if centrality == N:
            Nodes = list(graph.vertices())
        elif centrality > 0:
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            Nodes = sorted_nodes[:centrality]
        else:
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            Nodes = sorted_nodes[centrality:]

    elif base == 'clustering':
        clustering_coeffs = local_clustering(graph)
        sorted_nodes = sorted(graph.vertices(), key=lambda v: clustering_coeffs[v], reverse=True)

        if centrality == N:
            Nodes = list(graph.vertices())
        elif centrality > 0:
            Nodes = [int(v) for v in sorted_nodes[:centrality]]
        else:
            Nodes = [int(v) for v in sorted_nodes[centrality:]]

    # Compute H2 for each selected node
    for g in Nodes:
        ida = np.arange(N) != g
        idb = np.arange(N) == g
        A = L[np.ix_(ida, ida)].astype(complex)
        B = L[np.ix_(ida, idb)]

        H2 = []
        for f in w:
            A_copy = A.copy()
            np.fill_diagonal(A_copy, 1.0 + 1j * f)
            h = linalg.solve(A_copy, -B)
            H2.append(linalg.norm(h) ** 2)

        h2[g] = H2

    return h2

# ==================== Main Processing ====================

print("=" * 60)
print("LFC Gains Analysis - Single Graph File Processor")
print("=" * 60)
print(f"Node count (N): {args.n}")
print(f"Average degree (D): {args.d}")
print(f"Centrality (C): {args.c}")
print(f"Graph file: {args.gt_file}")
print("=" * 60)

# Load the graph
print("\nLoading graph...")
G = gt.load_graph(args.gt_file)
n_vertices = G.num_vertices()

print(f"Graph loaded: {n_vertices} vertices, {G.num_edges()} edges")

# Compute graph properties
print("Computing graph properties...")
G = get_laplacian_eigenvalues(G)
G = get_kirchhoff_index(G)
G = get_local_clustering(G)
G = get_transitivity(G)
G = get_ave_shortest_path(G)

# Define frequency range
w = np.logspace(-4, 1, 100)
frequencies = G.new_graph_property('vector<double>', val=w)
G.graph_properties['frequencies'] = frequencies

# Compute gains
print(f"Computing gains for {args.c} nodes...")
G.vertex_properties['gains'] = get_gain(G, w, args.n, args.c, base='degree')

# ==================== Output Directory Handling ====================

# Extract parent directory and create {degree}_gain folder
input_dir = os.path.dirname(args.gt_file)  # e.g., /nets/LFC/1000/mhk/4
parent_dir = os.path.dirname(input_dir)     # e.g., /nets/LFC/1000/mhk
degree_gain_folder = os.path.join(parent_dir, f"{args.d}_gain")  # e.g., /nets/LFC/1000/mhk/4_gain

os.makedirs(degree_gain_folder, exist_ok=True)

# Generate output filename
base_name = Path(args.gt_file).stem
csv_file = os.path.join(degree_gain_folder, f"{base_name}_c{args.c}_gains.csv")

# ==================== Save Results ====================

print("Saving results to CSV...")
data = {'ID': [], 'freq': [], 'prob': [], 'H2': []}

# Get graph properties
graph_id = G.gp.get('ID', 'unknown') if G.gp.get('ID') else 'unknown'
graph_prob = G.gp.get('probability', 1.0) if G.gp.get('probability') else 1.0

# Extract average gains across selected nodes
gains_2d = G.vp['gains'].get_2d_array(range(len(w)))
avg_gains = np.mean(gains_2d, axis=1)

for freq, gain in zip(w, avg_gains):
    data['ID'].append(graph_id)
    data['freq'].append(freq)
    data['prob'].append(graph_prob)
    data['H2'].append(gain)

df = pd.DataFrame(data)
df.to_csv(csv_file, sep='\t', index=False)

print("\n" + "=" * 60)
print(f"Results saved to: {csv_file}")
print(f"  Output folder: {degree_gain_folder}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print("=" * 60)
