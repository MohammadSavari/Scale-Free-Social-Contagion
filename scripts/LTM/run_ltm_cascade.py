'''
Runs the LTM cascade simulation on ONE realization seed's .gt files under
nets/LTM/1000/<net>/<k>_seed<seed>/ - producing a polarization-speed CSV
per (net, k, seed, selector), so LTM figures can be pooled across seeds
with +/- std bands the same way the LFC realizations are pooled.

Only the p-values already present in that seed's directory are used (the
4-5 "pick_props" values generate_ltm.py generated - see that script's
docstring), not a dense p-sweep.

Output: nets/LTM/1000/<net>/<k>_seed<seed>_<selector><percentage>.csv
  columns: ID, network, p, th, seed(node), <cascade sizes 0.1..0.9>

Usage (one (net, k, seed) combo per call, for SLURM array parallelism):
    python run_ltm_cascade.py --net ws --k 16 --seed 0
    python run_ltm_cascade.py --net ws --k 16 --seed 0 --selectors top:5 bot:5
'''

import argparse
import random
import time
from pathlib import Path

import graph_tool as gt
import graph_tool.spectral
import networkx as nx
import numpy as np
import pandas as pd

CASCADES = np.round(np.linspace(0.1, 0.9, 9), 1)
THRESHOLD = np.linspace(0.01, 0.5, 16)


def linear_threshold_model(G, threshold, seed_nodes=None, init_spread=True, max_iter=None):
    '''
    Linear threshold model cascade simulation.
    '''
    if seed_nodes is None:
        seed_nodes = [x for x in np.random.choice(G.get_vertices(), 1)]

    if not isinstance(seed_nodes, list):
        seed_nodes = np.random.choice(G.get_vertices(), seed_nodes)

    if max_iter is None:
        max_iter = G.num_vertices()

    if not isinstance(threshold, list):
        threshold = [threshold]

    infections = []
    degree_dist = G.get_out_degrees(G.get_vertices())
    T = np.array((graph_tool.spectral.adjacency(G).T.toarray() / degree_dist).T)

    for th in threshold:
        infected = np.zeros(G.num_vertices(), dtype=int)
        infection_step = np.full(G.num_vertices(), np.inf, dtype=float)

        infected[seed_nodes] = 1
        infection_step[seed_nodes] = -1

        if init_spread:
            infected[T.dot(infected) > 0] = 1
            infection_step[np.logical_and(infected > 0, np.isinf(infection_step))] = 0
            i = 1
        else:
            i = 0
        while (not all(infected) and (i < max_iter) and i - 1 in infection_step):
            infected[T.dot(infected) >= th] = 1
            infection_step[np.logical_and(infected > 0, np.isinf(infection_step))] = i
            i += 1
        infected_step = G.new_vp(value_type='int', vals=infection_step)
        infections.append(infected_step)

    infected_vectormap = gt.group_vector_property(infections)
    return infected_vectormap, seed_nodes


def select_seed_nodes(G, t_b, perc):
    '''
    t_b: 'top' (highest-degree), 'bot' (lowest-degree), or 'rand'.
    perc: percentage of nodes (e.g. 5 -> 5%).
    '''
    degrees = dict(nx.from_numpy_array(gt.spectral.adjacency(G).T.toarray()).degree())
    num_nodes = max(1, int(len(degrees) * abs(perc) / 100))
    if t_b == 'top':
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        return sorted_nodes[:num_nodes]
    elif t_b == 'bot':
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        return sorted_nodes[-num_nodes:]
    elif t_b == 'rand':
        return random.sample(list(degrees.keys()), num_nodes)
    raise ValueError(f"Unknown t_b '{t_b}'")


def run_one_graph(graph_path, t_b, perc):
    G = gt.load_graph(str(graph_path))
    seed_nodes = select_seed_nodes(G, t_b, perc)

    rows = []
    for node in seed_nodes:
        infected_vectormap, _ = linear_threshold_model(G, list(THRESHOLD), seed_nodes=[node])
        spread = gt.ungroup_vector_property(infected_vectormap, range(len(THRESHOLD)))
        for idx, th in enumerate(THRESHOLD):
            speeds = []
            cascade_sizes = CASCADES
            infected = 0
            val, counts = np.unique(spread[idx].a, return_counts=True)
            counts = counts / G.num_vertices()
            for i, new in enumerate(counts[val > -2]):
                infected += new
                while len(cascade_sizes) > 0 and infected > cascade_sizes[0] and val[i] > 0:
                    speeds.append(infected / val[i])
                    cascade_sizes = cascade_sizes[1:]
            nan_padding = len(CASCADES) - len(speeds)
            speeds = np.pad(speeds, (0, nan_padding), constant_values=np.nan)
            rows.append([G.gp.ID, G.gp.ntype, G.gp.probability, th, node] + list(speeds))
    return rows


def parse_selector(spec):
    t_b, perc = spec.split(':')
    return t_b, float(perc)


def selector_label(t_b, perc):
    perc_label = int(perc) if float(perc).is_integer() else perc
    sign = '-' if t_b == 'bot' else ''
    return f'{t_b}{sign}{perc_label}' if t_b == 'bot' else f'{t_b}{perc_label}'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--net', choices=['ws', 'mhk', 'ke'], required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True, help='Realization seed (0-99).')
    parser.add_argument('--root', type=str, default='nets')
    parser.add_argument('--selectors', type=str, nargs='+', default=['top:5'],
                         help="Node selectors as 't_b:percentage', e.g. top:5 bot:5")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_dir = Path(args.root) / 'LTM' / '1000' / args.net / f'{args.k}_seed{args.seed}'
    graph_paths = sorted(seed_dir.glob('*.gt'))
    if not graph_paths:
        raise FileNotFoundError(f'No .gt files found in {seed_dir}')

    cols = ['ID', 'network', 'p', 'th', 'seed'] + CASCADES.astype(str).tolist()
    for t_b, perc in [parse_selector(s) for s in args.selectors]:
        start = time.time()
        rows = []
        for graph_path in graph_paths:
            rows.extend(run_one_graph(graph_path, t_b, perc))
        df = pd.DataFrame(rows, columns=cols)

        out_path = seed_dir.parent / f'{args.k}_seed{args.seed}_{selector_label(t_b, perc)}.csv'
        df.to_csv(out_path, sep='\t', index=False)
        elapsed = time.time() - start
        print(f'{out_path} saved ({len(df)} rows, {elapsed:.1f}s)')


if __name__ == '__main__':
    main()
