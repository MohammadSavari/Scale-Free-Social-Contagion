'''
Fetches real-world small-world and scale-free networks from the
Netzschleuder repository (graph_tool.collection.ns - https://networks.skewed.de)
and saves them as .gt files under real_world/, with the exact same derived
properties (local_clustering, transitivity, shortest_path, eig_laplacian,
gains_top5, gains_bot5) that ../LFC/generate_lfc.py computes for the
synthetic mhk/ws/ke sweep, so extract_real_world_csv.py can be pointed at
real_world/ without modification (e.g. `python extract_real_world_csv.py`).

10 networks are registered (small-world and scale-free real-world
counterparts to the synthetic sweep). A 'power' (Western US power grid,
N~4941) network was deliberately excluded: its gain computation is O(N^4)
in N, estimated at ~540h - impractical at this cluster's walltime limits,
so it is not part of the reproducible package (see README.md's "Known
limitations" section).

Networks pulled from netzschleuder are not guaranteed simple/connected in
general, so fetch_real_world_graph() defensively undirects, strips
self-loops/parallel edges, and extracts the largest connected component
before computing anything - this only changes the registered networks if
the upstream data changes.

Example:
    python generate_real_world.py
    python generate_real_world.py --only jazz_collab
'''

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import graph_tool as gt
import graph_tool.collection as gtc
import graph_tool.generation as gg
import graph_tool.topology as gtt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'LFC'))
from generate_lfc import (
    get_gain,
    get_local_clutsering,
    get_transitivity,
    get_ave_shortest_path,
    get_laplacian_eigenvalues,
)

REAL_WORLD_NETWORKS = {
    'jazz_collab': {
        'ns_name': 'jazz_collab',
        'category': 'small_world',
        'citation': 'Gleiser & Danon (2003), "Community Structure in Jazz"',
    },
    'celegans_metabolic': {
        'ns_name': 'celegans_metabolic',
        'category': 'scale_free',
        'citation': 'Jeong et al. (2000) / Duch & Arenas (2005)',
    },
    'celegansneural': {
        'ns_name': 'celegansneural',
        'category': 'small_world',
        'citation': 'White, Southgate, Thompson & Brenner (1986), C. elegans neurons',
    },
    'dolphins': {
        'ns_name': 'dolphins',
        'category': 'small_world',
        'citation': 'Lusseau et al. (2003), Dolphin social network',
    },
    'football': {
        'ns_name': 'football',
        'category': 'small_world',
        'citation': 'Girvan & Newman (2002), NCAA college football 2000',
    },
    'faa_routes': {
        'ns_name': 'faa_routes',
        'category': 'scale_free',
        'citation': 'US FAA, Preferred Routes (2010)',
    },
    'interactome_yeast': {
        'ns_name': 'interactome_yeast',
        'category': 'scale_free',
        'citation': 'Coulomb et al. (2005), yeast interactome',
    },
    'collins_yeast': {
        'ns_name': 'collins_yeast',
        'category': 'scale_free',
        'citation': 'Collins et al. (2007), yeast interactome',
    },
    'polblogs': {
        'ns_name': 'polblogs',
        'category': 'scale_free',
        'citation': 'Adamic & Glance (2005), Political blogs network',
    },
    'uni_email': {
        'ns_name': 'uni_email',
        'category': 'scale_free',
        'citation': 'Guimera et al. (2003), Uni. R-V email network',
    },
}


RAW_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / 'nets' / 'real_world' / '.ns_raw_cache'


def fetch_raw_graph(ns_name, cache_dir=RAW_CACHE_DIR):
    '''
    Returns the raw (uncleaned) graph_tool Graph for ns_name, from a local
    cache if present, else downloads it from Netzschleuder and populates the
    cache. SLURM compute nodes typically have no internet access (unlike the
    login node) - `graph_tool.collection.ns[...]` fails there with a
    connection timeout, so any network fetch has to happen on the login node
    ahead of time. Run this once interactively on the login node (e.g. via
    generate_real_world.py) to warm the cache before submitting
    submit_real_world_generate.sh.
    '''
    cache_path = cache_dir / f"{ns_name.replace('/', '_')}.gt"
    if cache_path.exists():
        return gt.load_graph(str(cache_path))
    G = gtc.ns[ns_name]
    cache_dir.mkdir(parents=True, exist_ok=True)
    G.save(str(cache_path))
    return G


def fetch_real_world_graph(ns_name):
    '''
    Returns a simple, undirected, unweighted graph_tool Graph restricted to
    its largest connected component (contiguous vertex ids, all extra
    property maps dropped - only the topology is kept, matching what
    mhk_network/ws_network hand back in generate_lfc.py). Fetches the raw
    network via fetch_raw_graph() (cached locally - see there).
    '''
    G = fetch_raw_graph(ns_name)
    if G.is_directed():
        G = gt.Graph(G, directed=False)

    gg.remove_parallel_edges(G)
    gg.remove_self_loops(G)

    Gc = gtt.extract_largest_component(G, directed=False, prune=True)
    T = gt.Graph(directed=False)
    T.add_edge_list(Gc.get_edges())
    return T


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch real-world networks and save them as annotated .gt files.')
    parser.add_argument('--only', type=str, choices=list(REAL_WORLD_NETWORKS), default=None,
                         help='Fetch only this one network (default: all registered networks)')
    parser.add_argument('--root', type=str, default=None,
                         help='Output root directory (default: ../../nets/real_world, alongside nets/LFC and nets/LTM)')
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root if args.root is not None else str(Path(__file__).resolve().parent.parent.parent / 'nets' / 'real_world')
    names = [args.only] if args.only else list(REAL_WORLD_NETWORKS)
    w = np.logspace(-4, 1, 100)

    for name in names:
        spec = REAL_WORLD_NETWORKS[name]
        print(f'fetching {name} ({spec["ns_name"]}) ...')
        G = fetch_real_world_graph(spec['ns_name'])
        n = G.num_vertices()

        G.graph_properties['ID'] = G.new_graph_property('int64_t', val=int(time.time() * 1000))
        G.graph_properties['ntype'] = G.new_graph_property('string', val='real_world')
        G.graph_properties['category'] = G.new_graph_property('string', val=spec['category'])
        G.graph_properties['name'] = G.new_graph_property('string', val=name)
        G.graph_properties['source'] = G.new_graph_property('string', val=f'netzschleuder:{spec["ns_name"]}')
        # No (p, seed) sweep applies to a fixed real-world graph; kept as
        # sentinels so this .gt has the same graph-property schema as the
        # synthetic sweep and extract_lfc_csv.py's (ID, p) indexing works.
        # -1.0 (not NaN): pandas silently corrupts every column of a row
        # whose MultiIndex level is NaN in extract_lfc_csv.py's
        # DataFrame.from_dict(..., orient='index') - real probabilities are
        # always in (0, 1], so -1.0 is unambiguous and sorts/prints cleanly.
        G.graph_properties['probability'] = G.new_graph_property('double', val=-1.0)
        G.graph_properties['seed'] = G.new_graph_property('int64_t', val=-1)
        G.graph_properties['frequencies'] = G.new_graph_property('vector<double>', val=w)

        n_select = max(1, round(n * 0.05))
        G.vertex_properties['gains_top5'] = get_gain(G, w, n, centrality=n_select, base='degree')
        G.vertex_properties['gains_bot5'] = get_gain(G, w, n, centrality=-n_select, base='degree')
        G = get_local_clutsering(G)
        G = get_transitivity(G)
        G = get_ave_shortest_path(G)
        G = get_laplacian_eigenvalues(G)

        out_dir = f'{root}/{spec["category"]}/{name}'
        os.makedirs(out_dir, exist_ok=True)
        out_path = f'{out_dir}/{name}.gt'
        G.save(out_path)
        print(f'  N={n} E={G.num_edges()} -> {out_path}')

    print('Done all!')


if __name__ == '__main__':
    main()
