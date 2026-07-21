'''
Scans a directory tree for .gt files produced by generate_lfc.py and
rebuilds the per-directory CSVs that used to be written directly by the
generation script:
  - '<directory>_props.csv': one row per graph (CC, T, SP, l2, lmax_l2, Rg),
    indexed by (ID, p).
  - '<directory>_{t_b}_{perc}_corr_gains_degree.csv' for each (t_b, perc) in
    --selectors (default top-5%, bot-5%) - H2 averaged over the top/bottom
    X% of nodes by degree, indexed by (ID, freq, p). Matches the naming the
    figure notebooks expect (f'{k}_{t_b}_{perc}_corr_gains_{centrality}.csv').

generate_lfc.py only computes gains for the top 5% and bottom 5% of nodes by
degree (stored as the gains_top5 / gains_bot5 vertex properties), not for
every node, so only t_b:perc combinations matching what was actually
computed there (top:5, bot:5) will produce output here - anything else is
skipped with a warning. Node selection and gain averaging are both derived
from properties already stored on each .gt file - nothing is recomputed via
the expensive linear solves.

Example:
    python extract_lfc_csv.py --root nets
    python extract_lfc_csv.py --root nets --selectors top:5 bot:5
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
        'l2': eig[1],
        'lmax_l2': np.max(eig) / eig[1],
        'Rg': G.num_vertices() * np.sum(1 / eig[1:]),
    }


def select_nodes_by_degree(G, t_b, perc):
    '''
    t_b : 'top' (highest-degree nodes) or 'bot' (lowest-degree nodes)
    perc : percentage of nodes to select
    '''
    degrees = G.get_total_degrees(G.get_vertices())
    order = np.argsort(-degrees, kind='stable')
    n_select = max(1, round(len(degrees) * perc / 100))
    if t_b == 'top':
        return order[:n_select]
    elif t_b == 'bot':
        return order[-n_select:]
    raise ValueError(f"Unknown t_b '{t_b}', expected 'top' or 'bot'")


def gains_property_name(t_b, perc):
    perc_label = int(perc) if float(perc).is_integer() else perc
    return f'gains_{t_b}{perc_label}'


def get_selected_gains(G, t_b, perc):
    '''
    Returns None if this graph doesn't have a gains_{t_b}{perc} vertex
    property (i.e. this selector wasn't computed at generation time).
    '''
    prop_name = gains_property_name(t_b, perc)
    if not G.vertex_properties.get(prop_name, False):
        return None
    freqs = list(G.gp.frequencies)
    gains_2d = G.vp[prop_name].get_2d_array(range(len(freqs)))
    selected = select_nodes_by_degree(G, t_b, perc)
    mean_gains = np.mean(gains_2d[:, selected], axis=1)
    return dict(zip(freqs, mean_gains))


def parse_selector(spec):
    t_b, perc = spec.split(':')
    return t_b, float(perc)


def parse_args():
    parser = argparse.ArgumentParser(description='Regenerate network property and gains CSVs from saved .gt files.')
    parser.add_argument('--root', type=str, default='nets', help='Root directory to recursively search for .gt files')
    parser.add_argument('--selectors', type=str, nargs='+', default=['top:5', 'bot:5'],
                         help="Node selectors as 't_b:percentage', e.g. top:5 bot:5 (must match what was computed at generation time)")
    parser.add_argument('--k', type=int, nargs='+', default=None,
                         help='Restrict to these average-degree values (matches the <k>_seed<seed> directory prefix); '
                              'default: process every k found under --root. Lets a large sweep be staged in pieces '
                              '(e.g. one k now, the rest via a separate batch job) without reprocessing what is already done.')
    return parser.parse_args()


def main():
    args = parse_args()
    selectors = [parse_selector(s) for s in args.selectors]
    k_filter = set(args.k) if args.k is not None else None

    by_dir = {}
    for path in Path(args.root).rglob('*.gt'):
        # Skip dot-directories (e.g. real_world/.ns_raw_cache/) - those hold
        # raw, unannotated graphs (no ID/probability/gains properties) that
        # were never meant to be scanned here, and crash the dict comps
        # below with a KeyError on G.gp.ID if included.
        if any(part.startswith('.') for part in path.relative_to(args.root).parts):
            continue
        if k_filter is not None:
            # Directory name is '<k>_seed<seed>' (see generate_lfc.py); non-matching
            # directory name shapes (e.g. real_world/) are left alone by falling through.
            k_str = path.parent.name.split('_seed')[0]
            if not k_str.isdigit() or int(k_str) not in k_filter:
                continue
        by_dir.setdefault(path.parent, []).append(path)

    for directory, paths in sorted(by_dir.items()):
        # (path, G) pairs, not just G: G.gp.ID is a millisecond timestamp and
        # collides across graphs generated in the same millisecond (confirmed
        # only 90/100 unique IDs in a 100-p sweep) - path.stem (the .gt
        # filename, which generate_lfc.py already makes genuinely unique via
        # a SLURM job/task or pid suffix) is used as the real ID instead.
        graphs = [(path, gt.load_graph(str(path))) for path in paths]

        prop_rows = {(path.stem, G.gp.probability): get_graph_props(G) for path, G in graphs}
        network_props = pd.DataFrame.from_dict(prop_rows, orient='index', columns=['CC', 'T', 'SP', 'l2', 'lmax_l2', 'Rg'])
        network_props.index.names = ['ID', 'p']

        spec_file = f'{directory}_props.csv'
        network_props.to_csv(spec_file, sep='\t', mode='w', header=True)
        print(spec_file)

        graphs_with_freqs = [(path, G) for path, G in graphs if G.gp.get('frequencies', False)]

        for t_b, perc in selectors:
            perc_label = int(perc) if float(perc).is_integer() else perc
            selector_rows = {}
            for path, G in graphs_with_freqs:
                gains = get_selected_gains(G, t_b, perc)
                if gains is None:
                    continue
                for freq, h2 in gains.items():
                    selector_rows[(path.stem, freq, G.gp.probability)] = {'H2': h2}
            if not selector_rows:
                print(f'{directory}: skipping {t_b}:{perc_label} - not computed at generation time')
                continue
            selector_gains = pd.DataFrame.from_dict(selector_rows, orient='index', columns=['H2'])
            selector_gains.index.names = ['ID', 'freq', 'p']
            selector_file = f'{directory}_{t_b}_{perc_label}_corr_gains_degree.csv'
            selector_gains.to_csv(selector_file, sep='\t', mode='w', header=True)
            print(selector_file)

    print('Done all!')


if __name__ == '__main__':
    main()
