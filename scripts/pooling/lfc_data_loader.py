'''
Helpers for the figure notebooks to read LFC/LTM data that is spread across
one directory per seed (nets/{model}/{n}/{n_type}/{k}_seed{seed}/), instead
of a single directory per (model, n, n_type, k).

Each function below pools data across every seed found for a given
(model, nodes, network, k) and returns a DataFrame with the exact same
shape/index as the single-file `pd.read_csv(...)` calls the original
notebooks used to make, so downstream plotting code (which already does
`.groupby(level=...).mean()` over the pooled rows) does not need to change.

Requires the per-seed CSVs produced by extract_lfc_csv.py to already exist
(run it once after generating .gt files with generate_lfc.py).
'''

import glob
import os

import graph_tool as gt
import graph_tool.spectral
import numpy as np
import pandas as pd


def _seed_dirs(model, nodes, network, k, root='nets'):
    # `{k}_seed*` also matches the sibling CSVs extract_lfc_csv.py writes
    # (e.g. '16_seed0_props.csv' starts with '16_seed0'), not just the seed
    # directories themselves - filter to directories only, otherwise the
    # CSV paths below get built as '<...>_props.csv_props.csv' etc.
    return sorted(p for p in glob.glob(f'{root}/{model}/{nodes}/{network}/{k}_seed*') if os.path.isdir(p))


def _perc_label(perc):
    return int(perc) if float(perc).is_integer() else perc


def pooled_props(model, nodes, network, k, root='nets'):
    '''
    Pools '<seed_dir>_props.csv' across every seed found for this
    (model, nodes, network, k). Indexed by (ID, p).
    '''
    frames = []
    for seed_dir in _seed_dirs(model, nodes, network, k, root):
        frames.append(pd.read_csv(f'{seed_dir}_props.csv', sep='\t', index_col=[0, 1]))
    if not frames:
        raise FileNotFoundError(f'No props CSVs found under {root}/{model}/{nodes}/{network}/{k}_seed*_props.csv')
    return pd.concat(frames)


def pooled_gains(model, nodes, network, k, t_b, perc, centrality='degree', root='nets'):
    '''
    Pools '<seed_dir>_{t_b}_{perc}_corr_gains_{centrality}.csv' across every
    seed found for this (model, nodes, network, k). Indexed by freq, with
    ID/p/H2 as columns.

    generate_lfc.py only computes gains for the top 5% and bottom 5% of
    nodes by degree, so only (t_b, perc) in {('top', 5), ('bot', 5)} will
    find matching CSVs from the current pipeline.
    '''
    perc_label = _perc_label(perc)
    frames = []
    for seed_dir in _seed_dirs(model, nodes, network, k, root):
        path = f'{seed_dir}_{t_b}_{perc_label}_corr_gains_{centrality}.csv'
        frames.append(pd.read_csv(path, sep='\t', index_col=[1]))
    if not frames:
        raise FileNotFoundError(
            f'No gains CSVs found under {root}/{model}/{nodes}/{network}/{k}_seed*_{t_b}_{perc_label}_corr_gains_{centrality}.csv'
        )
    return pd.concat(frames)


def pooled_gains_all(model, nodes, network, k, root='nets'):
    '''
    Legacy-only: pools '<seed_dir>_gains.csv' (H2 averaged across ALL
    nodes, no top/bot selector) across every seed found for this
    (model, nodes, network, k). generate_lfc.py no longer computes gains
    for every node (only the top 5% / bottom 5% by degree, via gains_top5 /
    gains_bot5), so extract_lfc_csv.py no longer writes '_gains.csv' - this
    will raise FileNotFoundError against any data produced by the current
    pipeline. Kept only so old '_gains.csv' files from a previous run don't
    hit an ImportError.
    '''
    frames = []
    for seed_dir in _seed_dirs(model, nodes, network, k, root):
        frames.append(pd.read_csv(f'{seed_dir}_gains.csv', sep='\t', index_col=[1]))
    if not frames:
        raise FileNotFoundError(f'No gains CSVs found under {root}/{model}/{nodes}/{network}/{k}_seed*_gains.csv')
    return pd.concat(frames)


def pooled_polarization(model, nodes, network, k, t_b, perc, root='nets'):
    '''
    Pools '<seed_dir>_{selector_label}.csv' (run_ltm_cascade.py output)
    across every seed found for this (model, nodes, network, k). Columns
    are ID, network, p, th, seed, <cascade sizes 0.1..0.9>, plus one extra
    column, 'realization', not present in the original single-file schema
    (see below) - no index set; callers set_index(['ID','network','p','th',
    'seed']) same as the original notebook cells do.

    Because the per-seed-node 'seed' column is not part of the groupby keys
    those cells use (['p','th','network']), concatenating every realization's
    rows here is sufficient to pool across realizations too - no separate
    aggregation step needed before the notebook's own
    `.groupby(['p','th','network']).agg(custom_mean)` (mean) or an
    equivalent `.std()` call for a +/- band.

    'realization' is the seed directory's own basename (e.g. '16_seed13'),
    added here as a realization identifier that's safe to group by - unlike
    the graph's 'ID' column (an int(time.time()*1000) millisecond epoch
    timestamp stamped at generation time by generate_ltm.py), which can
    collide between two different seeds generated in the same millisecond
    under concurrent SLURM array submission. Confirmed via inventory: 9 such
    collisions out of 400 graph-writes in the mhk sweep (100 seeds x 4
    p-values), each silently merging two distinct realizations' selected-
    node rows into one group wherever code grouped by 'ID' - undercounting
    e.g. p=0.0's realization count from 100 to 98. 'realization' is just the
    (already-unique, filesystem-derived) seed directory name, so it can't
    collide the way a coarse timestamp can.
    '''
    t_b_norm = t_b if t_b != 'bot' else 'bot'
    sign = '-' if t_b_norm == 'bot' else ''
    perc_label = _perc_label(perc)
    selector_label = f'{t_b_norm}{sign}{perc_label}' if t_b_norm == 'bot' else f'{t_b_norm}{perc_label}'

    frames = []
    for seed_dir in _seed_dirs(model, nodes, network, k, root):
        path = f'{seed_dir}_{selector_label}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
            df['realization'] = os.path.basename(seed_dir)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f'No polarization CSVs found under {root}/{model}/{nodes}/{network}/{k}_seed*_{selector_label}.csv'
        )
    return pd.concat(frames, ignore_index=True)


def find_graph_for_p(model, nodes, network, k, p, root='nets', tol=1e-9):
    '''
    Loads one representative .gt file (from any seed) whose stored
    'probability' graph property is closest to p.
    '''
    best_path, best_diff, best_graph = None, None, None
    for seed_dir in _seed_dirs(model, nodes, network, k, root):
        for path in glob.glob(f'{seed_dir}/*.gt'):
            G = gt.load_graph(path)
            diff = abs(G.gp.probability - p)
            if best_diff is None or diff < best_diff:
                best_path, best_diff, best_graph = path, diff, G
            if diff <= tol:
                return G
    if best_path is None:
        raise FileNotFoundError(f'No .gt files found under {root}/{model}/{nodes}/{network}/{k}_seed*/')
    return best_graph


def _eig_cache_path(model, nodes, network, k, root='nets'):
    return f'{root}/{model}/{nodes}/{network}/{k}_eig_cache.npz'


def pooled_eig_laplacians(model, nodes, network, k, root='nets', decimals=6, use_cache=True):
    '''
    One pass over every .gt file for this (model, nodes, network, k),
    grouping normalized-Laplacian eigenvalues by their stored 'probability'
    graph property (rounded to `decimals` - the generation sweep derives p
    identically for every seed via np.linspace(0.001, 1, 100)[P_IDX], so
    realizations that target the same p produce bit-identical-up-to-float-
    roundtrip values).

    Lets the eigenspectrum panel pool eigenvalues across every seed
    realization that shares a given p, instead of reading a single
    representative graph. Returns {p: concatenated eigenvalue array}.

    This opens every individual .gt file for (model, nodes, network, k)
    (potentially thousands for a full seed sweep) - measured minutes of wall
    time when tried interactively on a login node, almost entirely Lustre
    I/O wait. Do not call this interactively; run precompute_eig_cache.py
    via submit_eig_cache.sh on a compute node instead, which populates the
    on-disk cache this function checks first (`use_cache=True`, the
    default) so later calls - e.g. from the figures notebook - are instant.
    '''
    cache_path = _eig_cache_path(model, nodes, network, k, root)
    if use_cache and os.path.exists(cache_path):
        with np.load(cache_path) as data:
            return {float(key[2:]): data[key] for key in data.files}

    groups = {}
    for seed_dir in _seed_dirs(model, nodes, network, k, root):
        for path in glob.glob(f'{seed_dir}/*.gt'):
            G = gt.load_graph(path)
            p = round(float(G.gp.probability), decimals)
            eig = np.linalg.eigvalsh(graph_tool.spectral.laplacian(G, norm=True).todense())
            groups.setdefault(p, []).append(eig)
    if not groups:
        raise FileNotFoundError(f'No .gt files found under {root}/{model}/{nodes}/{network}/{k}_seed*/')
    pooled = {p: np.concatenate(v) for p, v in groups.items()}

    if use_cache:
        np.savez(cache_path, **{f'p_{p:.6f}': arr for p, arr in pooled.items()})

    return pooled
