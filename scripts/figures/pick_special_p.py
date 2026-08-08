'''
Pick the `specialp` indices for the LFC figures, so that the 4 plotted
curves have CC values spread evenly across the reachable range.

Why
---
The notebook's LFC panels plot 4 of the 100 p-values, and their legend is
keyed on C | T | l | R_g. p itself is not evenly spaced in clustering, so
picking indices by hand tends to give bunched, near-duplicate C values,
which defeats the point of the panel. Choosing the indices from the
realized CC(p) curve keeps the four curves visibly distinct.

What this does
--------------
For each (net, k) it reproduces the notebook's p-ordering exactly:

    prop_special = props.groupby(level=1).mean()      # level 1 == p, sorted
    gain_special.index.get_level_values(0).unique()   # same ascending p

then places 4 targets evenly across the realized CC range and reports the
index whose CC is closest to each target. It also cross-checks that
get_sorted_filenames() lists the .gt files in that same ascending-p order,
because the notebook indexes BOTH with the same `specialp`.

Usage:
    python pick_special_p.py                 # mhk/ws at k=8,16 plus ke
    python pick_special_p.py --nets mhk ws --ks 16
'''

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# cwd-relative, like every other script's --root default: run from the repo
# root, where nets/ lives.
ROOT = Path('nets')
NODES = 240
MODEL = 'LFC'
N_TARGETS = 4


def get_sorted_filenames(directory):
    '''Same natural sort the notebook uses (cell 1).'''
    def sort_key(filename):
        stem = os.path.splitext(filename)[0]
        parts = re.split(r'(\d+)', stem)
        return [int(part) if part.isdigit() else part for part in parts]
    return [os.path.splitext(f)[0]
            for f in sorted(os.listdir(directory), key=sort_key)]


def cc_curve(net, k, seed=1):
    '''(p_array, cc_array) in the notebook's ascending-p order.'''
    f = ROOT / MODEL / str(NODES) / net / f'{k}_seed{seed}_props.csv'
    if not f.exists():
        return None, None
    props = pd.read_csv(f, sep='\t', index_col=[0, 1])
    # groupby(level=1) is exactly what the notebook does; it sorts by p
    per_p = props.groupby(level=1)['CC'].mean()
    return per_p.index.to_numpy(), per_p.to_numpy()


def closest_indices(cc, targets):
    '''Indices whose CC is closest to each target, without reuse.

    Distinctness matters: two targets collapsing onto one index would
    silently plot three curves where the legend promises four.
    '''
    picks, used = [], set()
    for t in targets:
        order = np.argsort(np.abs(cc - t))
        j = next(int(i) for i in order if int(i) not in used)
        used.add(j)
        picks.append(j)
    return picks


def even_cc_targets(cc, n_targets=N_TARGETS):
    '''n_targets CC values spread evenly across the realized range, ascending.'''
    lo, hi = float(np.min(cc)), float(np.max(cc))
    return [lo + (hi - lo) * i / (n_targets - 1) for i in range(n_targets)]


def check_file_order(net, k, ps, seed=1):
    '''The notebook indexes sorted_filenames with the SAME specialp, so the
    filename order must agree with the ascending-p order.'''
    d = ROOT / MODEL / str(NODES) / net / f'{k}_seed{seed}'
    if not d.is_dir():
        return 'no .gt directory'
    names = get_sorted_filenames(d)
    if len(names) != len(ps):
        return f'MISMATCH: {len(names)} .gt files vs {len(ps)} p-values'
    parsed = [float(n.split('_p')[-1]) for n in names]
    bad = [(i, parsed[i], ps[i]) for i in range(len(ps))
           if abs(parsed[i] - ps[i]) > 1e-6]
    return 'ok' if not bad else f'MISMATCH at {bad[:3]}'


def report(net, k, targets=None, target_src=''):
    '''Report the picks for one (net, k), ordered LOW CC -> HIGH CC.

    `targets` overrides the default even spacing -- used to point ws at
    mhk's realized CC values so the two networks are compared at matched
    clustering rather than at matched p (p means different things in the
    two generators, so equal p is not a comparison).
    '''
    ps, cc = cc_curve(net, k)
    if ps is None:
        print(f'--- {net} k={k}: no props CSV, skipped')
        return None
    if len(ps) < N_TARGETS:
        print(f'--- {net} k={k}: only {len(ps)} p-value(s) '
              f'(CC={cc.round(4).tolist()}) -- specialp not applicable')
        return None

    if targets is None:
        targets = even_cc_targets(cc)
        target_src = 'even spacing over own range'
    targets = sorted(targets)
    picks = closest_indices(cc, targets)

    print(f'--- {net} k={k}  ({len(ps)} p, CC range '
          f'[{cc.min():.4f}, {cc.max():.4f}])  targets: {target_src}')
    print(f'    .gt order vs p order: {check_file_order(net, k, ps)}')
    print(f'    {"idx":>5} {"p":>10} {"CC":>8} {"target":>8} {"err":>8}')
    for j, t in zip(picks, targets):
        print(f'    {j:>5} {ps[j]:>10.6f} {cc[j]:>8.4f} {t:>8.4f} '
              f'{abs(cc[j] - t):>8.4f}')
    gaps = [round(cc[picks[i + 1]] - cc[picks[i]], 4) for i in range(len(picks) - 1)]
    print(f'    specialp = {picks}   # low->high CC, gaps {gaps}')
    return picks, [float(cc[j]) for j in picks]


def main():
    global ROOT
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ks', nargs='+', type=int, default=[16, 8])
    ap.add_argument('--root', type=str, default=str(ROOT),
                    help='Root directory holding LFC/ (default: nets, relative to cwd)')
    a = ap.parse_args()

    ROOT = Path(a.root)

    out = {}
    for k in a.ks:
        # mhk sets the reference CC values (its reachable range is the
        # narrower of the two, so it is the binding constraint)...
        got = report('mhk', k)
        if not got:
            continue
        picks, mhk_cc = got
        out[('mhk', k)] = picks
        # ...and ws is matched to them, one index per mhk CC value.
        got_ws = report('ws', k, targets=mhk_cc, target_src="mhk's realized CC")
        if got_ws:
            out[('ws', k)] = got_ws[0]
        report('ke', k)

    print('\n=== paste into the notebook ===')
    for (net, k), picks in sorted(out.items()):
        print(f"    ('{net}', {k:>2}): {picks},")


if __name__ == '__main__':
    main()
