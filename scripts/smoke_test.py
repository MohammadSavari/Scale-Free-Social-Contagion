'''
L0 smoke test -- gates every generation stage.

Verifies, on the NETWORKX graph before conversion (so a failure is reported
at generation time rather than after 21,400 SLURM tasks), that both new
generators satisfy the properties the whole pipeline assumes:

  * connected
  * exactly N nodes
  * no self-loops
  * 2*E == round(N*k)  -- the exact-average-degree claim

then converts with nx_to_gt and re-checks vertex/edge counts survive.

One array task per k. ke is only valid at k in {8, 16} (at k=2 its m floors
at 2, growth overshoots to <k>~4, and the correction pass would delete about
half the edges, very likely disconnecting the graph).

Usage:
    python smoke_test.py --k 16
    python smoke_test.py --all          # every k, serially (slow)
'''

import argparse
import sys

from pathlib import Path

import networkx as nx
import numpy as np

# net_functions.py lives one level up, in scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_functions import mhk_network, ke_network, nx_to_gt

MHK_K = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
KE_K = [8, 16]
KE_SIZES = [1000]
MHK_SIZES = [240, 1000]
MHK_PS = [0.0, 0.15, 0.5, 0.95, 1.0]


def check(label, G, N, k, failures):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    expected_e = int(round(N * k / 2))
    problems = []
    if n != N:
        problems.append(f'nodes {n} != {N}')
    if 2 * e != 2 * expected_e:
        problems.append(f'E {e} != {expected_e} (<k>={2*e/n:.4f}, want {k})')
    if nx.number_of_selfloops(G) != 0:
        problems.append(f'{nx.number_of_selfloops(G)} self-loops')
    if not nx.is_connected(G):
        ncc = nx.number_connected_components(G)
        problems.append(f'DISCONNECTED ({ncc} components)')

    T = nx_to_gt(G, expected_n=N)
    if T.num_vertices() != N:
        problems.append(f'gt vertices {T.num_vertices()} != {N}')
    if T.num_edges() != e:
        problems.append(f'gt edges {T.num_edges()} != {e}')

    status = 'OK  ' if not problems else 'FAIL'
    print(f'  {status} {label:34s} N={n:5d} E={e:6d} <k>={2*e/n:8.4f}'
          + ('   ' + '; '.join(problems) if problems else ''))
    if problems:
        failures.append((label, problems))


def run_for_k(k, failures):
    print(f'--- k={k} ---')
    for N in MHK_SIZES:
        for p in MHK_PS:
            try:
                G = mhk_network(N, k, p, seed=1234 + k)
                check(f'mhk N={N} p={p}', G, N, k, failures)
            except Exception as exc:
                print(f'  FAIL mhk N={N} p={p:<5} raised {type(exc).__name__}: {exc}')
                failures.append((f'mhk N={N} k={k} p={p}', [str(exc)]))
    if k in KE_K:
        for N in KE_SIZES:
            try:
                G = ke_network(N, k, seed=1234 + k)
                check(f'ke  N={N}', G, N, k, failures)
            except Exception as exc:
                print(f'  FAIL ke N={N} raised {type(exc).__name__}: {exc}')
                failures.append((f'ke N={N} k={k}', [str(exc)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    failures = []
    ks = MHK_K if args.all else [args.k]
    if args.k is None and not args.all:
        raise SystemExit('pass --k <k> or --all')
    for k in ks:
        run_for_k(k, failures)

    print()
    if failures:
        print(f'SMOKE TEST FAILED: {len(failures)} problem(s)')
        for label, problems in failures:
            print(f'  {label}: {"; ".join(problems)}')
        sys.exit(1)
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
