import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pooling'))
from lfc_data_loader import pooled_eig_laplacians


def main():
    parser = argparse.ArgumentParser(
        description='Precompute and cache pooled Laplacian eigenvalues for one (network, k), '
                     'so the figure notebooks\' eigenspectrum panels do not have to scan '
                     'thousands of .gt files interactively (see submit_eig_cache.sh).')
    parser.add_argument('--network', required=True, choices=['mhk', 'ws'])
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--root', default='nets')
    args = parser.parse_args()

    pooled = pooled_eig_laplacians('LFC', 240, args.network, args.k, root=args.root)
    print(f'{args.network} k={args.k}: cached eigenvalues for {len(pooled)} p values')


if __name__ == '__main__':
    main()
