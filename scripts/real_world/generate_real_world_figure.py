'''
Figure-2/S15-style summary figure for the real-world networks under
../nets/real_world/ (see generate_real_world.py, which fetches and
annotates them).

One column per network found via ../nets/real_world/**/*.gt:
  - top row: collective frequency response H^2(omega) for the top-5% /
    bottom-5% degree nodes, using the same gains_top5/gains_bot5 vertex
    properties ../LFC/generate_lfc.py's get_gain() already computed
    (random-walk normalized Laplacian - left unchanged), each curve
    rescaled to its own max of 1 so the two model families are directly
    comparable regardless of raw H^2 magnitude.
  - bottom row: the *symmetric* normalized Laplacian eigenvalue spectrum
    (graph_tool laplacian(norm=True)), as a density histogram - the same
    quantity as the main figure notebook's Figure 2/9 subplot (b). This is
    computed fresh here because the eig_laplacian vertex property saved by
    generate_lfc.py is the *unnormalized* Laplacian (a different quantity,
    kept as-is for the sweep pipeline).

Example:
    python generate_real_world_figure.py
'''

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import graph_tool as gt
import graph_tool.spectral

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'LFC'))
from extract_lfc_csv import get_selected_gains, get_graph_props

PERC = 5


def characteristics_label(G):
    '''
    Same C (clustering) / T (transitivity) / l (avg. shortest path) / R_g
    (Kirchhoff index) properties the main figure notebook prints in its
    legends - computed the same way, via extract_lfc_csv.get_graph_props().
    Unlike the synthetic sweep (fixed N=240, Rg always order 1e3),
    real-world Rg spans several orders of magnitude across networks, so the
    exponent is picked per-network instead of a fixed /1e3.
    '''
    props = get_graph_props(G)
    rg = props['Rg']
    exp = int(np.floor(np.log10(abs(rg)))) if rg else 0
    mantissa = rg / 10**exp
    rg_label = rf'{mantissa:.1f}\!\times\!10^{{{exp}}}'
    return (rf'$C={props["CC"]:.2f} \,|\, T={props["T"]:.2f} \,|\, '
            rf'\ell={props["SP"]:.2f} \,|\, R_g={rg_label}$')


def load_networks(root):
    root = Path(root)
    paths = sorted(
        p for p in root.rglob('*.gt')
        # Skip dot-directories (e.g. .ns_raw_cache/) - those hold raw,
        # unannotated graphs with no category/name graph properties.
        if not any(part.startswith('.') for part in p.relative_to(root).parts)
    )
    if not paths:
        raise FileNotFoundError(f'No .gt files found under {root}')
    return [gt.load_graph(str(p)) for p in paths]


def normalized_gain_curve(G, t_b, perc=PERC):
    gains = get_selected_gains(G, t_b, perc)
    if gains is None:
        return None, None
    freqs = np.array(sorted(gains))
    h2 = np.array([gains[f] for f in freqs])
    return freqs, h2 / h2.max()


def normalized_laplacian_eigenvalues(G):
    L = gt.spectral.laplacian(G, norm=True)
    return np.linalg.eigvalsh(L.todense())


def main():
    # Same font/text setup the main figure notebook uses for every figure:
    # LaTeX-rendered text in the mathptmx (Times-like) face, so this summary
    # figure matches the paper's other plots.
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{mathptmx}')

    root = Path(__file__).resolve().parent.parent.parent / 'nets' / 'real_world'
    networks = load_networks(root)

    ncols = len(networks)
    fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(4.2 * ncols, 6.4), squeeze=False)

    for col, G in enumerate(networks):
        ax_gain = axs[0][col]
        ax_eig = axs[1][col]

        name = G.gp.name
        category = G.gp.category

        for t_b, style in [('top', 'solid'), ('bot', 'dashed')]:
            freqs, h2_norm = normalized_gain_curve(G, t_b)
            if freqs is None:
                continue
            ax_gain.plot(freqs, h2_norm, linestyle=style, label=f'{t_b} 5\\%')

        ax_gain.set_xscale('log')
        ax_gain.set_yscale('log')
        ax_gain.set_xlabel(r'Frequency $(\omega)$', labelpad=2)
        if col == 0:
            ax_gain.set_ylabel(r'Normalized response $(H^2 / H^2_{max})$', labelpad=2.5)
        ax_gain.set_title(f'{name} ({category})', fontsize=9)

        leg = ax_gain.legend(title=characteristics_label(G), loc='lower left', fontsize=7,
                              borderpad=0.2, markerscale=0.8, handlelength=2.5, handletextpad=0.4)
        leg.get_title().set_fontsize(7)
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_alpha(1.0)
        leg.get_frame().set_edgecolor('white')

        ax_gain.grid(True, which='major', ls=':')
        ax_gain.tick_params(axis='x', labelsize=7)
        ax_gain.tick_params(axis='y', labelsize=7)

        eig = normalized_laplacian_eigenvalues(G)
        ax_eig.hist(eig, bins=100, density=True, alpha=0.7)
        ax_eig.set_xlabel('Normalized Laplacian eigenvalues', labelpad=2)
        if col == 0:
            ax_eig.set_ylabel('Density', labelpad=2.5)
        ax_eig.set_xlim(-0.02, 2.05)
        ax_eig.grid(True, which='major', ls=':')
        ax_eig.tick_params(axis='x', labelsize=7)
        ax_eig.tick_params(axis='y', labelsize=7)

    fig.tight_layout()
    out_path = root / 'figs' / 'LFC_real_world_summary.pdf'
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
