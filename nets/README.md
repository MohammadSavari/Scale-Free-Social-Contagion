# `nets/` - network data

This directory ships empty on purpose. It is where the network data lives,
and you populate it one of two ways.

**Unpack it** from the chunked archives in `../data/`, run from the repo
root:

```bash
bash reassemble.sh all            # every archive
bash reassemble.sh LFC_240_ws     # or just one
```

**Or regenerate it** from scratch with the pipeline in `../scripts/` - see
[../scripts/README.md](../scripts/README.md) for the stage order and the
SLURM submission scripts. Both routes produce the same layout.

## Layout

```
nets/
├── LFC/                                # spectral-gain sweep
│   ├── 240/{ws,mhk,ke}/
│   ├── 1000/mhk/
│   └── 5000/mhk/
│       └── <k>_seed<seed>/*.gt         # one graph per (k, p, seed)
│           <k>_seed<seed>_props.csv    # CC, T, SP, l2, lmax_l2, Rg (+ V, E, kbar)
│           <k>_seed<seed>_{top,bot}_5_corr_gains_degree.csv
│
├── LTM/1000/{ws,mhk,ke}/               # linear-threshold cascade realizations
│   ├── <k>_seed<seed>/p<p>.gt
│   ├── <k>_seed<seed>_props.csv
│   ├── <k>_seed<seed>_{top5,top10,bot-5,bot-10}.csv
│   └── <k>_p<p>_eig_norm.npy           # pooled normalized-Laplacian spectra
│
└── real_world/{scale_free,small_world}/
    ├── <name>/<name>.gt
    ├── <name>_props.csv
    └── <name>_{top,bot}_5_corr_gains_degree.csv
```

Every `.gt` file carries its parameters as graph properties (`ntype`,
`probability`, `seed`, `frequencies`) plus the computed structural
properties and gains. **Nothing reads a `.gt` file by name** - the readers
glob `*.gt` and key off the stored `probability`, so the two filename
conventions present (`k<k>_p<p>.gt` and `<ID>_<jobid>_<taskid>.gt`) are
interchangeable. Never index a sorted `.gt` listing positionally.

Contents of this directory other than this file are not tracked by git.
