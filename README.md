# Scale-Free-Social-Contagion

Data-availability and reproducibility for 

```bash
"How social contagions shape collective consensus in the presence of scale-free networks"
```

<p align="center">
  <img src="figures/previews/Figure_S21.png" width="60%" alt="Figure S21 - leader-follower response + LTM cascade illustration">
</p>

<p align="center">Figure S21: leader-follower response + LTM cascade illustration (<a href="figures/Figure_S21.pdf">PDF</a>).</p>

## Environment

See [requirements.txt](requirements.txt) for the exact package versions used.
On a local machine, `graph-tool` is easiest to get via conda-forge (it's
not pip-installable) - the rest can come from either conda or pip. 

```bash
conda create -n scale-free-social-contagion -c conda-forge python=3.10 graph-tool=2.56 \
    numpy=1.25.2 pandas=2.1.0 scipy=1.11.2 networkx=3.4.2 matplotlib=3.7.2
conda activate scale-free-social-contagion
pip install ipykernel  # only needed to run the notebooks under scripts/figures/
```

(If you're instead on the HPC cluster this was originally produced on,
the same packages come from environment modules instead:
`module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10`.)

## Pipeline

`nets/` holds the network data. It is either **unpacked** from the archives
in `data/` (below) or **regenerated** from scratch by the pipeline in
`scripts/` - see [scripts/README.md](scripts/README.md) for the full
stage-by-stage breakdown and run order. Either route produces the same
layout, and `figures/` follows from it.

## Data

`nets/` is several GB spread over tens of thousands of small per-seed files,
so it ships as nine `tar.gz` archives split into `<=10MB` chunks under
`data/` - see [data/README.md](data/README.md) for the archive list, what
each one contains, and why it's chunked.

To unpack, run `reassemble.sh` from this directory:

```bash
bash reassemble.sh LFC_240_ws     # one archive
bash reassemble.sh all            # every archive
```

It concatenates the parts, verifies the result against
`data/checksums_full.sha256`, and extracts it into `nets/` - the layout the
scripts and the figure notebook read from. The seed values used to generate
the data are recorded in the models' README files.

## Figures

Once `nets/` is populated, `scripts/figures/Figure_generator_nets_manual.ipynb`
turns it into the paper's figures under `figures/` (flat, one PDF per
figure). To run it non-interactively:

```bash
cd scripts/figures && python run_figures.py
```

## Folder structure

```
.
├── README.md                     # this file
├── LICENSE                       # CC BY 4.0
├── requirements.txt              # Python packages used (see "Environment" above)
├── reassemble.sh                 # data/*.tar.gz.part* -> nets/
├── package_data.sh               # nets/ -> data/*.tar.gz.part* (the reverse)
├── submit_package_data.sh        # SLURM wrapper for package_data.sh
│
├── scripts/                      # generation + figure pipeline (see scripts/README.md)
│   ├── net_functions.py           # THE generators + graph helpers (single source)
│   ├── smoke_test.py              # asserts exact average degree, connectivity
│   ├── LFC/                       # LFC sweep, any N (generate/extract/eig-cache)
│   ├── LTM/                       # N=1000 LTM cascade realizations
│   ├── real_world/                # Netzschleuder real-world reference networks
│   ├── pooling/                   # shared lfc_data_loader.py utility
│   └── figures/                   # notebook: nets/ -> figures/ PDFs
│
├── nets/                          # network data - unpacked or generated here
│   ├── LFC/{240,1000,5000}/{ws,mhk,ke}/...
│   ├── LTM/1000/{ws,mhk,ke}/...
│   └── real_world/{scale_free,small_world}/...
│
├── figures/                       # paper figures (Figure_1.pdf ... Figure_S21.pdf)
│
├── data/                          # chunked <=10MB archives of nets/
│   ├── README.md                  # archive list, contents, how to reassemble
│   ├── checksums_full.sha256      # sha256 of each whole (reassembled) archive
│   ├── checksums_parts.sha256     # sha256 of each individual chunk
│   └── <archive>.tar.gz.part###   # e.g. LFC_240_ws.tar.gz.part000, part001, ...
│
└── logs/                          # SLURM job logs (populated at submission time)
```

## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) -
see [LICENSE](LICENSE). You're free to share and adapt this data/code for any purpose, including commercially, as long as you give appropriate credit.
