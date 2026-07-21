'''
Thin wrapper around ../LFC/extract_lfc_csv.py that defaults --root to
../../nets/real_world, so the real-world '<network>_props.csv' and
'<network>_{top,bot}_5_corr_gains_degree.csv' files (read by
generate_real_world_figure.py) can be (re)generated from here without
having to remember the --root flag.

Reuses extract_lfc_csv.py as-is (no logic duplicated) - it already works
against any directory tree of .gt files, grouping by parent directory, which
is exactly the real_world/<category>/<network>/<network>.gt layout
generate_real_world.py writes.

Example:
    python extract_real_world_csv.py
    python extract_real_world_csv.py --selectors top:5 bot:5
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'LFC'))
import extract_lfc_csv as csv_gen

if __name__ == '__main__':
    if '--root' not in sys.argv:
        sys.argv += ['--root', str(Path(__file__).resolve().parent.parent.parent / 'nets' / 'real_world')]
    csv_gen.main()
