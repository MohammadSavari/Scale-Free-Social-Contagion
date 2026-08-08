'''
Executes Figure_generator_nets_manual.ipynb cell by cell against ../../nets,
skipping any figure whose input data is absent instead of aborting the whole
run.

Each code cell is exec'd into ONE shared namespace (so cells that only define
helpers still feed later cells) but wrapped individually in try/except. A cell
that raises - typically FileNotFoundError for data not present - is reported
as SKIP and execution continues. That is what makes this usable against a
partial nets/ tree: you get every figure the data supports plus an explicit
list of what was missing, rather than one traceback and nothing else.

Usage (cwd must be this directory, so ../../nets and ../../figures resolve):
    python run_figures.py                 # all cells
    python run_figures.py --only 6 7 9    # just these cell indices
    python run_figures.py --skip 13 24    # force-skip specific cells
'''

import argparse
import io
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

NB = Path(__file__).resolve().parent / 'Figure_generator_nets_manual.ipynb'
FIG_DIR = Path(__file__).resolve().parents[2] / 'figures'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--notebook', default=str(NB))
    ap.add_argument('--only', type=int, nargs='+', default=None)
    ap.add_argument('--skip', type=int, nargs='+', default=[])
    ap.add_argument('--verbose', action='store_true',
                    help='echo each cell\'s stdout instead of capturing it')
    args = ap.parse_args()

    nb = json.loads(Path(args.notebook).read_text())
    cells = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells'])
             if c['cell_type'] == 'code' and ''.join(c['source']).strip()]

    before = set(FIG_DIR.glob('*.pdf')) if FIG_DIR.exists() else set()
    ns = {'__name__': '__main__'}
    ok = skipped = 0
    results = []

    for i, src in cells:
        if args.only is not None and i not in args.only:
            continue
        if i in args.skip:
            print(f'[cell {i:2d}] SKIP (requested)', flush=True)
            results.append((i, 'SKIP', 'requested')); skipped += 1
            continue

        label = next((l.strip() for l in src.split('\n')[:6]
                      if l.strip() and not l.strip().startswith(("'''", '"""', '#'))), '')[:60]
        t0 = time.time()
        buf = io.StringIO()
        try:
            if args.verbose:
                exec(compile(src, f'<cell {i}>', 'exec'), ns)
            else:
                with redirect_stdout(buf):
                    exec(compile(src, f'<cell {i}>', 'exec'), ns)
            dt = time.time() - t0
            print(f'[cell {i:2d}] OK   {dt:7.1f}s  {label}', flush=True)
            out = buf.getvalue().strip()
            for line in out.split('\n')[-4:]:
                if line.strip():
                    print(f'          | {line[:120]}', flush=True)
            results.append((i, 'OK', label)); ok += 1
        except Exception as exc:
            dt = time.time() - t0
            first = traceback.format_exception_only(type(exc), exc)[-1].strip()
            print(f'[cell {i:2d}] SKIP {dt:7.1f}s  {label}', flush=True)
            print(f'          -> {first[:200]}', flush=True)
            results.append((i, 'SKIP', first[:200])); skipped += 1

    print(f'\n=== {ok} cells ok, {skipped} skipped ===')
    after = set(FIG_DIR.glob('*.pdf')) if FIG_DIR.exists() else set()
    new = sorted(p.name for p in after - before)
    allf = sorted(p.name for p in after)
    print(f'figures written this run ({len(new)}): {new}')
    print(f'figures present total  ({len(allf)}): {allf}')
    print('\nskipped cells:')
    for i, st, msg in results:
        if st == 'SKIP':
            print(f'  cell {i:2d}: {msg}')


if __name__ == '__main__':
    main()
