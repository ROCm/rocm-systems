#!/usr/bin/env python3
"""reshard_batch_bench_user_window logs -> markdown (median over launches of seq/con GB/s and speedup)."""
import glob, os, re, statistics, sys
from collections import defaultdict
root = sys.argv[1]
d = defaultdict(list); st = defaultdict(set)
for f in sorted(glob.glob(os.path.join(root, '*_c*_r*.log'))):
    m = re.match(r'(.+)_(pack|direct|pipe)_c(\d+)_r(\d+|v)\.log', os.path.basename(f))
    lab, cp, nc, rnd = m.groups()
    t = open(f, errors='ignore').read()
    ex = re.search(r'=== EXIT (\d+)', t)
    rc = int(ex.group(1)) if ex else None
    # Success markers precede teardown: a launch counts only with a recorded zero exit.
    ok = rc == 0 and 'Benchmark completed successfully' in t and 'VALIDATION FAILED' not in t
    res = ('PASS' if rnd in ('0', 'v') else 'OK') if ok else ('HANG' if rc == 124 else f'FAIL_rc{rc}')
    st[(lab, cp, nc)].add(res)
    if not ok or rnd == 'v':   # round v: validation-only tensors, not in the tables
        continue
    for line in t.splitlines():
        p = line.split()
        if len(p) == 8 and p[0].startswith('['):
            d[(lab, p[1], p[0].replace(',', 'x'), cp, nc)].append((float(p[6]), float(p[7]), float(p[4])))
labs = sorted({k[0] for k in st})   # from statuses, so scales where every launch failed still show
for lab in labs:
    print(f'\n#### {lab}: 4 tensors, sequential vs concurrent, GiB/s (median of 3 launches); speedup = seq ms / con ms\n')
    print('| pattern | tensor (int8) | copy | ' + ' | '.join(f'C={c} seq / con / x' for c in ('1', '2', '4')) + ' | validate |')
    print('|---|---|---|---|---|---|---|')
    for pat in ('same-dim', 'cross-dim'):
        for ten in ('[1024x1024]', '[4096x4096]', '[16384x16384]'):
            for cp in ('pack', 'direct', 'pipe'):
                cells = []
                for nc in ('1', '2', '4'):
                    v = d.get((lab, pat, ten, cp, nc))
                    cells.append(' / '.join(f'{statistics.median(x):.2f}' for x in zip(*v)) if v else '-')
                val = ' '.join(f'C{nc}:{"/".join(sorted(st[(lab, cp, nc)]))}' for nc in ('1', '2', '4') if (lab, cp, nc) in st)
                print(f'| {pat} | {ten} | {cp.upper()} | ' + ' | '.join(cells) + f' | {val} |')
