#!/usr/bin/env python3
"""Aggregate reshard_bench matrix CSVs -> markdown tables (median of launches, min/max spread, %SOL)."""
import csv, glob, os, statistics, sys
from collections import defaultdict

root = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(root, '*', 'results.csv'))):
    if os.path.basename(os.path.dirname(f)) == 'batch':
        continue
    rows += list(csv.DictReader(open(f)))
# <label>rerun rows replace the same points of <label> (interference window / infra flakes).
rerun = {(r['label'][:-5], r['shape'], r['size'], r['config']) for r in rows if r['label'].endswith('rerun')}
rows = [r for r in rows if (r['label'], r['shape'], r['size'], r['config']) not in rerun]
for r in rows:
    r['dir'] = r['label']
    if r['label'].endswith('rerun'):
        r['label'] = r['label'][:-5]

SIZES = ['64K', '256K', '1M', '4M', '16M', '64M', '256M', '1G', '4G', '16G']
CFGS = ['D_PACK', 'D_DIRECT', 'D_PIPE', 'R_PACK']
LINK = 76.8   # GB/s per xGMI link per direction (MI355X: 7 links x 153.6 GB/s bidir)
NIC = 50.0    # GB/s per GPU, 1x Thor2 400G per GPU

def sol(label, shape, src, dst, nnodes):
    """Per-source-rank SOL in GB/s (1e9). Intra-node: links to distinct peers; inter-node: NIC."""
    src, dst = int(src), int(dst)
    peers = 1 if shape == 'same00' else (2 if shape == 'mesh2' else dst)
    peers = min(peers, dst)
    if nnodes == 1:
        return peers * LINK
    return NIC * min(1.0, dst / src)

NPS = {}
def nn(label):
    # nodes = ceil(ranks / 8); ranks recorded per label in the CSV
    return -(-int(NPS.get(label, 8)) // 8)

import re
def log_text(r):
    try:
        return open(os.path.join(root, r['dir'], r['log']), errors='ignore').read()
    except OSError:
        return ''

def src_bw(r):
    t = log_text(r)
    m = re.search(r'--- Sources only.*?\nTime per iteration \(ms\):\s+Min=[0-9.]+\s+Max=([0-9.]+)\nBandwidth \(GB/s\):\s+Min=[0-9.]+\s+Max=[0-9.]+\s+Avg=([0-9.]+)', t, re.S)
    return (float(m.group(2)), float(m.group(1))) if m else None

for r in rows:
    NPS[r['label']] = r['np']
    # PASS/OK markers precede teardown: a recorded nonzero exit turns the launch into a failure.
    ex = re.search(r'=== EXIT (\d+)', log_text(r))
    if r['result'] in ('OK', 'PASS') and ex and ex.group(1) != '0':
        r['result'] = 'FAIL_rc' + ex.group(1)

g = defaultdict(list)
status = defaultdict(set)
meta = {}
for r in rows:
    k = (r['label'], r['shape'], r['size'], r['config'])
    meta[k] = (r['src'], r['dst'], r['np'])
    status[k].add(r['result'])
    if r['result'] in ('OK', 'PASS') and r['round'] != 'v':   # round v: validation-only shape
        sb = src_bw(r)
        if sb is None and not r['bw_rank_avg']:
            continue
        bw, tm = sb if sb else (float(r['bw_rank_avg']), float(r['time_ms_max'] or 0))
        g[k].append((bw, float(r['total_gbs'] or 0), tm))

def cell(k):
    st = status.get(k)
    if not st:
        return ''
    bad = [s for s in st if s not in ('OK', 'PASS')]
    v = g.get(k)
    if not v:
        return '/'.join(sorted(bad)) or '?'
    bws = [x[0] for x in v]
    med = statistics.median(bws)
    s = f'{med:.2f}' if med < 10 else f'{med:.1f}'
    if len(bws) > 1 and med > 0:
        spread = (max(bws) - min(bws)) / med * 100
        if spread >= 5:
            s += f' ±{spread:.0f}%'
    if 'PASS' not in st:
        s += ' (no val)'
    if bad:
        s += ' ' + '/'.join(sorted(bad))
    return s

labels = sorted({r['label'] for r in rows}, key=lambda l: (nn(l), l))
out = []
for lab in labels:
    ORDER = ['same00', 'cross01', 'cross10', 'mesh2', 'rcross01', 'rmesh']
    shapes = sorted({k[1] for k in meta if k[0] == lab}, key=lambda x: (ORDER.index(x) if x in ORDER else 99, x))
    for shp in shapes:
        ks = [k for k in meta if k[0] == lab and k[1] == shp]
        src, dst, np_ = meta[ks[0]]
        cfgs = [c for c in CFGS if any(k[3] == c for k in ks)]
        sizes = [z for z in SIZES if any(k[2] == z for k in ks)]
        S = sol(lab, shp, src, dst, nn(lab)) * 1e9 / 2**30
        out.append(f'\n#### {lab} ({np_} ranks, {src}->{dst}), {shp}; SOL ≈ {S:.0f} GiB/s per src rank\n')
        out.append('| size | ' + ' | '.join(cfgs) + ' | D_PACK %SOL | D_PACK lat ms |')
        out.append('|---' * (len(cfgs) + 3) + '|')
        for z in sizes:
            cs = [cell((lab, shp, z, c)) for c in cfgs]
            v = g.get((lab, shp, z, 'D_PACK'))
            pct = f'{statistics.median([x[0] for x in v]) / S * 100:.0f}%' if v else ''
            lat = f'{statistics.median([x[2] for x in v]):.3f}' if v else ''
            out.append(f'| {z} | ' + ' | '.join(cs) + f' | {pct} | {lat} |')
print('\n'.join(out))
