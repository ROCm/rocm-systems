#!/usr/bin/env python3
"""Analyze RCCL device kernel assembly (.s) into structured per-kernel summaries.

The RCCL device linker (see cmake/DeviceLinker.cmake, tools/rccl-device-compile)
can save per-kernel assembly when -DRCCL_DEVICE_SAVE_TEMPS=ON.  For each
specialized kernel it writes two files next to the object:

    <base>.full.s        full compiler output (kernel + devFunc + callees +
                         .amdgpu_metadata YAML + "; Kernel info:" block)
    <base>.extracted.s   the device function after entry/metadata stripping

This tool parses a .s file (default: the richer .full.s) and emits a structured
JSON summary.  The point is to keep the compiler's own accounting -- register
counts, scratch, occupancy, spill/save-restore overhead, call graph -- so that
A/B analysis after a source change never has to guess from disassembly or hand
instruction-counting.

Per file it records:
  * every function (role, mangled/demangled name, source line range)
  * per-function compiler-reported resources (NumVgprs, NumAgprs, TotalNumSgprs,
    ScratchSize, codeLenInByte) and the abstract .set symbols (num_vgpr, ...)
  * per-function instruction mix (valu/salu/vmem/smem/ds/mfma/control/other),
    loop-header count, and branch count
  * spill / save-restore overhead (folded spill stores/reloads + bytes,
    SGPR-spill-to-VGPR-lane markers)
  * the intra-TU call graph (which function calls which, via @rel32 refs)
  * the kernel entry's full "; Kernel info:" block (Occupancy, VGPRBlocks,
    SGPRBlocks, LDSByteSize, AccumOffset, WaveLimiterHint, ...)
  * the kernel's .amdgpu_metadata YAML (vgpr/agpr/sgpr_count, spill counts,
    segment sizes, max_flat_workgroup_size, kernarg size, ...)

Usage:
    # one file -> print JSON to stdout (or -o FILE)
    analyze_device_asm.py path/to/specialized_foo.full.s [-o out.json]

    # a whole directory -> write <base>.analysis.json beside each *.full.s and
    # an aggregate summary (JSON + CSV)
    analyze_device_asm.py --dir path/to/specialized \\
        [--ext full.s|extracted.s] \\
        [--aggregate summary.json] [--csv summary.csv] [--jobs N]

Demangling uses llvm-cxxfilt / c++filt if found; pass --no-demangle to skip.
"""

import argparse
import concurrent.futures
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys


# ---------------------------------------------------------------------------
# Line classifiers
# ---------------------------------------------------------------------------

_RE_TYPE_FUNC = re.compile(r'^\s*\.type\s+([^,]+),\s*@function')
_RE_SIZE = re.compile(r'^\s*\.size\s+([^,]+),')
_RE_SET_ABS = re.compile(
    r'^\s*\.set\s+\.L(.+?)\.(num_vgpr|num_agpr|numbered_sgpr|private_seg_size|'
    r'uses_flat_scratch|has_dyn_sized_stack|has_recursion|has_indirect_call)\s*,\s*(.+?)\s*$')
_RE_REL32 = re.compile(r'([A-Za-z_$][\w$]*)@rel32@lo')
_RE_LOOP_HEADER = re.compile(r';\s*=>.*Loop Header')
_RE_FOLDED = re.compile(r';\s*(\d+)-byte Folded (Spill|Reload)')
_RE_SGPR_LANE = re.compile(r'SGPR spill to VGPR lane')

# "; Key: value" or "; Key = value" comment metrics
_RE_COMMENT_KV = re.compile(r'^\s*;\s*([A-Za-z][\w:.\-> ]*?)\s*[:=]\s*(.+?)\s*$')

# Marker the extractor leaves where it stripped the kernel trampoline.
_RE_EXTRACT_TAG = re.compile(r';\s*RCCL-EXTRACT:\s*(kernel entry|end kernel entry)\s*(\S+)?')

# The short per-function comment block carries these (also present on kernels).
_REPORTED_KEYS = {
    'codeLenInByte', 'TotalNumSgprs', 'NumVgprs', 'NumAgprs', 'TotalNumVgprs',
    'ScratchSize',
}
# Extended metrics that only the kernel entry's "; Kernel info:" block carries
# (plus MemoryBound, which also appears on ordinary functions).
_KERNEL_INFO_KEYS = {
    'MemoryBound', 'FloatMode', 'IeeeMode', 'LDSByteSize',
    'SGPRBlocks', 'VGPRBlocks', 'NumSGPRsForWavesPerEU', 'NumVGPRsForWavesPerEU',
    'AccumOffset', 'Occupancy', 'WaveLimiterHint',
}

# amdgpu_metadata YAML keys worth keeping (kernel aggregate).
_METADATA_KEYS = {
    '.vgpr_count', '.agpr_count', '.sgpr_count', '.vgpr_spill_count',
    '.sgpr_spill_count', '.group_segment_fixed_size',
    '.private_segment_fixed_size', '.kernarg_segment_size',
    '.max_flat_workgroup_size', '.wavefront_size', '.uses_dynamic_stack',
    '.name', '.symbol',
}


def _to_num(s):
    s = s.strip()
    if re.fullmatch(r'-?\d+', s):
        return int(s)
    try:
        return int(s, 0)
    except (ValueError, TypeError):
        return s


def _classify_opcode(op):
    if op.startswith('v_mfma') or op.startswith('v_smfmac'):
        return 'mfma'
    if op.startswith('v_'):
        return 'valu'
    if op.startswith('ds_'):
        return 'ds'
    if op.startswith(('global_', 'flat_', 'buffer_', 'scratch_')):
        return 'vmem'
    if op.startswith('s_load') or op.startswith('s_buffer_load') or op.startswith('s_store'):
        return 'smem'
    if op.startswith(('s_branch', 's_cbranch', 's_setpc', 's_swappc', 's_call',
                      's_endpgm', 's_cbranch_execz', 's_cbranch_execnz')):
        return 'control'
    if op.startswith('s_'):
        return 'salu'
    return 'other'


def _role_for(name):
    if 'ncclDevKernel' in name:
        return 'kernel'
    if 'ncclDevFunc' in name:
        return 'devFunc'
    for token in ('runRing', 'runTree', 'runWork', 'runTreeUpDown', 'runScatter',
                  'runGather', 'runCollNet'):
        if token in name:
            return token
    return 'other'


def _is_instruction(line):
    """True if the line is an assembly instruction (not a directive/label/comment)."""
    if not line or line[0] not in ' \t':
        return False  # labels / symbols start at column 0
    s = line.strip()
    if not s or s.startswith(('.', ';', '#')):
        return False
    if s.endswith(':'):
        return False
    return True


# ---------------------------------------------------------------------------
# Metadata (amdgpu_metadata YAML) parsing
# ---------------------------------------------------------------------------

def _parse_amdgpu_metadata(lines):
    start = end = None
    for i, line in enumerate(lines):
        st = line.strip()
        if st == '.amdgpu_metadata':
            start = i
        elif st == '.end_amdgpu_metadata':
            end = i
            break
    if start is None or end is None:
        return {}
    out = {}
    for line in lines[start + 1:end]:
        st = line.strip().lstrip('- ').strip()
        m = re.match(r'(\.[\w]+):\s*(.+?)\s*$', st)
        if m and m.group(1) in _METADATA_KEYS:
            key = m.group(1).lstrip('.')
            if key not in out:  # keep first (kernel-level) occurrence
                out[key] = _to_num(m.group(2))
    return out


# ---------------------------------------------------------------------------
# Core parse
# ---------------------------------------------------------------------------

def parse_asm(path):
    with open(path, errors='replace') as f:
        lines = f.readlines()

    functions = []          # ordered list of dicts
    by_name = {}            # mangled -> dict
    cur = None              # function currently accumulating instructions
    attach = None           # function to attach trailing comment metrics to
    # When the extractor strips the kernel trampoline (RCCL_DEVICE_SAVE_TEMPS
    # extracted.s), its resource comments survive inside an RCCL-EXTRACT tag.
    # Route those into a separate bucket so they don't misattach to the devFunc.
    removed_kernel = {'reported': {}, 'kernel_info': {}, 'symbol': None}
    in_removed_kernel = False

    def new_func(name, line_no):
        d = {
            'name_mangled': name,
            'role': _role_for(name),
            'line_start': line_no,
            'line_end': None,
            'abstract': {},
            'reported': {},
            'kernel_info': {},
            'instr': {'total': 0, 'valu': 0, 'salu': 0, 'vmem': 0, 'smem': 0,
                      'ds': 0, 'mfma': 0, 'control': 0, 'other': 0, 'branch': 0},
            'loops': 0,
            'spills': {'spill_store': 0, 'spill_reload': 0,
                       'spill_store_bytes': 0, 'spill_reload_bytes': 0,
                       'sgpr_spill_to_vgpr_lane': 0},
            'calls': [],
            '_calls_set': set(),
        }
        functions.append(d)
        by_name[name] = d
        return d

    for i, line in enumerate(lines, 1):
        mt = _RE_TYPE_FUNC.match(line)
        if mt:
            name = mt.group(1).strip()
            cur = new_func(name, i)
            attach = cur
            continue

        # Abstract .set symbols (num_vgpr, private_seg_size, flags, ...)
        ms = _RE_SET_ABS.match(line)
        if ms:
            tgt = by_name.get(ms.group(1).strip())
            if tgt is not None:
                tgt['abstract'][ms.group(2)] = _to_num(ms.group(3))
            continue

        mz = _RE_SIZE.match(line)
        if mz and cur is not None and mz.group(1).strip() == cur['name_mangled']:
            cur['line_end'] = i
            # instructions stop; trailing comment block will be attached to `attach`
            cur = None
            continue

        # Comment metric lines.
        if line.lstrip().startswith(';'):
            mtag = _RE_EXTRACT_TAG.search(line)
            if mtag:
                if mtag.group(1) == 'kernel entry':
                    in_removed_kernel = True
                    removed_kernel['symbol'] = mtag.group(2)
                else:  # "end kernel entry"
                    in_removed_kernel = False
                continue
            # Route to the stripped-kernel bucket, or to the current function.
            target = removed_kernel if in_removed_kernel else attach
            if target is None:
                continue
            mk = _RE_COMMENT_KV.match(line)
            if mk:
                key = mk.group(1).strip()
                val = mk.group(2)
                if key in _REPORTED_KEYS:
                    target['reported'][key] = _to_num(val)
                elif key in _KERNEL_INFO_KEYS:
                    if key == 'LDSByteSize':
                        lm = re.match(r'\s*(\d+)', val)
                        target['kernel_info'][key] = int(lm.group(1)) if lm else _to_num(val)
                    else:
                        target['kernel_info'][key] = _to_num(val)
            if in_removed_kernel:
                continue
            if _RE_LOOP_HEADER.search(line):
                attach['loops'] += 1
            if _RE_SGPR_LANE.search(line):
                attach['spills']['sgpr_spill_to_vgpr_lane'] += 1
            mf = _RE_FOLDED.search(line)
            if mf:
                nbytes = int(mf.group(1))
                if mf.group(2) == 'Spill':
                    attach['spills']['spill_store'] += 1
                    attach['spills']['spill_store_bytes'] += nbytes
                else:
                    attach['spills']['spill_reload'] += 1
                    attach['spills']['spill_reload_bytes'] += nbytes
            continue

        # Instruction accounting (only inside a function body).
        if cur is not None and _is_instruction(line):
            op = line.strip().split(None, 1)[0]
            cat = _classify_opcode(op)
            cur['instr']['total'] += 1
            cur['instr'][cat] += 1
            if cat == 'control' and op.startswith(('s_branch', 's_cbranch')):
                cur['instr']['branch'] += 1
            for cm in _RE_REL32.finditer(line):
                callee = cm.group(1)
                if callee not in cur['_calls_set']:
                    cur['_calls_set'].add(callee)
                    cur['calls'].append(callee)

    metadata = _parse_amdgpu_metadata(lines)

    # The kernel entry (ncclDevKernel_..._Specialized) is a tiny trampoline that
    # tail-calls the devFunc; its per-function register/scratch numbers are the
    # aggregate max() over callees, NOT its own usage.  Pull its whole-kernel
    # info (occupancy/blocks/aggregate regs) aside and exclude it from the
    # analyzed function list so it does not double-count against the devFunc.
    kernel_funcs = [d for d in functions if d['role'] == 'kernel']
    real_funcs = [d for d in functions if d['role'] != 'kernel']
    if not real_funcs:                 # degenerate TU: keep whatever exists
        real_funcs, kernel_funcs = functions, []

    kernel_info = {}
    kernel_aggregate = {}
    kernel_symbol = None
    if kernel_funcs:
        k = kernel_funcs[0]
        kernel_info = k['kernel_info']
        kernel_aggregate = dict(k['reported'])
        kernel_symbol = k['name_mangled']
    elif removed_kernel['reported'] or removed_kernel['kernel_info']:
        # Kernel was stripped by the extractor; recover its info from the tag.
        kernel_info = removed_kernel['kernel_info']
        kernel_aggregate = removed_kernel['reported']
        kernel_symbol = removed_kernel['symbol']

    # Call graph edges limited to the real (non-kernel) functions in this TU.
    defined = {d['name_mangled'] for d in real_funcs}
    callgraph = {}
    for d in real_funcs:
        edges = [c for c in d['calls'] if c in defined]
        if edges:
            callgraph[d['name_mangled']] = edges
    for d in functions:
        d.pop('_calls_set', None)

    base = os.path.basename(path)
    short = re.sub(r'^specialized_', '', base)
    short = re.sub(r'\.(full|extracted)\.s$', '', short)
    short = re.sub(r'\.s$', '', short)

    abspath = os.path.abspath(path)
    am = re.search(r'rccl_device_(gfx[0-9a-z]+)', abspath) or \
        re.search(r'[/-](gfx[0-9a-z]+)/', abspath)
    arch = am.group(1) if am else None

    return {
        'file': base,
        'path': abspath,
        'arch': arch,
        'kernel': short,
        'kernel_symbol': kernel_symbol,
        'num_functions': len(real_funcs),
        'functions': real_funcs,
        'callgraph': callgraph,
        'kernel_info': kernel_info,
        'kernel_aggregate': kernel_aggregate,
        'amdgpu_metadata': metadata,
        'totals': _file_totals(real_funcs, kernel_aggregate, kernel_info, metadata),
    }


def _file_totals(functions, kernel_aggregate, kernel_info, metadata):
    t = {
        'instr_total': sum(f['instr']['total'] for f in functions),
        'spill_store': sum(f['spills']['spill_store'] for f in functions),
        'spill_reload': sum(f['spills']['spill_reload'] for f in functions),
        'spill_store_bytes': sum(f['spills']['spill_store_bytes'] for f in functions),
        'spill_reload_bytes': sum(f['spills']['spill_reload_bytes'] for f in functions),
        'sgpr_spill_to_vgpr_lane': sum(f['spills']['sgpr_spill_to_vgpr_lane'] for f in functions),
    }
    # Whole-kernel aggregate (from the trampoline's max() / kernel-info block).
    t['kernel_NumVgprs'] = kernel_aggregate.get('NumVgprs')
    t['kernel_NumAgprs'] = kernel_aggregate.get('NumAgprs')
    t['kernel_TotalNumVgprs'] = kernel_aggregate.get('TotalNumVgprs')
    t['kernel_TotalNumSgprs'] = kernel_aggregate.get('TotalNumSgprs')
    t['kernel_ScratchSize'] = kernel_aggregate.get('ScratchSize')
    t['kernel_Occupancy'] = kernel_info.get('Occupancy')
    t['kernel_VGPRBlocks'] = kernel_info.get('VGPRBlocks')
    t['kernel_LDSByteSize'] = kernel_info.get('LDSByteSize')
    t['meta_vgpr_count'] = metadata.get('vgpr_count')
    t['meta_agpr_count'] = metadata.get('agpr_count')
    t['meta_sgpr_count'] = metadata.get('sgpr_count')
    t['meta_vgpr_spill_count'] = metadata.get('vgpr_spill_count')
    t['meta_sgpr_spill_count'] = metadata.get('sgpr_spill_count')
    t['meta_private_segment_fixed_size'] = metadata.get('private_segment_fixed_size')
    t['meta_group_segment_fixed_size'] = metadata.get('group_segment_fixed_size')
    return t


# ---------------------------------------------------------------------------
# Demangling (batched, best-effort)
# ---------------------------------------------------------------------------

def _find_cxxfilt():
    for name in ('llvm-cxxfilt', 'c++filt'):
        p = shutil.which(name)
        if p:
            return p
    # srock LLVM, per repo toolchain rule
    for cand in ('/work/lmeadows/rocm/srock/llvm/bin/llvm-cxxfilt',):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def demangle_summary(summary, cxxfilt):
    names = [f['name_mangled'] for f in summary['functions']]
    if not names or not cxxfilt:
        return
    try:
        out = subprocess.run([cxxfilt], input='\n'.join(names) + '\n',
                             capture_output=True, text=True, check=True).stdout
        demangled = out.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return
    for f, dm in zip(summary['functions'], demangled):
        f['name_demangled'] = dm


# ---------------------------------------------------------------------------
# CSV aggregate
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    'arch', 'kernel', 'num_functions',
    'kernel_NumVgprs', 'kernel_NumAgprs', 'kernel_TotalNumVgprs',
    'kernel_TotalNumSgprs', 'kernel_ScratchSize', 'kernel_Occupancy',
    'kernel_VGPRBlocks', 'kernel_LDSByteSize',
    'meta_vgpr_count', 'meta_agpr_count', 'meta_sgpr_count',
    'meta_vgpr_spill_count', 'meta_sgpr_spill_count',
    'meta_private_segment_fixed_size', 'meta_group_segment_fixed_size',
    'instr_total', 'spill_store', 'spill_reload',
    'spill_store_bytes', 'spill_reload_bytes', 'sgpr_spill_to_vgpr_lane',
]


def write_csv(summaries, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(_CSV_COLUMNS)
        for s in sorted(summaries, key=lambda x: (x.get('arch') or '', x['kernel'])):
            t = s['totals']
            row = [s.get('arch'), s['kernel'], s['num_functions']]
            row += [t.get(c) for c in _CSV_COLUMNS[3:]]
            w.writerow(row)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _analyze_one(path, demangle, cxxfilt):
    s = parse_asm(path)
    if demangle:
        demangle_summary(s, cxxfilt)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='*', help='.s file(s) to analyze')
    ap.add_argument('--dir', help='directory of specialized kernels to scan')
    ap.add_argument('--root', help='recursively scan a build tree for saved assembly')
    ap.add_argument('--ext', default='full.s', choices=['full.s', 'extracted.s'],
                    help='which saved assembly to parse under --dir (default: full.s)')
    ap.add_argument('-o', '--output', help='output JSON for a single input file')
    ap.add_argument('--aggregate', help='write combined JSON of all summaries here')
    ap.add_argument('--csv', help='write aggregate CSV table here')
    ap.add_argument('--no-per-file', action='store_true',
                    help='under --dir, do not write <base>.analysis.json beside each source')
    ap.add_argument('--no-demangle', action='store_true', help='skip demangling')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    cxxfilt = None if args.no_demangle else _find_cxxfilt()
    demangle = not args.no_demangle

    files = list(args.inputs)
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, f'*.{args.ext}')))
    if args.root:
        files += sorted(glob.glob(os.path.join(args.root, '**', f'*.{args.ext}'),
                                  recursive=True))
    files = sorted(set(files))
    if not files:
        ap.error('no input: pass .s file(s), --dir, or --root')

    # Single file, no aggregate: print (or -o).
    if len(files) == 1 and not args.dir and not args.root \
            and not args.aggregate and not args.csv:
        s = _analyze_one(files[0], demangle, cxxfilt)
        text = json.dumps(s, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(text + '\n')
            print(f'wrote {args.output}', file=sys.stderr)
        else:
            print(text)
        return

    summaries = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_analyze_one, p, demangle, cxxfilt): p for p in files}
        for fut in concurrent.futures.as_completed(futs):
            p = futs[fut]
            try:
                summaries.append(fut.result())
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f'ERROR analyzing {p}: {e}', file=sys.stderr)

    if not args.no_per_file:
        for s in summaries:
            base = re.sub(r'\.(full|extracted)\.s$', '', s['path'])
            base = re.sub(r'\.s$', '', base)
            with open(base + '.analysis.json', 'w') as f:
                json.dump(s, f, indent=2)

    if args.aggregate:
        with open(args.aggregate, 'w') as f:
            json.dump(sorted(summaries, key=lambda x: x['kernel']), f, indent=2)
        print(f'wrote {args.aggregate} ({len(summaries)} kernels)', file=sys.stderr)
    if args.csv:
        write_csv(summaries, args.csv)
        print(f'wrote {args.csv} ({len(summaries)} kernels)', file=sys.stderr)
    if not args.aggregate and not args.csv and args.no_per_file:
        print(json.dumps(sorted(summaries, key=lambda x: x['kernel']), indent=2))

    print(f'analyzed {len(summaries)} kernels', file=sys.stderr)


if __name__ == '__main__':
    main()
