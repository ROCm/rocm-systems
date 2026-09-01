#!/bin/bash
# Does each variant actually emit the instruction width and temporal hint it
# claims? Every timing number in this investigation is meaningless if a variant
# silently compiled to something else - and that is a live risk, since the whole
# question is about a hint that a compiler is free to drop.
#
# Expectations are not written here. They are read from the binary via
# `isolated_copy --print-isa-expectations`, which prints the table in
# src/common/variants.h, so there is one definition and it sits next to the code
# it describes. Adding a variant needs no change to this script.
#
# Exits non-zero on any mismatch.
set -uo pipefail
cd "$(dirname "$0")"

HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1250}
CXXFILT=${CXXFILT:-/opt/rocm/llvm/bin/llvm-cxxfilt}
mkdir -p isa build

echo "=== emitting device assembly ==="
$HIPCC --offload-arch=$ARCH -x hip -O3 -std=c++17 --offload-device-only -S \
       src/experiments/isolated_copy.cc -o isa/isolated_copy.s || exit 1
wc -l isa/isolated_copy.s

[ -x build/isolated_copy ] || ./build.sh isolated_copy >/dev/null || exit 1
./build/isolated_copy --print-isa-expectations > isa/expectations.tsv 2>/dev/null
echo "=== expectations declared by variants.h ==="
sed -n '2,$p' isa/expectations.tsv | awk -F'\t' '{printf "  %-20s elem=%2sB mode=%s  %s %s\n", $2, $3, $4, $5, $6}'

# Kernel labels look like  _Z14blitCopyBuffer...:   ; @_Z14blitCopyBuffer...
# so the label regex has to tolerate a trailing comment.
awk '
  /^_Z[A-Za-z0-9_]*:/ { cur = $0; sub(/:.*$/, "", cur); next }
  /global_(load|store)/ {
    line = $0
    gsub(/^[ \t]+/, "", line)
    sub(/[ \t]*;.*$/, "", line)
    print cur "|" line
  }
' isa/isolated_copy.s > isa/raw_ops.txt

python3 - <<'PY'
import collections, os, re, subprocess, sys

CXXFILT = os.environ.get('CXXFILT', '/opt/rocm/llvm/bin/llvm-cxxfilt')

# Classify from the demangled signature rather than the mangling, which changes
# whenever the code moves namespace and gives no useful error when it does:
#   void bench::blitCopyBuffer<unsigned long vector[2], (bench::AccessMode)2>(...)
ELEM_BYTES = [('unsigned long vector[2]', 16), ('unsigned long', 8), ('unsigned int', 4)]
DEM_RE = re.compile(r'blitCopyBuffer<([^,]+), \(\w+(?:::\w+)*\)(\d)>')

expectations = []
with open('isa/expectations.tsv') as f:
    for line in f:
        if not line.startswith('ISA\t'):
            continue
        _, name, elem, mode, opcode, hint = line.rstrip('\n').split('\t')
        expectations.append((name, int(elem), int(mode), opcode,
                             None if hint == '-' else hint))

rows = []
with open('isa/raw_ops.txt') as f:
    for line in f:
        if '|' not in line:
            continue
        sym, op = line.rstrip('\n').split('|', 1)
        rows.append((sym, op.strip()))

syms = sorted({s for s, _ in rows})
demangled = {}
if syms:
    out = subprocess.run([CXXFILT], input='\n'.join(syms), capture_output=True,
                         text=True).stdout.splitlines()
    demangled = dict(zip(syms, out))

def classify(sym):
    m = DEM_RE.search(demangled.get(sym, ''))
    if not m:
        return None
    elem_text = m.group(1).strip()
    for text, nbytes in ELEM_BYTES:
        if elem_text == text:
            return (nbytes, int(m.group(2)))
    return None

# Group the emitted memory ops by the (element size, mode) of their kernel.
groups = collections.defaultdict(collections.Counter)
unmatched = set()
for sym, op in rows:
    key = classify(sym)
    if key is None:
        if 'blitCopyBuffer' in demangled.get(sym, ''):
            unmatched.add(demangled[sym])
        continue
    opcode = op.split()[0]
    th = re.search(r'th:\S+', op)
    groups[key][opcode + (' ' + th.group(0) if th else '')] += 1

if unmatched:
    print('  WARNING: could not classify these blit kernels:')
    for u in sorted(unmatched):
        print(f'    {u}')

print()
print('=== emitted memory ops per instantiation ===')
for key in sorted(groups):
    elem, mode = key
    users = [n for n, e, md, _, _ in expectations if e == elem and md == mode]
    print(f"  ElemT={elem:2d}B mode={mode}  used by: {', '.join(users) or '(none)'}")
    for form, n in sorted(groups[key].items()):
        print(f"      {form:<44} x{n}")

print()
print('=== expectation check ===')
failures = 0
for name, elem, mode, opcode, hint in expectations:
    forms = set(groups.get((elem, mode), {}))
    want = opcode + (' ' + hint if hint else '')
    ok = want in forms
    if hint is None:
        # A plain expectation additionally requires that this opcode carries no
        # hint anywhere in the kernel.
        ok = ok and not any(f.startswith(opcode + ' th:') for f in forms)
    if not ok:
        failures += 1
    print(f"  [{'OK ' if ok else 'BAD'}] {name:<20} expects {want}")
    if not ok:
        print(f"        found instead: {sorted(forms)}")

print()
if failures:
    print(f'ISA_EXPECTATIONS: {failures} MISMATCH(ES) - timing results are not trustworthy')
    sys.exit(1)
print('ISA_EXPECTATIONS: ALL MATCH')
PY
rc=$?

echo
echo "=== only nt-both-128 may carry an NT load hint ==="
n=$(grep -c 'global_load_b128 .*th:TH_LOAD_NT' isa/isolated_copy.s || true)
echo "  global_load_b128 ... th:TH_LOAD_NT occurrences: $n"

echo
echo "=== support kernels must be plain (no temporal hints) ==="
bad=0
while IFS='|' read -r sym op; do
  hint=$(echo "$op" | grep -o 'th:[A-Z_]*' || true)
  name=$($CXXFILT "$sym" | sed 's/(.*//')
  echo "  ${name} -> $(echo "$op" | awk '{print $1}') ${hint:-(no hint)}"
  [ -n "$hint" ] && bad=1
done < <(awk '
  /^_Z[A-Za-z0-9_]*:/ { cur = $0; sub(/:.*$/, "", cur); next }
  /global_(load|store)/ {
    if (cur ~ /sweepReadKernel|chaseKernel/) {
      line = $0; gsub(/^[ \t]+/, "", line); sub(/[ \t]*;.*$/, "", line)
      print cur "|" line
    }
  }' isa/isolated_copy.s | sort -u)
if [ "$bad" != 0 ]; then
  echo "  BAD: a support kernel carries a temporal hint, so it is not a neutral probe"
  rc=1
fi

echo
[ "$rc" = 0 ] && echo "ISA_OK" || echo "ISA_FAILED"
exit $rc
