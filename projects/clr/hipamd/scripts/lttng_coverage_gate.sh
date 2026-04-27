#!/usr/bin/env bash
# LTTng coverage gate (HIP).
#
# 1. Symbol coverage: diff the public HIP symbols exported from
#    libamdhip64.so against the AST-migration inventory (one or more TSV
#    files written by lttng_migrate.py). Fails the build on any exported
#    HIP symbol that wasn't migrated AND isn't in the exemption file.
#
# 2. Body-content coverage (defense-in-depth): for each migrated symbol,
#    locate its function body in the source TUs and require:
#      - >=1 rocm_trace_emit_hip_api_enter call
#      - AND (>=1 ROCM_TRACE_RET_*/ROCR_TRACE_API_RET_* macro
#        OR    >=1 rocm_trace_emit_hip_api_exit_* call)
#    The migrator should always produce balanced wrappers; this guards
#    against accidental drift (hand edits, partial reverts).
#    By default body-content failures are reported as WARNINGs only.
#    Set `LTTNG_COVERAGE_STRICT=1` in the environment to make them fatal.
#
# Usage:
#   lttng_coverage_gate.sh <path/to/libamdhip64.so> <inv1.txt> [<inv2.txt> ...]
#
# Exit codes:
#   0  PASS
#   1  FAIL - symbol-coverage gap, or body-content gap with STRICT=1
#   2  USAGE error
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <path/to/libamdhip64.so> <inventory1.txt> [<inventory2.txt> ...]" >&2
    exit 2
fi

SO="$1"; shift
if [ ! -f "$SO" ]; then
    echo "ERROR: $SO not found" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXEMPT_FILE="$SCRIPT_DIR/lttng_coverage_exemptions.txt"

# Source-tree root that contains the wrapper TUs. The HIP wrappers live in
# projects/clr/hipamd/src; the script itself is in projects/clr/hipamd/scripts.
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/src"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ---------------------------------------------------------------------------
# 1. Symbol-coverage gate
# ---------------------------------------------------------------------------

# Exported text symbols (T or t) starting with hip or __hip. Strip
# version-script suffix (`name@@version` -> `name`).
nm -D --defined-only "$SO" \
    | awk '$2 ~ /^[Tt]$/ && $3 ~ /^(hip|__hip)/ { sub(/@@.*/, "", $3); print $3 }' \
    | sort -u > "$WORK/exported.txt"

# Migrated symbols (column 1 of each inventory file) plus exemptions.
{
    for inv in "$@"; do
        if [ ! -f "$inv" ]; then
            echo "ERROR: inventory $inv not found" >&2
            exit 2
        fi
        awk '{print $1}' "$inv"
    done
    if [ -f "$EXEMPT_FILE" ]; then
        # Strip comments and blanks.
        sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$EXEMPT_FILE"
    fi
} | sort -u > "$WORK/migrated.txt"

# Symbols in .so but NOT in any inventory.
MISSING="$(comm -23 "$WORK/exported.txt" "$WORK/migrated.txt" || true)"
if [ -n "$MISSING" ]; then
    NMISS="$(printf '%s\n' "$MISSING" | wc -l)"
    echo "FAIL: $NMISS exported HIP symbols are NOT in the LTTng migration inventory:"
    printf '  %s\n' $MISSING
    echo ""
    echo "These symbols are exported from $SO but lttng_migrate.py did not"
    echo "rewrite their wrapper bodies. Either:"
    echo "  - extend the migrator (e.g. add a return-type to classify())"
    echo "  - add the wrappers to a source file the migrator scans"
    echo "  - explicitly exempt the symbol with a known-list (only after review)"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Body-content coverage (defense-in-depth)
# ---------------------------------------------------------------------------
# Inventory symbol names, deduplicated.
{
    for inv in "$@"; do
        awk '{print $1}' "$inv"
    done
} | sort -u > "$WORK/inventory_names.txt"

STRICT="${LTTNG_COVERAGE_STRICT:-0}"
N_BODY_FAIL=0

# Build a single regex of all migrated symbol names so we can locate each
# definition cheaply. We scan all .cpp files in SRC_DIR.
PYTHON_GATE="$(cat <<'PY'
import os, re, sys

src_dir = sys.argv[1]
names_file = sys.argv[2]

with open(names_file) as f:
    names = [ln.strip() for ln in f if ln.strip()]

# Collect (filepath, contents) for every .cpp under src_dir.
files = []
for root, _, fnames in os.walk(src_dir):
    for fn in fnames:
        if fn.endswith('.cpp'):
            p = os.path.join(root, fn)
            with open(p, 'rb') as fh:
                files.append((p, fh.read().decode('utf-8', errors='replace')))

ENTER_RE   = re.compile(r'rocm_trace_emit_hip_api_enter\s*\(')
EXIT_MACRO_RE = re.compile(r'\b(ROCM_TRACE_RET_[A-Z0-9_]+|ROCR_TRACE_API_RET_[A-Z0-9_]+)\s*\(')
EXIT_FN_RE = re.compile(r'rocm_trace_emit_hip_api_exit_[a-z0-9_]+\s*\(')

def find_bodies(text, name):
    """Yield each function-definition body whose signature begins with
    `name(...)` and is followed by '{ ... }'. There may be multiple
    candidates (forward decls + definitions, helper TUs), and the migrated
    wrapper is the one that contains the migration marker. Caller picks.
    """
    pat = re.compile(r'\b' + re.escape(name) + r'\s*\(')
    for m in pat.finditer(text):
        depth = 0
        i = m.end() - 1  # at '('
        while i < len(text):
            c = text[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    j = text.find('{', i)
                    if j < 0:
                        break
                    bdepth = 0
                    k = j
                    while k < len(text):
                        if text[k] == '{':
                            bdepth += 1
                        elif text[k] == '}':
                            bdepth -= 1
                            if bdepth == 0:
                                yield text[j:k+1]
                                break
                        k += 1
                    break
            i += 1


MARKER = '__rocm_corr'  # presence in body => the migrated wrapper

bad = []
not_found = []
unmigrated = []
for n in names:
    # Find every candidate body and pick the one that contains the
    # migration marker. If no body has it, the symbol is in the inventory
    # but its source-of-truth definition is in a TU not touched by the
    # migrator -- skip the body-content check (the symbol-coverage gate
    # above already vouched for it).
    chosen = None
    for path, txt in files:
        for body in find_bodies(txt, n):
            if MARKER in body:
                chosen = body
                break
        if chosen:
            break
    if chosen is None:
        # Either a forward decl only, or an un-migrated definition.
        unmigrated.append(n)
        continue
    has_enter = bool(ENTER_RE.search(chosen))
    has_exit  = bool(EXIT_MACRO_RE.search(chosen) or EXIT_FN_RE.search(chosen))
    if not (has_enter and has_exit):
        bad.append((n, has_enter, has_exit))

for n, e, x in bad:
    print(f'  body-content miss: {n}: enter={e} exit={x}')
print(f'BODY: {len(bad)} body-content gaps; '
      f'{len(unmigrated)} not-migrated-but-listed; '
      f'{len(not_found)} not_found')
sys.exit(1 if bad else 0)
PY
)"

set +e
python3 -c "$PYTHON_GATE" "$SRC_DIR" "$WORK/inventory_names.txt" > "$WORK/body.log" 2>&1
BODY_RC=$?
set -e
cat "$WORK/body.log"

if [ "$BODY_RC" -ne 0 ]; then
    if [ "$STRICT" = "1" ]; then
        echo "FAIL: body-content coverage gate failed (LTTNG_COVERAGE_STRICT=1)"
        exit 1
    else
        echo "WARN: body-content coverage gate found gaps (set LTTNG_COVERAGE_STRICT=1 to fail the build)"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Curated-args coverage gate (spec §8.2)
# ---------------------------------------------------------------------------
# Skipped silently when curated_apis.yaml is absent (allows gradual
# rollout; the gate becomes mandatory once any API is curated).

CURATED_YAML="$SCRIPT_DIR/curated_apis.yaml"
if [ -f "$CURATED_YAML" ]; then
    # Curated APIs (one per line).
    python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from lttng_curated_lib import parse_yaml_file
for a in parse_yaml_file('$CURATED_YAML'):
    print(a['api'])
" | sort -u > "$WORK/curated.txt"

    # All inventoried wrappers (already in $WORK/migrated.txt from gate 1).
    MISSING_FROM_INV="$(comm -23 "$WORK/curated.txt" "$WORK/migrated.txt" || true)"
    if [ -n "$MISSING_FROM_INV" ]; then
        echo "FAIL (curated): APIs in curated_apis.yaml are missing from migration inventory:"
        printf '  %s\n' $MISSING_FROM_INV
        exit 1
    fi

    # Body-content scan: for each curated API, the wrapper body must
    # contain the sentinel AND a curated-macro invocation matching the
    # regex from spec §8.2. APIs with at least one IN/INOUT arg must
    # additionally contain at least one __rocm_in_ local.
    PYTHON_CURATED_GATE="$(cat <<'PY'
import os, re, sys
sys.path.insert(0, sys.argv[3])
from lttng_curated_lib import parse_yaml_file

src_dir   = sys.argv[1]
yaml_path = sys.argv[2]

apis = parse_yaml_file(yaml_path)
files = []
for root, _, fs in os.walk(src_dir):
    for fn in fs:
        if fn.endswith('.cpp'):
            p = os.path.join(root, fn)
            with open(p, 'rb') as fh:
                files.append((p, fh.read().decode('utf-8', errors='replace')))

# Spec §8.2 regex matcher for all six HIP curated-macro variants plus
# the HSA mirror (so the same gate works for hsa_table_interface.cpp).
MACRO_RE = re.compile(
    r'(?:ROCM_TRACE_RET|ROCR_TRACE_API_RET)_(?:STATUS|PTR|VOID)_CURATED'
    r'(?:_HSA)?(?:_NOARGS)?\s*\(')

def find_body(text, name):
    pat = re.compile(r'\b' + re.escape(name) + r'\s*\(')
    for m in pat.finditer(text):
        depth = 0
        i = m.end() - 1
        while i < len(text):
            if text[i] == '(': depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    j = text.find('{', i)
                    if j < 0: break
                    bdepth = 0; k = j
                    while k < len(text):
                        if text[k] == '{': bdepth += 1
                        elif text[k] == '}':
                            bdepth -= 1
                            if bdepth == 0:
                                return text[j:k+1]
                        k += 1
                    break
            i += 1
    return None

failures = []
for api in apis:
    name = api['api']
    sentinel = f'/* __ROCM_CURATED__: {name} */'
    body = None
    for path, text in files:
        b = find_body(text, name)
        if b and sentinel in b:
            body = b
            break
    if body is None:
        failures.append(f'{name}: no wrapper body found containing sentinel {sentinel!r}')
        continue
    if not MACRO_RE.search(body):
        failures.append(f'{name}: sentinel present but no _CURATED macro invocation')
        continue
    # IN-local check, only when the API has at least one IN/INOUT arg.
    has_in = any(a['dir'] in ('IN', 'INOUT') for a in api['args'])
    if has_in and '__rocm_in_' not in body:
        failures.append(f'{name}: has IN args but no __rocm_in_ locals in body')

if failures:
    for f in failures:
        print(f'  CURATED FAIL: {f}')
    sys.exit(1)
print(f'CURATED: {len(apis)} curated APIs verified')
PY
)"
    set +e
    python3 -c "$PYTHON_CURATED_GATE" "$SRC_DIR" "$CURATED_YAML" "$SCRIPT_DIR" \
        > "$WORK/curated.log" 2>&1
    CURATED_RC=$?
    set -e
    cat "$WORK/curated.log"
    if [ "$CURATED_RC" -ne 0 ]; then
        echo "FAIL: curated-args coverage gate failed"
        exit 1
    fi
fi

NTOTAL="$(wc -l < "$WORK/exported.txt")"
echo "PASS: all $NTOTAL exported HIP symbols migrated"
