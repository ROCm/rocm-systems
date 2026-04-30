#!/usr/bin/env bash
# LTTng coverage gate (HSA).
#
# 1. Symbol coverage: diff the public HSA symbols exported from
#    libhsa-runtime64.so against the LTTng migration inventory.
#    Fails the build on any miss.
#
# 2. Body-content coverage (defense-in-depth): for each migrated HSA
#    wrapper, confirm its function body contains:
#      - >=1 rocm_trace_emit_hsa_api_enter call
#      - AND (>=1 ROCR_TRACE_API_RET_* macro
#        OR    >=1 rocm_trace_emit_hsa_api_exit_* call)
#    Guards against drift (hand edits, partial reverts). WARN by default;
#    set LTTNG_COVERAGE_STRICT=1 to make body-content gaps fatal.
#
# Usage:
#   lttng_coverage_gate.sh <path/to/libhsa-runtime64.so> <inventory.txt>
#
# Exits 0 on full coverage, non-zero otherwise.
set -euo pipefail

SO="${1:?usage: $0 <path/to/libhsa-runtime64.so> <inventory.txt>}"
INV="${2:?inventory required}"

if [ ! -f "$SO" ]; then
    echo "FAIL: shared object not found: $SO"
    exit 1
fi
if [ ! -f "$INV" ]; then
    echo "FAIL: inventory not found: $INV"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# hsa-runtime root is the parent of scripts/.
HSA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# ---------------------------------------------------------------------------
# 1. Symbol-coverage gate
# ---------------------------------------------------------------------------
# Extract exported text symbols starting with hsa_ or __hsa_.
nm -D --defined-only "$SO" \
  | awk '$2 ~ /^[Tt]$/ && $3 ~ /^(hsa_|__hsa_)/ {
           sub(/@.*/, "", $3); print $3
         }' \
  | sort -u > "$TMP_DIR/exported.txt"

# Migrated symbols (first column of inventory).
awk '{print $1}' "$INV" | sort -u > "$TMP_DIR/migrated.txt"

# Aliases declared in hsacore.so.link share the address (and function body)
# of an already-migrated symbol. Add them so the gate does not flag them.
LINK_SCRIPT="$(dirname "$INV")/../hsacore.so.link"
if [ -f "$LINK_SCRIPT" ]; then
    awk '/=/ { sub(/;.*/, ""); split($0, a, "="); gsub(/[ \t]/, "", a[1]); print a[1] }' \
        "$LINK_SCRIPT" >> "$TMP_DIR/migrated.txt"
    sort -u -o "$TMP_DIR/migrated.txt" "$TMP_DIR/migrated.txt"
fi

MISSING=$(comm -23 "$TMP_DIR/exported.txt" "$TMP_DIR/migrated.txt" || true)

# Some symbols are intentionally NOT in hsa_table_interface.cpp — they are
# implemented in other TUs (e.g., hsa_ext_image*.cpp, hsa_ext_program* in the
# loader, etc.) but still get exported via the version script. Those are not
# part of the table-interface migration scope.
ALLOWLIST_RE='^hsa_ext_image_|^hsa_ext_sampler_|^hsa_ext_program_|^hsa_amd_image_get_info_max_dim$|^hsa_ven_amd_|^__hsa_amd_aql_queue_get_async_handler$|^__hsa_amd_aql_queue_set_async_handler$'

if [ -n "$MISSING" ]; then
    UNEXPECTED=$(echo "$MISSING" | grep -vE "$ALLOWLIST_RE" || true)
    if [ -n "$UNEXPECTED" ]; then
        echo "FAIL: the following exported HSA symbols were NOT migrated to LTTng:"
        echo "$UNEXPECTED"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 2. Body-content gate (defense-in-depth)
# ---------------------------------------------------------------------------
STRICT="${LTTNG_COVERAGE_STRICT:-0}"

# We scan only hsa_table_interface.cpp -- that's where every migrated
# wrapper lives.
SRC_FILE="$HSA_ROOT/core/common/hsa_table_interface.cpp"
if [ ! -f "$SRC_FILE" ]; then
    echo "WARN: $SRC_FILE not found; skipping body-content gate"
else
    PYTHON_GATE="$(cat <<'PY'
import re, sys

src_path = sys.argv[1]
inv_path = sys.argv[2]

with open(src_path) as f:
    text = f.read()
with open(inv_path) as f:
    names = [ln.split('\t', 1)[0].strip() for ln in f if ln.strip()]

ENTER_RE = re.compile(r'rocm_trace_emit_hsa_api_enter\s*\(')
EXIT_MACRO_RE = re.compile(r'\bROCR_TRACE_API_RET_[A-Z0-9_]+\s*\(')
EXIT_FN_RE = re.compile(r'rocm_trace_emit_hsa_api_exit_[a-z0-9_]+\s*\(')

def find_bodies(text, name):
    """Yield each function-definition body whose signature begins with
    `name(...)` followed by '{ ... }'."""
    pat = re.compile(r'\b' + re.escape(name) + r'\s*\(')
    for m in pat.finditer(text):
        depth = 0
        i = m.end() - 1
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


MARKER = '__rocm_corr'

bad = []
not_found = []
unmigrated = []
for n in names:
    chosen = None
    for body in find_bodies(text, n):
        if MARKER in body:
            chosen = body
            break
    if chosen is None:
        # Listed in inventory but definition we found has no migration
        # marker -- treat as un-migrated for body-content purposes.
        unmigrated.append(n)
        continue
    has_enter = bool(ENTER_RE.search(chosen))
    has_exit = bool(EXIT_MACRO_RE.search(chosen) or EXIT_FN_RE.search(chosen))
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
    python3 -c "$PYTHON_GATE" "$SRC_FILE" "$INV" > "$TMP_DIR/body.log" 2>&1
    BODY_RC=$?
    set -e
    cat "$TMP_DIR/body.log"
    if [ "$BODY_RC" -ne 0 ]; then
        if [ "$STRICT" = "1" ]; then
            echo "FAIL: body-content coverage gate failed (LTTNG_COVERAGE_STRICT=1)"
            exit 1
        else
            echo "WARN: body-content coverage gate found gaps (set LTTNG_COVERAGE_STRICT=1 to fail the build)"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 3. HSA curated-args coverage gate
# ---------------------------------------------------------------------------
# Skipped silently when curated_apis.yaml is absent (allows gradual
# rollout; the gate becomes mandatory once any API is curated).

CURATED_YAML="$SCRIPT_DIR/curated_apis.yaml"
if [ -f "$CURATED_YAML" ]; then
    # Curated APIs (one per line). Drives the inventory-membership check
    # below; the actual body-content scan lives in lttng_coverage_check.py.
    python3 "$SCRIPT_DIR/lttng_coverage_check.py" list-curated \
        --yaml "$CURATED_YAML" | sort -u > "$TMP_DIR/curated.txt"

    # All inventoried wrappers (already in $TMP_DIR/migrated.txt from gate 1).
    MISSING_FROM_INV="$(comm -23 "$TMP_DIR/curated.txt" "$TMP_DIR/migrated.txt" || true)"
    if [ -n "$MISSING_FROM_INV" ]; then
        echo "FAIL (curated): APIs in HSA curated_apis.yaml are missing from migration inventory:"
        printf '  %s\n' $MISSING_FROM_INV
        exit 1
    fi

    # Body-content scan: each curated wrapper body must have the sentinel,
    # a _CURATED_HSA macro invocation, and (for APIs with
    # any IN/INOUT arg) at least one __rocm_in_ local. The shared
    # lttng_coverage_check.py:gate_curated regex matches both HIP
    # (_CURATED) and HSA (_CURATED_HSA) variants, so the same script
    # serves both providers.
    set +e
    python3 "$SCRIPT_DIR/lttng_coverage_check.py" curated \
        --src-dir "$HSA_ROOT" \
        --yaml "$CURATED_YAML" \
        > "$TMP_DIR/curated.log" 2>&1
    CURATED_RC=$?
    set -e
    cat "$TMP_DIR/curated.log"
    if [ "$CURATED_RC" -ne 0 ]; then
        echo "FAIL: HSA curated-args coverage gate failed"
        exit 1
    fi
fi

N_MIGRATED=$(wc -l < "$TMP_DIR/migrated.txt")
N_EXPORTED=$(wc -l < "$TMP_DIR/exported.txt")
echo "PASS: all $N_EXPORTED exported HSA symbols migrated ($N_MIGRATED inventory entries)"
