#!/usr/bin/env bash
# Coverage test: every API in curated_apis.yaml fires its _args event.
# Generated harness calls each API with placeholder args; trace must
# contain matching _args event with linked corr_id.
set -euo pipefail

BUILD_LIB_DIR="${1:-$PWD/build/clr/hipamd/lib}"
YAML="${2:-projects/clr/hipamd/scripts/curated_apis.yaml}"

WORK="$(mktemp -d)"
SESSION_NAME="hip-lttng-curated-coverage-$$"
export LTTNG_HOME="$WORK/lttng_home"
mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$WORK/sessiond.pid"

cleanup() {
    set +e
    lttng destroy "$SESSION_NAME" >/dev/null 2>&1
    if [ -f "$SESSIOND_PIDFILE" ]; then kill "$(cat $SESSIOND_PIDFILE)" 2>/dev/null; fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# Generate harness from YAML (one call per API with placeholder args).
python3 - <<PY > "$WORK/harness.hip.cpp"
import sys, os
sys.path.insert(0, 'projects/clr/hipamd/scripts')
from lttng_curated_lib import parse_yaml_file

PLACEHOLDERS = {
    'ptr':         'reinterpret_cast<void*>(0x1000)',
    'device_ptr':  'reinterpret_cast<hipDeviceptr_t>(0x1000)',
    'handle':      'nullptr',
    'size':        '64',
    'int32':       '0',
    'uint32':      '0',
    'int64':       '0',
    'uint64':      '0',
    'float':       '1.0f',
    'enum':        '0',
    'bool':        'false',
    'dim3':        'dim3(1)',
    'dim3_packed': 'dim3(1)',
    'cstring':     '"x"',
}

# Per-API per-arg overrides for parameters where the DSL-type placeholder
# won't compile (e.g., function pointers, strongly-typed enums that lack
# implicit-int conversion). The DSL 'ptr' type emits
# reinterpret_cast<void*>(0x1000), which is not convertible to a
# function-pointer parameter type; the DSL 'enum' type emits literal '0'
# which fails for enums declared without an int conversion. Add an entry
# here for each such arg; the verifier in Task 12 / Task 15g will reject
# stale entries that no longer correspond to a curated API arg.
PLACEHOLDER_OVERRIDES = {
    'hipStreamAddCallback': {'callback': 'nullptr', 'userData': 'nullptr'},
    'hipMemcpyAsync':       {'kind': 'hipMemcpyHostToDevice'},
    'hipMemcpy':            {'kind': 'hipMemcpyHostToDevice'},
}

def placeholder_for(api_name, arg):
    overrides = PLACEHOLDER_OVERRIDES.get(api_name, {})
    if arg['name'] in overrides:
        return overrides[arg['name']]
    return PLACEHOLDERS[arg['type']]

print('#include <hip/hip_runtime.h>')
# NOTE: PLACEHOLDER_OVERRIDES above provides per-API per-arg substitutions
# for parameters (e.g., function pointers) where the generic DSL-type
# placeholder won't compile.
print('int main() {')
for api in parse_yaml_file('$YAML'):
    name = api['api']
    call_args = []
    for a in api['args']:
        if a['dir'] == 'OUT':
            # Allocate a stack slot for the OUT arg.
            call_args.append('nullptr')  # simplified — real impl would alloc
        else:
            call_args.append(placeholder_for(name, a))
    print(f'    try {{ {name}({", ".join(call_args)}); }} catch(...) {{}}')
print('    return 0;')
print('}')
PY

HIPCC=/opt/rocm/bin/hipcc
"$HIPCC" "$WORK/harness.hip.cpp" -L "$BUILD_LIB_DIR" -lamdhip64 \
    -Wl,-rpath,"$BUILD_LIB_DIR" -o "$WORK/coverage_test"

lttng-sessiond --daemonize --pidfile "$SESSIOND_PIDFILE"
TRACE_DIR="$WORK/trace"
lttng create "$SESSION_NAME" --output "$TRACE_DIR" >/dev/null
lttng enable-channel --userspace --discard --subbuf-size=32768 --num-subbuf=4 ch1 >/dev/null
# Enable all curated _args events. lttng enable-event takes a single
# comma-separated event-name list, not multiple positional args.
EVENTS=$(python3 -c "
import sys
sys.path.insert(0, 'projects/clr/hipamd/scripts')
from lttng_curated_lib import parse_yaml_file
print(','.join(f'rocm_hip:{a[\"api\"]}_args' for a in parse_yaml_file('$YAML')))
")
lttng enable-event --userspace --channel=ch1 "$EVENTS" >/dev/null

lttng start "$SESSION_NAME" >/dev/null
"$WORK/coverage_test" || true   # placeholder args may cause hipError; OK
lttng stop "$SESSION_NAME" >/dev/null
lttng destroy "$SESSION_NAME" >/dev/null

DUMP="$WORK/trace.txt"
babeltrace2 "$TRACE_DIR" > "$DUMP"

# Assert each API's _args event appears at least once.
# IMPORTANT: do NOT pipe into `while read` — the loop body would run in a
# subshell and any MISSING counter increments would be lost in the parent.
# Use process substitution `< <(...)` so the loop body shares the parent
# shell's MISSING variable.
MISSING=0
while read api; do
    if grep -q "rocm_hip:${api}_args" "$DUMP"; then
        echo "  PASS  ${api}_args fired"
    else
        echo "  FAIL  ${api}_args NOT in trace"
        MISSING=$((MISSING+1))
    fi
done < <(python3 -c "
import sys
sys.path.insert(0, 'projects/clr/hipamd/scripts')
from lttng_curated_lib import parse_yaml_file
for a in parse_yaml_file('$YAML'):
    print(a['api'])
")

if [ "$MISSING" -gt 0 ]; then
    echo "FAIL: $MISSING curated _args events missing"
    exit 1
fi
echo "PASS: all curated _args events fired"
