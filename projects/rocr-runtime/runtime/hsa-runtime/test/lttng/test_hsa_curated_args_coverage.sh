#!/usr/bin/env bash
# Coverage test: every API in HSA curated_apis.yaml fires its _args event.
#
# Generated harness calls each API with a mix of real HSA resources
# (real agent, real signal, real queue, real memory pool) and per-API
# overrides for parameters where placeholders won't compile (e.g. function
# pointers).
#
# All HSA APIs are called INLINE (not in fork()) because HSA's runtime
# spawns background threads (queue, signal, KFD interrupt) that fork()
# does not duplicate; calling APIs that depend on those threads from a
# forked child reliably crashes or hangs.
set -euo pipefail

BUILD_LIB_DIR="${1:-$PWD/build/rocr/rocr/lib}"
YAML="${2:-projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml}"

if [ ! -f "$BUILD_LIB_DIR/libhsa-runtime64.so" ] && \
   ! ls "$BUILD_LIB_DIR"/libhsa-runtime64.so* >/dev/null 2>&1; then
    echo "ERROR: libhsa-runtime64.so not found in $BUILD_LIB_DIR" >&2
    exit 2
fi

WORK="$(mktemp -d)"
SESSION_NAME="hsa-lttng-curated-coverage-$$"
export LTTNG_HOME="$WORK/lttng_home"
mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$WORK/sessiond.pid"

cleanup() {
    set +e
    lttng destroy "$SESSION_NAME" >/dev/null 2>&1
    if [ -f "$SESSIOND_PIDFILE" ]; then
        kill "$(cat $SESSIOND_PIDFILE)" 2>/dev/null
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# Generate the harness from the YAML so new curated APIs auto-extend.
python3 - "$YAML" > "$WORK/harness.cpp" <<'PY'
import sys
sys.path.insert(0, 'projects/rocr-runtime/runtime/hsa-runtime/scripts')
from lttng_curated_lib import parse_yaml_file

yaml_path = sys.argv[1]

# APIs to SKIP at the runtime call site (still validated statically by
# the coverage gate via sentinel + macro presence). Document each skip.
SKIP_APIS = {
    # hsa_amd_queue_intercept_create is not in the public version script
    # (`hsacore.so.def`) — it's an internal symbol used by
    # rocprofiler-register but not exported to user-space code that links
    # against -lhsa-runtime64. Calling it from the harness fails at link
    # time with 'undefined reference'. Static coverage gate verifies the
    # wrapper has the sentinel + _CURATED_HSA macro; runtime fire is
    # intentionally not exercised here.
    'hsa_amd_queue_intercept_create',
}

# Per-API per-arg overrides. Most APIs use real resources allocated at
# harness startup. Function-pointer params (callback, data) get nullptr.
PLACEHOLDER_OVERRIDES = {
    'hsa_queue_create': {
        'agent':                 'real_agent',
        'size':                  '256',
        'type':                  'HSA_QUEUE_TYPE_MULTI',
        'callback':              'nullptr',
        'data':                  'nullptr',
        'private_segment_size':  'UINT32_MAX',
        'group_segment_size':    'UINT32_MAX',
        'queue':                 '&real_queue_for_create',
    },
    'hsa_queue_destroy': {
        'queue': 'real_queue_for_destroy',
    },
    'hsa_amd_queue_intercept_create': {
        'agent_handle':          'real_agent',
        'size':                  '256',
        'type':                  'HSA_QUEUE_TYPE_MULTI',
        'callback':              'nullptr',
        'data':                  'nullptr',
        'private_segment_size':  'UINT32_MAX',
        'group_segment_size':    'UINT32_MAX',
        'queue':                 '&real_intercept_queue_for_create',
    },
    'hsa_signal_create': {
        'initial_value':  '0',
        'num_consumers':  '0',
        'consumers':      'NULL',
        'signal':         '&real_signal_for_create',
    },
    'hsa_signal_destroy': {
        'signal': 'real_signal_for_destroy',
    },
    'hsa_amd_signal_create': {
        'initial_value':  '0',
        'num_consumers':  '0',
        'consumers':      'NULL',
        'attributes':     '0',
        'signal':         '&real_amd_signal_for_create',
    },
    'hsa_amd_memory_pool_allocate': {
        'memory_pool':  'real_pool',
        'size':         '4096',
        'flags':        '0',
        'ptr':          '&real_pool_alloc_out',
    },
    'hsa_amd_memory_pool_free': {
        'ptr': 'real_pool_alloc_for_free',
    },
    'hsa_amd_memory_async_copy': {
        'dst':                 'real_pool_alloc_dst',
        'dst_agent':           'real_agent',
        'src':                 'real_pool_alloc_src',
        'src_agent':           'real_agent',
        'size':                '64',
        'num_dep_signals':     '0',
        'dep_signals':         'NULL',
        'completion_signal':   'real_completion_signal',
    },
    'hsa_amd_memory_async_copy_on_engine': {
        'dst':                 'real_pool_alloc_dst',
        'dst_agent':           'real_agent',
        'src':                 'real_pool_alloc_src',
        'src_agent':           'real_agent',
        'size':                '64',
        'num_dep_signals':     '0',
        'dep_signals':         'NULL',  # not in YAML but is a C param
        'completion_signal':   'real_completion_signal',
        'engine_id':           'HSA_AMD_SDMA_ENGINE_0',
        'force_copy_on_sdma':  'false',
    },
}

# C wrappers' actual call signatures may include MORE args than the YAML
# captures (e.g. async_copy_on_engine drops dep_signals from the YAML
# but the C call still needs it). The harness emits the actual C call,
# not the YAML-curated subset, so per-API overrides above include extra
# args (like dep_signals for async_copy_on_engine).
EXTRA_C_ARGS = {
    'hsa_amd_memory_async_copy_on_engine': [
        # YAML: dst, dst_agent, src, src_agent, size, num_dep_signals,
        #       completion_signal, engine_id, force_copy_on_sdma.
        # C:   dst, dst_agent, src, src_agent, size, num_dep_signals,
        #      dep_signals, completion_signal, engine_id, force_copy_on_sdma.
        # We need to slot 'dep_signals' between num_dep_signals and
        # completion_signal at the C call site.
    ],
}

print('#include <hsa/hsa.h>')
print('#include <hsa/hsa_ext_amd.h>')
print('#include <stdint.h>')
print('#include <stdio.h>')
print('#include <stdlib.h>')
print()
print('// Real-resource helpers.')
print('static hsa_status_t find_gpu(hsa_agent_t a, void* d) {')
print('  hsa_device_type_t t;')
print('  if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t) == HSA_STATUS_SUCCESS')
print('      && t == HSA_DEVICE_TYPE_GPU) {')
print('    *((hsa_agent_t*)d) = a; return HSA_STATUS_INFO_BREAK;')
print('  }')
print('  return HSA_STATUS_SUCCESS;')
print('}')
print('static hsa_status_t find_pool(hsa_amd_memory_pool_t p, void* d) {')
print('  hsa_amd_segment_t seg;')
print('  if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg)')
print('      == HSA_STATUS_SUCCESS && seg == HSA_AMD_SEGMENT_GLOBAL) {')
print('    *((hsa_amd_memory_pool_t*)d) = p; return HSA_STATUS_INFO_BREAK;')
print('  }')
print('  return HSA_STATUS_SUCCESS;')
print('}')
print()
print('int main() {')
print('  hsa_init();')
print()
print('  // Real resources allocated up front so coverage calls have valid args.')
print('  hsa_agent_t real_agent = {0};')
print('  hsa_iterate_agents(&find_gpu, &real_agent);')
print('  if (real_agent.handle == 0) {')
print('    fprintf(stderr, "no GPU agent; harness cannot exercise HSA APIs\\n");')
print('    hsa_shut_down(); return 0; // emit nothing rather than fail compile')
print('  }')
print()
print('  hsa_amd_memory_pool_t real_pool = {0};')
print('  hsa_amd_agent_iterate_memory_pools(real_agent, &find_pool, &real_pool);')
print()
print('  // Pre-allocated buffers / signals reused across coverage calls.')
print('  void* real_pool_alloc_out = NULL;  // OUT for pool_allocate')
print('  void* real_pool_alloc_for_free = NULL;')
print('  if (real_pool.handle != 0) {')
print('    hsa_amd_memory_pool_allocate(real_pool, 4096, 0, &real_pool_alloc_for_free);')
print('  }')
print('  void* real_pool_alloc_dst = NULL;')
print('  void* real_pool_alloc_src = NULL;')
print('  if (real_pool.handle != 0) {')
print('    hsa_amd_memory_pool_allocate(real_pool, 64, 0, &real_pool_alloc_dst);')
print('    hsa_amd_memory_pool_allocate(real_pool, 64, 0, &real_pool_alloc_src);')
print('  }')
print()
print('  hsa_signal_t real_signal_for_create = {0};')
print('  hsa_signal_t real_signal_for_destroy = {0};')
print('  hsa_signal_create(0, 0, NULL, &real_signal_for_destroy);')
print('  hsa_signal_t real_amd_signal_for_create = {0};')
print('  hsa_signal_t real_completion_signal = {0};')
print('  hsa_signal_create(1, 0, NULL, &real_completion_signal);')
print()
print('  hsa_queue_t* real_queue_for_create = NULL;')
print('  hsa_queue_t* real_queue_for_destroy = NULL;')
print('  hsa_queue_create(real_agent, 256, HSA_QUEUE_TYPE_MULTI, NULL, NULL,')
print('                   UINT32_MAX, UINT32_MAX, &real_queue_for_destroy);')
print('  hsa_queue_t* real_intercept_queue_for_create = NULL;')
print()
print('  // Per-API curated calls -- one per YAML API.')
for api in parse_yaml_file(yaml_path):
    name = api['api']
    if name in SKIP_APIS:
        print(f'  // SKIP {name}: not exported (see SKIP_APIS comment)')
        continue
    overrides = PLACEHOLDER_OVERRIDES.get(name, {})
    arg_strs = []
    # YAML order matches C call order EXCEPT for hsa_amd_memory_async_copy_on_engine
    # which has dep_signals injected between num_dep_signals and completion_signal.
    yaml_args = list(api['args'])
    for a in yaml_args:
        ph = overrides.get(a['name'], 'NULL')
        arg_strs.append(ph)
    # Inject dep_signals for async_copy_on_engine at the right position.
    if name == 'hsa_amd_memory_async_copy_on_engine':
        # YAML order: dst, dst_agent, src, src_agent, size, num_dep_signals,
        #             completion_signal, engine_id, force_copy_on_sdma
        # Insert dep_signals between num_dep_signals (idx 5) and
        # completion_signal (idx 6) → after position 5, before position 6.
        arg_strs.insert(6, 'NULL')
    print(f'  {name}({", ".join(arg_strs)});')
print()
print('  hsa_shut_down();')
print('  return 0;')
print('}')
PY

g++ -std=c++17 "$WORK/harness.cpp" -I/opt/rocm/include \
    -L "$BUILD_LIB_DIR" -lhsa-runtime64 -Wl,-rpath,"$BUILD_LIB_DIR" \
    -o "$WORK/coverage_test"

# Start sessiond.
lttng-sessiond --daemonize --pidfile "$SESSIOND_PIDFILE"
TRACE_DIR="$WORK/trace"
lttng create "$SESSION_NAME" --output "$TRACE_DIR" >/dev/null
lttng enable-channel --userspace --discard --subbuf-size=32768 --num-subbuf=4 ch1 >/dev/null

# Enable each <api>_args event from the YAML.
python3 - "$YAML" <<'PY' | xargs -r lttng enable-event --userspace --channel=ch1 >/dev/null
import sys
sys.path.insert(0, 'projects/rocr-runtime/runtime/hsa-runtime/scripts')
from lttng_curated_lib import parse_yaml_file
events = []
for a in parse_yaml_file(sys.argv[1]):
    events.append(f"rocm_hsa:{a['api']}_args")
print(','.join(events))
PY

lttng start "$SESSION_NAME" >/dev/null

APP_RC=0
"$WORK/coverage_test" || APP_RC=$?
echo "  coverage_test exit=$APP_RC"

lttng stop "$SESSION_NAME" >/dev/null
lttng destroy "$SESSION_NAME" >/dev/null

DUMP="$WORK/trace.txt"
babeltrace2 "$TRACE_DIR" > "$DUMP"

# Assert each <api>_args event appears at least once.
# Use process substitution (< <(...)) instead of pipe-into-while so the
# MISSING counter accumulates in the parent shell.
# APIs in the SKIP set (mirrored to harness above) are reported as SKIP
# rather than FAIL since the harness intentionally does not call them.
SKIP_RE='^(hsa_amd_queue_intercept_create)$'
MISSING=0
TOTAL=0
SKIPPED=0
while read api; do
    TOTAL=$((TOTAL+1))
    if [[ "$api" =~ $SKIP_RE ]]; then
        echo "  SKIP  ${api}_args (not exported; statically validated only)"
        SKIPPED=$((SKIPPED+1))
        continue
    fi
    if grep -q "rocm_hsa:${api}_args" "$DUMP"; then
        echo "  PASS  ${api}_args fired"
    else
        echo "  FAIL  ${api}_args NOT in trace"
        MISSING=$((MISSING+1))
    fi
done < <(python3 - "$YAML" <<'PY'
import sys
sys.path.insert(0, 'projects/rocr-runtime/runtime/hsa-runtime/scripts')
from lttng_curated_lib import parse_yaml_file
for a in parse_yaml_file(sys.argv[1]):
    print(a['api'])
PY
)

if [ "$MISSING" -gt 0 ]; then
    echo "FAIL: $MISSING / $TOTAL HSA curated _args events missing ($SKIPPED skipped)"
    exit 1
fi
echo "PASS: all $((TOTAL-SKIPPED)) HSA curated _args events fired ($SKIPPED skipped)"
