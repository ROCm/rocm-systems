#!/usr/bin/env python3
"""Codegen: curated_apis.yaml -> rocm_<provider>_curated_tp.h + rocm_trace_emit_curated.h.

Per spec §5. The output files are checked in (see spec §3.2). Build does
not invoke this script by default; it runs only when the developer
explicitly regenerates (opt-in CMake target) or the CI drift gate runs it
and asserts `git diff --exit-code`.

Usage:
    python3 lttng_curated_codegen.py \\
        --yaml      path/to/curated_apis.yaml \\
        --provider  rocm_hip                  \\
        --status-type hipError_t              \\
        --status-success hipSuccess           \\
        --out-tp    path/to/rocm_hip_curated_tp.h \\
        --out-emit  path/to/rocm_trace_emit_curated.h

CI usage (wired into .github/workflows/lttng-curated-gates.yml by Task 13.5):

    # Drift gate: codegen output (consuming checked-in sidecar) must
    # match checked-in headers. Runs without libclang.
    python3 projects/clr/hipamd/scripts/lttng_curated_codegen.py \\
        --yaml projects/clr/hipamd/scripts/curated_apis.yaml \\
        --sigs projects/clr/hipamd/scripts/curated_apis_sigs.json \\
        --provider rocm_hip --status-type hipError_t --status-success hipSuccess \\
        --out-tp projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h \\
        --out-emit projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h
    git diff --exit-code -- 'projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h' \\
                            'projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h'

    # Verifier gate: YAML signatures must match HIP headers; also
    # rewrites curated_apis_sigs.json so a header-side signature change
    # for an OUT-handle param is caught by the diff (C10 fix).
    python3 projects/clr/hipamd/scripts/lttng_curated_verify.py \\
        --yaml projects/clr/hipamd/scripts/curated_apis.yaml \\
        --header /opt/rocm/include/hip/hip_runtime_api.h \\
        --extra-arg=-D__HIP_PLATFORM_AMD__=1 --extra-arg=-I/opt/rocm/include \\
        --out-sidecar projects/clr/hipamd/scripts/curated_apis_sigs.json
    git diff --exit-code -- 'projects/clr/hipamd/scripts/curated_apis_sigs.json'
"""
import argparse, hashlib, os, sys, textwrap
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lttng_curated_lib import parse_yaml_file, expanded_field_count

# ---- Per-DSL-type emit-side info ----
# (lttng_field_macro, c_type_for_helper_param, cast_expression_template)
TYPE_INFO = {
    'handle':      ('lttng_ust_field_integer_hex', 'uint64_t', '(uint64_t)(uintptr_t)({arg})'),
    'ptr':         ('lttng_ust_field_integer_hex', 'uint64_t', '(uint64_t)(uintptr_t)({arg})'),
    'device_ptr':  ('lttng_ust_field_integer_hex', 'uint64_t', '(uint64_t)({arg})'),
    'size':        ('lttng_ust_field_integer',     'uint64_t', '(uint64_t)({arg})'),
    'int32':       ('lttng_ust_field_integer',     'int32_t',  '(int32_t)({arg})'),
    'uint32':      ('lttng_ust_field_integer',     'uint32_t', '(uint32_t)({arg})'),
    'int64':       ('lttng_ust_field_integer',     'int64_t',  '(int64_t)({arg})'),
    'uint64':      ('lttng_ust_field_integer',     'uint64_t', '(uint64_t)({arg})'),
    'float':       ('lttng_ust_field_float',       'float',    '(float)({arg})'),
    'enum':        ('lttng_ust_field_integer',     'int32_t',  '(int32_t)({arg})'),
    # Spec §4.1: bool emits canonical 0/1 via !!() to be storage-rep-independent.
    'bool':        ('lttng_ust_field_integer',     'uint32_t', '(uint32_t)(!!({arg}))'),
    'cstring':     ('lttng_ust_field_string',      'const char*', '({arg} ? {arg} : "")'),
    # dim3 / dim3_packed handled specially in emit_tp_event / emit_helper.
}

# ---- Helper formal-param types (real C types for the wrapper signature) ----
# These are placeholders; the migrator passes the wrapper's actual params
# directly to the helper. They are encoded as "void const* generic" since
# the helper's job is to cast and emit, not to be type-strict at the
# helper boundary. We use the real types for the OUT-pointer case so the
# helper can deref safely.
HELPER_PARAM_TYPE = {
    'handle':     'uint64_t',     # always cast at call site
    'ptr':        'const void*',
    'device_ptr': 'uint64_t',     # hipDeviceptr_t passed as uint64
    'size':       'size_t',
    'int32':      'int32_t',
    'uint32':     'uint32_t',
    'int64':      'int64_t',
    'uint64':     'uint64_t',
    'float':      'float',
    'enum':       'int32_t',
    'bool':       'int',          # C bool promotes to int
    'cstring':    'const char*',
    'dim3':       'dim3',
    'dim3_packed': 'dim3',         # helper takes dim3, packs internally
}

# OUT-param helpers must take a typed pointer-to-the-out-type.
#
# For pointer/handle OUT params, the correct C type AND deref expression
# differ between providers (debate-review C10):
#   HIP: hipStream_t is a typedef'd void* -> hipStream_t* helper param is
#        compatible with void**, deref is *(void**)p.
#   HSA: hsa_signal_t is { uint64_t handle; } -> hsa_signal_t* helper
#        param is NOT compatible with void**; the deref must read
#        p->handle (struct field), not *p (which would read the first
#        8 bytes of the struct — happens to work for hsa_signal_t but is
#        UB-adjacent and breaks for any handle type whose first field is
#        not the uint64 handle).
#   HSA: hsa_queue_t** (pointer-to-pointer) -> deref *p (the pointer
#        itself is the handle).
#
# Resolution: codegen consumes a JSON sidecar emitted by the verifier
# (--sigs <path>; see Task 4) that contains the libclang-resolved real
# C type for each (api, arg) pair. The sidecar is checked in alongside
# the generated headers so the build doesn't need libclang. For the
# HIP-only Task 3 unit tests (which run before the sidecar exists), an
# absent sidecar falls back to the legacy `void**` behavior so existing
# golden tests keep working.
def out_helper_emit(arg, real_c_type=None):
    """Return (helper_param_type, deref_expr_template) for an OUT arg.

    `real_c_type` is the libclang-resolved C type from the verifier
    sidecar — e.g. 'void **', 'hsa_signal_t *', 'hsa_queue_t **',
    'size_t *', 'hipDeviceptr_t *'. When None (no sidecar; legacy /
    unit-test path), fall back to provider-agnostic void**/T* shapes.

    The deref_expr_template uses '{p}' for the helper param name
    (e.g. 'ptr_out_ptr') so callers can substitute.
    """
    ty = arg['type']
    # Numeric/scalar OUT types are unaffected by the C10 fix.
    if ty in ('size', 'uint32', 'uint64'):
        return (f"{HELPER_PARAM_TYPE[ty]}*", "*{p}")
    if ty in ('int32', 'int64'):
        return (f"{HELPER_PARAM_TYPE[ty]}*", "*{p}")
    if ty == 'float':
        return ('float*', "*{p}")
    if ty == 'enum':
        return ('int32_t*', "*{p}")
    if ty == 'bool':
        return ('int*', "*{p}")
    if ty not in ('ptr', 'handle', 'device_ptr'):
        raise SystemExit(f"OUT not supported for type {ty}")

    # Pointer/handle OUT — provider-aware shape from sidecar.
    if real_c_type is None:
        # Legacy fallback (HIP-style, used by Task 3 unit tests that
        # have no sidecar).
        return ('void**', "(uint64_t)(uintptr_t)(*{p})")

    rct = real_c_type.strip()
    # Normalize whitespace inside the type spelling (libclang emits
    # 'hsa_signal_t *' with the space).
    rct_compact = rct.replace(' ', '')

    # HSA pointer-to-pointer (e.g. hsa_queue_t**) — handle is the
    # pointer itself, deref one level.
    if rct_compact.startswith('hsa_') and rct_compact.endswith('**'):
        return (rct, "(uint64_t)(uintptr_t)(*{p})")

    # HSA pointer-to-struct (e.g. hsa_signal_t*, hsa_agent_t*) — read
    # the .handle field of the struct, NOT the first 8 bytes.
    if rct_compact.startswith('hsa_') and rct_compact.endswith('*'):
        return (rct, "({p}->handle)")

    # HIP and everything else — handle is itself a typedef'd pointer
    # (hipStream_t == void*), so the helper param is pointer-to-pointer
    # and the deref is one level.
    return (rct or 'void**', "(uint64_t)(uintptr_t)(*{p})")


def out_helper_param_type(arg, real_c_type=None):
    """Back-compat shim: return only the helper param C type."""
    return out_helper_emit(arg, real_c_type)[0]


# ---- Codegen: tp.h ----

def emit_tp_event(provider, api):
    """Emit one LTTNG_UST_TRACEPOINT_EVENT block for `api`."""
    name = api['api']
    args = api['args']

    # Build TP_ARGS list and TP_FIELDS list, expanding dim3 / direction.
    tp_args = ['uint64_t', 'corr_id']
    tp_fields = ['        lttng_ust_field_integer(uint64_t, corr_id, corr_id)']

    for a in args:
        nm  = a['name']
        ty  = a['type']
        dr  = a['dir']
        # Per spec §4.4 v1: INOUT is rejected upstream.
        # OUT: emit one field, named <name> (consumer reads value-or-zero).
        # IN:  emit one field, named <name>.
        # dim3: 3 fields (<name>_x, _y, _z); dim3_packed: 1 field (uint64 hex).
        if ty == 'dim3':
            for axis in ('x', 'y', 'z'):
                tp_args += ['uint32_t', f'{nm}_{axis}']
                tp_fields.append(
                    f'        lttng_ust_field_integer(uint32_t, {nm}_{axis}, {nm}_{axis})')
        elif ty == 'dim3_packed':
            tp_args += ['uint64_t', nm]
            tp_fields.append(
                f'        lttng_ust_field_integer_hex(uint64_t, {nm}, {nm})')
        else:
            field_macro, _, _ = TYPE_INFO[ty]
            # tp_args type depends on the lttng field's underlying C type.
            tp_arg_type = {
                'lttng_ust_field_integer':     {'int32_t': 'int32_t', 'uint32_t': 'uint32_t',
                                                'int64_t': 'int64_t', 'uint64_t': 'uint64_t'}.get(
                                                    field_macro and HELPER_PARAM_TYPE[ty], 'uint64_t'),
                'lttng_ust_field_integer_hex': 'uint64_t',
                'lttng_ust_field_float':       'float',
                'lttng_ust_field_string':      'const char*',
            }[field_macro]
            tp_args += [tp_arg_type, nm]
            if field_macro == 'lttng_ust_field_string':
                tp_fields.append(f'        {field_macro}({nm}, {nm})')
            elif field_macro == 'lttng_ust_field_float':
                tp_fields.append(f'        {field_macro}(float, {nm}, {nm})')
            else:
                tp_fields.append(f'        {field_macro}({tp_arg_type}, {nm}, {nm})')

    args_str   = ', '.join(tp_args)
    fields_str = '\n'.join(tp_fields)

    return textwrap.dedent(f"""\
        LTTNG_UST_TRACEPOINT_EVENT(
            {provider}, {name}_args,
            LTTNG_UST_TP_ARGS({args_str}),
            LTTNG_UST_TP_FIELDS(
        {fields_str}
            )
        )
        """)


def emit_tp_h(provider, apis, yaml_path, yaml_sha256):
    out = []
    # Curated-events guard. This file may be included by BOTH the parent
    # provider tp.h (for the LTTng probe-define pass in rocm_*_tp.cpp)
    # AND by the curated emit-helper header (which other TUs include
    # via rocm_trace_emit.h). Without this guard, the second include in
    # any TU that picks up both chains produces duplicate
    # LTTNG_UST_TRACEPOINT_EVENT registration symbols (redefinition).
    #
    # The guard mirrors the LTTng-UST multi-read pattern: it is bypassed
    # when LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ is defined so the probe
    # TU's second pass (with CREATE_PROBES + DEFINE) re-emits the event
    # registration data.
    guard = f"_{provider.upper()}_CURATED_TP_H"
    out.append(f"""/* AUTO-GENERATED by lttng_curated_codegen.py from {os.path.basename(yaml_path)}.
 * DO NOT EDIT BY HAND. Regenerate via the `regenerate-lttng-curated`
 * CMake target or by invoking the codegen script directly.
 *
 * SHA256({os.path.basename(yaml_path)}) at generation: {yaml_sha256}
 *
 * Provider: {provider}
 * API count: {len(apis)}
 *
 * Spec: docs/superpowers/specs/2026-04-26-lttng-curated-args-design.md
 */
#if !defined({guard}) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define {guard}
""")
    # Include the rocm_dim3_pack.h header in case any API uses dim3_packed.
    needs_dim3_pack = any(a['type'] == 'dim3_packed' for api in apis for a in api['args'])
    if needs_dim3_pack:
        out.append('/* dim3_packed encoding is defined in rocm_dim3_pack.h, included by\n'
                   ' * the emit-helper header that includes us transitively. */\n')
    for api in apis:
        out.append(emit_tp_event(provider, api))
    out.append(f"\n#endif /* {guard} */\n")
    return '\n'.join(out)


# ---- Codegen: emit.h ----

def emit_helper(provider, api, status_type, status_success, sigs=None):
    """Emit one static-inline helper function.

    `sigs` is the optional sidecar dict for THIS api: a list of
    {'name': ..., 'c_type': ...} entries from the verifier (None if no
    sidecar was supplied — fall back to provider-agnostic shapes).
    """
    name = api['api']
    args = api['args']

    # Index sidecar by arg name for O(1) lookup.
    sig_by_name = {s['name']: s['c_type'] for s in sigs} if sigs else {}

    # Helper formal-param list. Order: corr_id, captured-args..., status.
    formal_params = ['uint64_t corr_id']
    cast_exprs = []   # for the lttng_ust_do_tracepoint() call
    deref_setups = [] # for OUT params (size_t value computed before do_tp)

    # Track if we have any OUT param needing the success gate.
    has_out = any(a['dir'] == 'OUT' for a in args)

    for a in args:
        nm = a['name']
        ty = a['type']
        dr = a['dir']
        if dr == 'IN':
            ptype = HELPER_PARAM_TYPE[ty]
            formal_params.append(f"{ptype} {nm}")
            if ty == 'dim3':
                # Expand to 3 args at do_tp call site.
                cast_exprs.extend([
                    f"(uint32_t){nm}.x", f"(uint32_t){nm}.y", f"(uint32_t){nm}.z"])
            elif ty == 'dim3_packed':
                # Pack at the helper's local before do_tp.
                deref_setups.append(f"        const uint64_t {nm}_packed = ROCM_DIM3_PACK({nm});")
                cast_exprs.append(f"{nm}_packed")
            else:
                _, _, cast_tmpl = TYPE_INFO[ty]
                cast_exprs.append(cast_tmpl.format(arg=nm))
        elif dr == 'OUT':
            real_c = sig_by_name.get(nm)  # None if no sidecar
            ptype, deref_tmpl = out_helper_emit(a, real_c)
            formal_params.append(f"{ptype} {nm}_out_ptr")
            # Setup line: compute deref value, gated by status.
            # For ptr/handle/device_ptr — provider-aware deref expr from
            # out_helper_emit (see C10 fix). For other types — just deref.
            if ty in ('ptr', 'handle', 'device_ptr'):
                # deref_tmpl already produces a uint64_t-typed expression
                # (either (uint64_t)(uintptr_t)(*p) for pointers or
                # p->handle for HSA struct handles, both uint64_t).
                deref_expr = deref_tmpl.format(p=f"{nm}_out_ptr")
                deref_setups.append(textwrap.dedent(f"""\
                            const uint64_t {nm}_val =
                                (status == {status_success} && {nm}_out_ptr != NULL)
                                    ? (uint64_t)({deref_expr}) : 0ULL;""").rstrip())
                cast_exprs.append(f"{nm}_val")
            else:
                # Numeric OUT: read or zero.
                deref_setups.append(textwrap.dedent(f"""\
                            const auto {nm}_val =
                                (status == {status_success} && {nm}_out_ptr != NULL)
                                    ? *{nm}_out_ptr : 0;""").rstrip())
                _, _, cast_tmpl = TYPE_INFO[ty]
                cast_exprs.append(cast_tmpl.format(arg=f"{nm}_val"))
        elif dr == 'INOUT':
            # Validated upstream; can't reach here in v1.
            raise SystemExit(f"INOUT in {name} reached codegen — should be rejected by parser")

    # Status param is last, always present (unused for all-IN APIs).
    formal_params.append(f"{status_type} status")
    if not has_out:
        # Mark unused to suppress -Wunused-parameter.
        formal_params[-1] = f"{status_type} /*status*/ /* unused: all-IN API */"

    formal_str = ',\n    '.join(formal_params)
    cast_str   = ',\n            '.join(cast_exprs) if cast_exprs else ''
    do_tp_args = f"corr_id,\n            {cast_str}" if cast_str else "corr_id"
    setups     = '\n'.join(deref_setups)

    body_inner = (setups + '\n' if setups else '') + textwrap.dedent(f"""\
                lttng_ust_do_tracepoint({provider}, {name}_args, {do_tp_args});""")

    return textwrap.dedent(f"""\
        static inline void rocm_trace_emit_{name}_args(
            {formal_str}) {{
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled({provider}, {name}_args)) {{
        {body_inner}
            }}
        }}
        """)


def emit_noop_helper(api, status_type, sigs=None):
    """Emit a no-op helper for HIP_ENABLE_LTTNG_UST=0 mode.

    The no-op signature MUST byte-match the active-mode signature so
    callers compile identically in both modes (C10: provider-aware
    OUT-handle types).
    """
    name = api['api']
    args = api['args']
    sig_by_name = {s['name']: s['c_type'] for s in sigs} if sigs else {}
    formals = ['uint64_t']
    for a in args:
        if a['dir'] == 'OUT':
            real_c = sig_by_name.get(a['name'])
            formals.append(out_helper_param_type(a, real_c))
        elif a['type'] == 'dim3':
            formals.append('dim3')
        elif a['type'] == 'dim3_packed':
            formals.append('dim3')
        else:
            formals.append(HELPER_PARAM_TYPE[a['type']])
    formals.append(status_type)
    formal_str = ', '.join(formals)
    return f"static inline void rocm_trace_emit_{name}_args({formal_str}) {{}}\n"


def emit_emit_h(provider, apis, status_type, status_success, yaml_path, yaml_sha256, sigs_by_api=None):
    """Generate rocm_trace_emit_curated.h."""
    macro_guard = f"ROCM_{provider.upper().replace('ROCM_', '')}_TRACE_EMIT_CURATED_H_"
    enable_macro = 'HIP_ENABLE_LTTNG_UST' if provider == 'rocm_hip' else 'HSA_ENABLE_LTTNG_UST'

    out = []
    out.append(f"""/* AUTO-GENERATED by lttng_curated_codegen.py from {os.path.basename(yaml_path)}.
 * DO NOT EDIT BY HAND. SHA256({os.path.basename(yaml_path)}) at generation: {yaml_sha256}
 *
 * Per-API typed emit helpers for curated parameter capture. See spec §5.2.
 *
 * Helper signature invariant (spec §6.2): every helper takes
 *   (uint64_t corr_id, <captured-args...>, <status_type> status)
 * — status is the call's success status, used to gate OUT-param deref.
 * All-IN APIs accept it but mark it unused.
 */
#ifndef {macro_guard}
#define {macro_guard}

#include <stdint.h>
#include <stddef.h>
#include "rocm_trace_tid.h"
""")
    # Provider-runtime header — pulled in UNCONDITIONALLY (outside the
    # HIP_ENABLE_LTTNG_UST guard) because BOTH the active-mode helpers and
    # the no-op fallbacks use provider types in their signatures (e.g.
    # `hipStream_t *`, `hipGraphNode_t *` for OUT-handle helpers per the
    # C10 sidecar fix). Pre-Phase 14 the no-op branch only used
    # uint64/void*/etc. and worked without the include; expanding to the
    # full curated set added OUT-handle helpers in the no-op branch too.
    #
    # HIP: hip_runtime_api.h hard-#errors when neither __HIP_PLATFORM_AMD__
    # nor __HIP_PLATFORM_NVIDIA__ is defined, but several rocclr internal
    # TUs that pull in rocm_trace_emit.h transitively (e.g.
    # device/rocm/rocvirtual.cpp) don't set the define. Force the AMD
    # platform here so the include is self-contained — this is HIP-on-AMD
    # code by definition; the NVIDIA path is irrelevant for libamdhip64.
    #
    # IMPORTANT: this define MUST precede rocm_dim3_pack.h since that
    # header transitively pulls in <hip/hip_runtime_api.h> for `dim3`.
    if provider == 'rocm_hip':
        out.append("""\
/* Force the AMD platform define so the host-only HIP runtime header is
 * self-contained. rocclr internal TUs that pull in rocm_trace_emit.h
 * (e.g. device/rocm/rocvirtual.cpp) don't set this themselves. This
 * file is built only into libamdhip64; there is no NVIDIA path. */
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>
""")
    else:
        out.append('#include <hsa/hsa.h>\n#include <hsa/hsa_ext_amd.h>\n')

    # Only include rocm_dim3_pack.h if any API uses dim3 or dim3_packed.
    # HSA APIs use no dim3 types and the HSA tree has no rocm_dim3_pack.h,
    # so emitting this include unconditionally would break the HSA build.
    # Must come AFTER the platform define + hip_runtime_api.h include
    # since rocm_dim3_pack.h transitively pulls hip_runtime_api.h for dim3.
    needs_dim3 = any(a['type'] in ('dim3', 'dim3_packed')
                     for api in apis for a in api['args'])
    if needs_dim3:
        out.append('#include "rocm_dim3_pack.h"\n')

    out.append(f"""
#if defined({enable_macro}) && {enable_macro}

#include <atomic>
#include "{provider}_curated_tp.h"
""")

    # Reuse the same disabled flag as the existing generic helpers.
    # The rocm_trace_disabled() inline definition is guarded with
    # ROCM_TRACE_DISABLED_DEFINED so that when this header is co-included
    # with the hand-written rocm_trace_emit.h (which also defines this
    # function), the second definition is suppressed. The hand-written
    # header MUST set ROCM_TRACE_DISABLED_DEFINED after defining the
    # function (the wiring commit for Task 7 also adds the #define).
    out.append(f"""
extern std::atomic<bool> {provider}_trace_g_disabled;
#ifndef ROCM_TRACE_DISABLED_DEFINED
#define ROCM_TRACE_DISABLED_DEFINED
static inline bool rocm_trace_disabled(void) {{
    return {provider}_trace_g_disabled.load(std::memory_order_relaxed);
}}
#endif

""")

    for api in apis:
        sigs = (sigs_by_api or {}).get(api['api'])
        out.append(emit_helper(provider, api, status_type, status_success, sigs=sigs))
        out.append('\n')

    out.append(f"""
#else  /* {enable_macro} not defined — all helpers are no-ops */

""")
    for api in apis:
        sigs = (sigs_by_api or {}).get(api['api'])
        out.append(emit_noop_helper(api, status_type, sigs=sigs))

    out.append(f"""
#endif  /* {enable_macro} */

#endif  /* {macro_guard} */
""")
    return ''.join(out)


# ---- Main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--yaml',     required=True)
    ap.add_argument('--provider', required=True, choices=['rocm_hip', 'rocm_hsa'])
    ap.add_argument('--status-type', required=True,
                    help='C type for call status, e.g. hipError_t or hsa_status_t')
    ap.add_argument('--status-success', required=True,
                    help='Success-status sentinel, e.g. hipSuccess or HSA_STATUS_SUCCESS')
    ap.add_argument('--out-tp',   required=True)
    ap.add_argument('--out-emit', required=True)
    ap.add_argument('--sigs', default=None,
                    help='Optional verifier signature sidecar JSON '
                         '({api: [{name, c_type}, ...]}) used to choose '
                         'provider-correct OUT-handle helper signatures '
                         '(C10 fix). Without --sigs, OUT-handle helpers '
                         'fall back to void** (HIP-style).')
    args = ap.parse_args()

    apis = parse_yaml_file(args.yaml)
    with open(args.yaml, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    sigs_by_api = None
    if args.sigs:
        import json
        with open(args.sigs) as f:
            sigs_by_api = json.load(f)

    tp_text   = emit_tp_h(args.provider, apis, args.yaml, sha256)
    emit_text = emit_emit_h(args.provider, apis,
                             args.status_type, args.status_success,
                             args.yaml, sha256, sigs_by_api=sigs_by_api)

    os.makedirs(os.path.dirname(args.out_tp) or '.', exist_ok=True)
    with open(args.out_tp, 'w') as f:
        f.write(tp_text)
    with open(args.out_emit, 'w') as f:
        f.write(emit_text)
    print(f"wrote {args.out_tp} ({len(tp_text)} B), "
          f"{args.out_emit} ({len(emit_text)} B), {len(apis)} APIs",
          file=sys.stderr)

if __name__ == '__main__':
    main()
