#!/usr/bin/env python3
"""TraceLogging codegen: curated_apis.yaml (+ real HIP/HSA header signatures)
-> the generated TraceLogging curated emit files:

  - rocm_trace_emit_curated.h    (per-API emit-helper DECLARATIONS; identical
                                   in shape to the classic-LTTng codegen so the
                                   call sites and rocm_trace_emit.h are byte-
                                   compatible between backends)
  - rocm_trace_emit_curated.cpp  (out-of-line emit-helper DEFINITIONS using
                                   TraceLoggingWrite(); also defines the
                                   TRACELOGGING_DEFINE_PROVIDER and the
                                   register/unregister lifecycle wrappers)

This is a BACKEND SWAP of shared/lttng/scripts/lttng_curated_codegen.py: it
reuses that pipeline's YAML parsing (lttng_curated_lib) and libclang real-header
type resolution (lttng_curated_verify) verbatim -- the source of truth for each
curated arg's real C type is UNCHANGED. Only the emit bodies differ:
TraceLoggingWrite(provider, "<api>_enter", ...) at enter and
TraceLoggingWrite(provider, "<api>_exit", ...) at exit.

Key differences from the classic backend (see NOTES in the bead rocm-8px.2):

  * Distinct event names: enter and exit are separate event names
    ("<api>_enter" / "<api>_exit"). LTTng does not support two events with the
    same name in one provider, so the classic combined-<api>+phase scheme is
    replaced by two distinct names.
  * Distinct provider names: rocm_hsa_tlg / rocm_hip_tlg, so the TraceLogging
    provider coexists with the classic rocm_hsa / rocm_hip provider (still used
    by the hand-written non-curated events).
  * No field-count chunking: TraceLoggingWrite builds a real field array rather
    than expanding a variadic macro, so the per-event PAYLOAD_BUDGET / _args_N
    sibling-event mechanism is unnecessary and dropped. One _enter event carries
    all IN fields; one _exit event carries all OUT fields + the return field.
  * Explicit registration lifecycle: rocm_<provider>_tlg_register() /
    _unregister() wrappers (refcounted, mutex-guarded) are emitted into the .cpp
    and called from each runtime's real init/shutdown path.

Usage mirrors lttng_curated_codegen.py (--provider/--yaml/--header/--sigs/
--tp-out is unused here; --emit-out + --emit-cpp-out are the outputs).
"""
import argparse
import dataclasses
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
# Reuse the classic pipeline's library + verifier verbatim.
LTTNG_SCRIPTS = os.path.abspath(os.path.join(HERE, '..', '..', 'lttng', 'scripts'))
sys.path.insert(0, LTTNG_SCRIPTS)
from lttng_curated_lib import (parse_yaml_file, validate_api, ParseError,
                               BudgetError, TYPE_EXPANSION, DIR_EXPANSION)
from lttng_curated_verify import (parse_declarations, compute_sidecar,
                                   expand_compact_apis,
                                   resolve_curated_return_kind,
                                   AmbiguousDeclarationError,
                                   AmbiguousInferenceError,
                                   UnsupportedReturnTypeError)

# Repo root: shared/tracelogging/scripts -> shared/tracelogging -> shared -> root
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

# ---------------------------------------------------------------------------
# Return-field info, keyed by the curated return kind resolved by
# lttng_curated_verify.resolve_curated_return_kind. Same kinds as the classic
# backend; only the field macro differs (TraceLoggingXxx vs lttng_ust_field_*).
#
#   (field_name, TraceLogging field macro, emit-helper trailing-param C type)
#
# STATUS is captured as the raw int32 status code (retstatus). PTR is a hex
# uint64 (retptr). U64/I64/U32/I32 are the integer retval. VOID/STRUCT carry no
# return field (STRUCT has no meaningful scalar return; treated like VOID, same
# as the classic backend).
# ---------------------------------------------------------------------------
RETURN_FIELD_INFO = {
    'STATUS': ('retstatus', 'TraceLoggingInt32',     'int32_t'),
    'PTR':    ('retptr',    'TraceLoggingHexUInt64', 'uint64_t'),
    'U64':    ('retval',    'TraceLoggingUInt64',    'uint64_t'),
    'I64':    ('retval',    'TraceLoggingInt64',     'int64_t'),
    'U32':    ('retval',    'TraceLoggingUInt32',    'uint32_t'),
    'I32':    ('retval',    'TraceLoggingInt32',     'int32_t'),
    'VOID':   None,
    'STRUCT': None,
}

# ---------------------------------------------------------------------------
# Per-DSL-type emit-side info: (TraceLogging field macro, emit-side C type used
# for the value cast, cast expression template applied to the raw arg name at
# the TraceLoggingWrite call site).
#
# Type mapping per the bead:
#   handle/ptr/device_ptr -> TraceLoggingHexUInt64
#   size/uint32           -> TraceLoggingUInt32   (size widened to uint64 field
#                            below via the size row's own macro)
#   int32/enum/bool       -> TraceLoggingInt32 / etc
#   int64/uint64          -> TraceLoggingInt64 / TraceLoggingUInt64
#   float                 -> TraceLoggingFloat32
#   cstring               -> TraceLoggingString
# There is no TraceLoggingHexUInt32; size is captured as a uint64 field (it was
# a 64-bit field in the classic backend too), so keep size as UInt64.
# ---------------------------------------------------------------------------
TYPE_INFO = {
    'handle':     ('TraceLoggingHexUInt64', 'uint64_t', '(uint64_t)(uintptr_t)({arg})'),
    'ptr':        ('TraceLoggingHexUInt64', 'uint64_t', '(uint64_t)(uintptr_t)({arg})'),
    'device_ptr': ('TraceLoggingHexUInt64', 'uint64_t', '(uint64_t)({arg})'),
    'size':       ('TraceLoggingUInt64',    'uint64_t', '(uint64_t)({arg})'),
    'int32':      ('TraceLoggingInt32',     'int32_t',  '(int32_t)({arg})'),
    'uint32':     ('TraceLoggingUInt32',    'uint32_t', '(uint32_t)({arg})'),
    'int64':      ('TraceLoggingInt64',     'int64_t',  '(int64_t)({arg})'),
    'uint64':     ('TraceLoggingUInt64',    'uint64_t', '(uint64_t)({arg})'),
    'float':      ('TraceLoggingFloat32',   'float',    '(float)({arg})'),
    'enum':       ('TraceLoggingInt32',     'int32_t',  '(int32_t)({arg})'),
    # bool canonicalizes to a 0/1 uint32 (storage-rep-independent, matching the
    # classic backend's wire representation).
    'bool':       ('TraceLoggingUInt32',    'uint32_t', '(uint32_t)(!!({arg}))'),
    'cstring':    ('TraceLoggingString',    'const char*', '({arg} ? {arg} : "")'),
    # dim3 / dim3_packed handled specially in the emit paths.
}

# Helper formal-parameter C type for each DSL type (IN direction). Identical to
# the classic backend so the generated helper signatures are byte-compatible.
HELPER_PARAM_TYPE = {
    'handle':      'uint64_t',
    'ptr':         'const void*',
    'device_ptr':  'uint64_t',
    'size':        'size_t',
    'int32':       'int32_t',
    'uint32':      'uint32_t',
    'int64':       'int64_t',
    'uint64':      'uint64_t',
    'float':       'float',
    'enum':        'int32_t',
    'bool':        'int',
    'cstring':     'const char*',
    'dim3':        'dim3',
    'dim3_packed': 'dim3',
}


def out_helper_emit(arg, real_c_type=None):
    """Return (helper_param_type, deref_expr_template) for an OUT arg. Identical
    logic to lttng_curated_codegen.out_helper_emit so the generated OUT-helper
    signatures byte-match the classic backend."""
    ty = arg['type']
    if ty in ('size', 'uint32', 'uint64', 'int32', 'int64'):
        return (f"{HELPER_PARAM_TYPE[ty]}*", "*{p}")
    if ty == 'float':
        return ('float*', "*{p}")
    if ty == 'enum':
        return ('int32_t*', "*{p}")
    if ty == 'bool':
        return ('int*', "*{p}")
    if ty not in ('ptr', 'handle', 'device_ptr'):
        raise SystemExit(f"OUT not supported for DSL type {ty!r}")

    if real_c_type is None:
        return ('void**', "(uint64_t)(uintptr_t)(*{p})")

    rct = real_c_type.strip()
    rct_compact = rct.replace(' ', '')
    if rct_compact.startswith('hsa_') and rct_compact.endswith('**'):
        return (rct, "(uint64_t)(uintptr_t)(*{p})")
    if rct_compact.startswith('hsa_') and rct_compact.endswith('*'):
        return (rct, "({p}->handle)")
    return (rct or 'void**', "(uint64_t)(uintptr_t)(*{p})")


def out_helper_param_type(arg, real_c_type=None):
    return out_helper_emit(arg, real_c_type)[0]


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ProviderConfig:
    key: str                # CLI value: 'hip' or 'hsa'
    tlg_provider: str       # TraceLogging provider symbol: rocm_hip_tlg / rocm_hsa_tlg
    provider_name: str      # human-readable provider name string literal
    guid: str               # fixed arbitrary GUID (LTTNG ignores it)
    status_type: str        # e.g. 'hipError_t' / 'hsa_status_t'
    status_success: str     # e.g. 'hipSuccess' / 'HSA_STATUS_SUCCESS'
    emit_includes: str      # literal include block for the emit files
    clang_format_emit: bool


HIP_EMIT_INCLUDES = """\
/* Force the AMD platform define so the host-only HIP runtime header is
 * self-contained (rocclr internal TUs that pull this in don't set it). */
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>
"""

HSA_EMIT_INCLUDES = """\
/* HSA headers live flat in runtime/hsa-runtime/inc/ and that dir is on the
 * include path while building libhsa-runtime64; use the quoted form. */
#include "hsa.h"
#include "hsa_ext_amd.h"
"""

# Fixed, arbitrary provider GUIDs. LTTng ignores the providerId entirely (see
# TraceLoggingProvider.h: "The providerId is currently ignored"); these are
# stable placeholders so the generated TRACELOGGING_DEFINE_PROVIDER is
# deterministic. Values are ad-hoc v4-style GUIDs, distinct per provider.
HSA_GUID = "(0x726f636d,0x6873,0x6100,0x74,0x6c,0x67,0x00,0x00,0x00,0x00,0x01)"
HIP_GUID = "(0x726f636d,0x6869,0x7000,0x74,0x6c,0x67,0x00,0x00,0x00,0x00,0x02)"

PROVIDERS = {
    'hip': ProviderConfig(
        key='hip', tlg_provider='rocm_hip_tlg', provider_name='rocm_hip_tlg',
        guid=HIP_GUID,
        status_type='hipError_t', status_success='hipSuccess',
        emit_includes=HIP_EMIT_INCLUDES, clang_format_emit=True,
    ),
    'hsa': ProviderConfig(
        key='hsa', tlg_provider='rocm_hsa_tlg', provider_name='rocm_hsa_tlg',
        guid=HSA_GUID,
        status_type='hsa_status_t', status_success='HSA_STATUS_SUCCESS',
        emit_includes=HSA_EMIT_INCLUDES, clang_format_emit=False,
    ),
}


def enter_event_name(api_name):
    return f"{api_name}_enter"


def exit_event_name(api_name):
    return f"{api_name}_exit"


# ---------------------------------------------------------------------------
# Field-expression builders (per phase)
# ---------------------------------------------------------------------------
def _in_field_macros(arg):
    """TraceLogging field-macro call(s) for an IN arg at ENTER. Each entry is a
    fully-rendered `TraceLoggingXxx(value, "name")` string."""
    nm, ty = arg['name'], arg['type']
    if ty == 'dim3':
        return [f'TraceLoggingUInt32((uint32_t){nm}.x, "{nm}_x")',
                f'TraceLoggingUInt32((uint32_t){nm}.y, "{nm}_y")',
                f'TraceLoggingUInt32((uint32_t){nm}.z, "{nm}_z")']
    if ty == 'dim3_packed':
        return [f'TraceLoggingHexUInt64({nm}_packed, "{nm}")']
    macro, _cty, cast_tmpl = TYPE_INFO[ty]
    return [f'{macro}({cast_tmpl.format(arg=nm)}, "{nm}")']


def _out_field_macro(arg, val_name):
    """TraceLogging field macro for an OUT arg at EXIT, given the local scratch
    value name that already holds the resolved value."""
    nm, ty = arg['name'], arg['type']
    macro, _cty, _cast = TYPE_INFO[ty]
    return f'{macro}({val_name}, "{nm}")'


# ---------------------------------------------------------------------------
# Emit helper bodies (TraceLoggingWrite)
# ---------------------------------------------------------------------------
def emit_enter_helper(cfg, api, return_kind):
    """Active-mode `rocm_trace_emit_<api>_enter(<IN params>)`: one
    TraceLoggingWrite to the "<api>_enter" event with all IN fields."""
    name = api['api']
    formal_params = []
    setups = []
    field_macros = []
    for a in api['args']:
        nm, ty, dr = a['name'], a['type'], a['dir']
        if dr != 'IN':
            continue
        formal_params.append(f"{HELPER_PARAM_TYPE[ty]} {nm}")
        if ty == 'dim3_packed':
            setups.append(f"    const uint64_t {nm}_packed = ROCM_DIM3_PACK({nm});")
        field_macros.extend(_in_field_macros(a))

    formal_str = ',\n    '.join(formal_params) if formal_params else 'void'
    write_args = _render_write(cfg, enter_event_name(name), field_macros)
    body = []
    body.append("    if (rocm_trace_disabled()) return;")
    body.extend(setups)
    body.append(write_args)
    body_str = '\n'.join(body)
    return textwrap.dedent(f"""\
        void rocm_trace_emit_{name}_enter(
            {formal_str}) {{
        {body_str}
        }}
        """)


def emit_exit_helper(cfg, api, return_kind, sigs=None):
    """Active-mode EXIT helper: one TraceLoggingWrite to "<api>_exit" with all
    OUT fields (success-gated deref) + the return field. Same signature shape as
    the classic backend."""
    name = api['api']
    args = api['args']
    sig_by_name = {s['name']: s['c_type'] for s in sigs} if sigs else {}
    ret_info = RETURN_FIELD_INFO.get(return_kind)
    has_return = ret_info is not None
    has_out = any(a['dir'] == 'OUT' for a in args)
    if has_out and return_kind != 'STATUS':
        raise SystemExit(
            f"{name}: OUT args on a {return_kind}-returning API are not "
            f"supported (no success status to gate the OUT deref)")

    formal_params = []
    setups = []
    field_macros = []
    for a in args:
        nm, ty, dr = a['name'], a['type'], a['dir']
        if dr != 'OUT':
            continue
        real_c = sig_by_name.get(nm)
        ptype, deref_tmpl = out_helper_emit(a, real_c)
        formal_params.append(f"{ptype} {nm}_out_ptr")
        gate = f"status == {cfg.status_success} && "
        if ty in ('ptr', 'handle', 'device_ptr'):
            deref_expr = deref_tmpl.format(p=f"{nm}_out_ptr")
            setups.append(textwrap.dedent(f"""\
                    const uint64_t {nm}_val =
                        ({gate}{nm}_out_ptr != NULL)
                            ? (uint64_t)({deref_expr}) : 0ULL;""").rstrip())
            field_macros.append(_out_field_macro(a, f"{nm}_val"))
        else:
            setups.append(textwrap.dedent(f"""\
                    const auto {nm}_val =
                        ({gate}{nm}_out_ptr != NULL)
                            ? *{nm}_out_ptr : 0;""").rstrip())
            _macro, _cty, cast_tmpl = TYPE_INFO[ty]
            field_macros.append(
                _out_field_macro_cast(a, cast_tmpl.format(arg=f"{nm}_val")))

    ret_field_macro = None
    if has_return:
        field_name, macro, param_type = ret_info
        if return_kind == 'STATUS':
            formal_params.append(f"{cfg.status_type} status")
            ret_field_macro = f'{macro}((int32_t)status, "{field_name}")'
        else:
            formal_params.append(f"{param_type} {field_name}")
            ret_field_macro = f'{macro}({field_name}, "{field_name}")'
    if ret_field_macro is not None:
        field_macros.append(ret_field_macro)

    formal_str = ',\n    '.join(formal_params) if formal_params else 'void'
    write_stmt = _render_write(cfg, exit_event_name(name), field_macros)
    body = []
    body.append("    if (rocm_trace_disabled()) return;")
    # Indent OUT-scratch setups to 4 spaces (they were dedented to col 0).
    for s in setups:
        body.extend('    ' + line if line else line for line in s.split('\n'))
    body.append(write_stmt)
    body_str = '\n'.join(body)
    return textwrap.dedent(f"""\
        void rocm_trace_emit_{name}_exit(
            {formal_str}) {{
        {body_str}
        }}
        """)


def _out_field_macro_cast(arg, cast_expr):
    """OUT field macro where the value is a cast expression (scalar OUT args)."""
    nm, ty = arg['name'], arg['type']
    macro, _cty, _cast = TYPE_INFO[ty]
    return f'{macro}({cast_expr}, "{nm}")'


def _render_write(cfg, event_name, field_macros):
    """Render a single TraceLoggingWrite call, one field per line."""
    if not field_macros:
        return (f'    TraceLoggingWrite({cfg.tlg_provider}, "{event_name}");')
    lines = [f'    TraceLoggingWrite({cfg.tlg_provider}, "{event_name}",']
    for i, fm in enumerate(field_macros):
        sep = ',' if i < len(field_macros) - 1 else ');'
        lines.append(f'        {fm}{sep}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# No-op stubs (LTTng disabled) — byte-compatible signatures
# ---------------------------------------------------------------------------
def _enter_formals(cfg, api):
    formals = []
    for a in api['args']:
        if a['dir'] == 'IN':
            if a['type'] in ('dim3', 'dim3_packed'):
                formals.append('dim3')
            else:
                formals.append(HELPER_PARAM_TYPE[a['type']])
    return formals


def _exit_formals(cfg, api, return_kind, sigs=None):
    sig_by_name = {s['name']: s['c_type'] for s in sigs} if sigs else {}
    ret_info = RETURN_FIELD_INFO.get(return_kind)
    formals = []
    for a in api['args']:
        if a['dir'] == 'OUT':
            real_c = sig_by_name.get(a['name'])
            formals.append(out_helper_param_type(a, real_c))
    if ret_info is not None:
        if return_kind == 'STATUS':
            formals.append(cfg.status_type)
        else:
            formals.append(ret_info[2])  # param_type
    return formals


def emit_noop_enter_helper(cfg, api):
    formal_str = ', '.join(_enter_formals(cfg, api))
    return f"void rocm_trace_emit_{api['api']}_enter({formal_str}) {{}}\n"


def emit_noop_exit_helper(cfg, api, return_kind, sigs=None):
    formal_str = ', '.join(_exit_formals(cfg, api, return_kind, sigs=sigs))
    return f"void rocm_trace_emit_{api['api']}_exit({formal_str}) {{}}\n"


def emit_decl_enter_helper(cfg, api):
    formals = _enter_formals(cfg, api)
    formal_str = ', '.join(formals) if formals else 'void'
    return f"void rocm_trace_emit_{api['api']}_enter({formal_str});\n"


def emit_decl_exit_helper(cfg, api, return_kind, sigs=None):
    formals = _exit_formals(cfg, api, return_kind, sigs=sigs)
    formal_str = ', '.join(formals) if formals else 'void'
    return f"void rocm_trace_emit_{api['api']}_exit({formal_str});\n"


# ---------------------------------------------------------------------------
# emit.h (declarations)
# ---------------------------------------------------------------------------
def emit_emit_h(cfg, apis, banner, return_kinds, sigs_by_api=None):
    macro_guard = f"ROCM_{cfg.key.upper()}_TRACE_EMIT_CURATED_H_"
    out = [banner]
    out.append(f"#ifndef {macro_guard}\n#define {macro_guard}\n\n"
               f"#include <stdint.h>\n#include <stddef.h>\n")
    out.append(cfg.emit_includes)
    needs_dim3 = any(a['type'] in ('dim3', 'dim3_packed')
                     for api in apis for a in api['args'])
    if needs_dim3:
        out.append('#include "rocm_dim3_pack.h"\n')
    # Registration lifecycle entrypoints (called from the runtime init/shutdown
    # path). Refcounted + mutex-guarded in the .cpp; safe for HSA's repeated
    # init/shutdown and HIP's one-shot init.
    out.append(f"""
#ifdef __cplusplus
extern "C" {{
#endif
/* Register/unregister the TraceLogging provider {cfg.provider_name}.
 * Refcounted: register on the 0->1 transition, unregister on 1->0. Safe to
 * call across repeated runtime init/shutdown cycles (HSA) and once (HIP). */
void rocm_{cfg.key}_tlg_register(void);
void rocm_{cfg.key}_tlg_unregister(void);
#ifdef __cplusplus
}}
#endif
""")
    out.append('\n')
    for api in apis:
        sigs = (sigs_by_api or {}).get(api['api'])
        return_kind = return_kinds[api['api']]
        out.append(emit_decl_enter_helper(cfg, api))
        out.append(emit_decl_exit_helper(cfg, api, return_kind, sigs=sigs))
    out.append(f"\n#endif  /* {macro_guard} */\n")
    return ''.join(out)


# ---------------------------------------------------------------------------
# emit.cpp (TraceLoggingWrite definitions + provider + lifecycle)
# ---------------------------------------------------------------------------
def _provider_and_lifecycle(cfg, enable_macro):
    """The TRACELOGGING_DEFINE_PROVIDER, kill switch, and refcounted register/
    unregister wrappers. These must live in the SAME TU as every
    TraceLoggingWrite (the provider symbol is a linker-section token)."""
    return f"""\
#include <atomic>
#include <mutex>
#include <cstdlib>
#include <tracelogging/TraceLoggingProvider.h>

/* Distinct provider name from the classic rocm_{cfg.key} LTTNG_UST provider so
 * both can register in one process (LTTng allows one provider per name). */
TRACELOGGING_DEFINE_PROVIDER(
    {cfg.tlg_provider},
    "{cfg.provider_name}",
    {cfg.guid});

/* Runtime-wide kill switch, defined in the companion rocm_trace_init.cpp
 * (per-DSO name to avoid ELF interposition across the HSA/HIP runtimes). The
 * emit helpers below short-circuit on it. */
extern std::atomic<bool> rocm_{cfg.key}_trace_g_disabled;

#ifndef ROCM_TRACE_DISABLED_DEFINED
#define ROCM_TRACE_DISABLED_DEFINED
static inline bool rocm_trace_disabled(void) {{
    return rocm_{cfg.key}_trace_g_disabled.load(std::memory_order_relaxed);
}}
#endif

namespace {{
/* Registration state machine. register() on the 0->1 refcount transition,
 * unregister() on 1->0. The mutex serializes register/unregister so they never
 * overlap (TraceLoggingRegister/Unregister are documented as not overlap-safe)
 * and never double-register. HSA calls register/unregister once per
 * hsa_init/hsa_shut_down cycle; HIP calls register once (one-shot init) and
 * unregister once at teardown. The provider symbol is a linker-section token,
 * so these wrappers MUST live in the same TU as the TraceLoggingWrite calls. */
std::mutex g_tlg_reg_mutex;
int        g_tlg_refcount = 0;
}}  /* namespace */

extern "C" void rocm_{cfg.key}_tlg_register(void) {{
    std::lock_guard<std::mutex> lk(g_tlg_reg_mutex);
    if (g_tlg_refcount++ == 0) {{
        /* Ignoring the return value is documented-safe: on failure,
         * TraceLoggingWrite and TraceLoggingUnregister become no-ops. */
        TraceLoggingRegister({cfg.tlg_provider});
    }}
}}

extern "C" void rocm_{cfg.key}_tlg_unregister(void) {{
    std::lock_guard<std::mutex> lk(g_tlg_reg_mutex);
    if (g_tlg_refcount > 0 && --g_tlg_refcount == 0) {{
        TraceLoggingUnregister({cfg.tlg_provider});
    }}
}}
"""


def _noop_lifecycle(cfg):
    return f"""\
extern "C" void rocm_{cfg.key}_tlg_register(void) {{}}
extern "C" void rocm_{cfg.key}_tlg_unregister(void) {{}}
"""


def emit_emit_cpp(cfg, apis, banner, return_kinds, emit_header_basename,
                  sigs_by_api=None):
    enable_macro = 'HIP_ENABLE_LTTNG_UST' if cfg.key == 'hip' else 'HSA_ENABLE_LTTNG_UST'
    out = [banner]
    out.append(f'#include "{emit_header_basename}"\n')
    out.append(f"\n#if defined({enable_macro}) && {enable_macro}\n\n")
    out.append(_provider_and_lifecycle(cfg, enable_macro))
    out.append('\n')
    for api in apis:
        sigs = (sigs_by_api or {}).get(api['api'])
        return_kind = return_kinds[api['api']]
        out.append(emit_enter_helper(cfg, api, return_kind))
        out.append('\n')
        out.append(emit_exit_helper(cfg, api, return_kind, sigs=sigs))
        out.append('\n')
    out.append(f"\n#else  /* {enable_macro} not defined — all helpers are no-ops */\n\n")
    out.append(_noop_lifecycle(cfg))
    out.append('\n')
    for api in apis:
        sigs = (sigs_by_api or {}).get(api['api'])
        return_kind = return_kinds[api['api']]
        out.append(emit_noop_enter_helper(cfg, api))
        out.append(emit_noop_exit_helper(cfg, api, return_kind, sigs=sigs))
    out.append(f"\n#endif  /* {enable_macro} */\n")
    return ''.join(out)


# ---------------------------------------------------------------------------
# Banners (mirror lttng_curated_codegen)
# ---------------------------------------------------------------------------
def _regen_cmd(cfg, yaml_path, emit_out, emit_cpp_out=None, sigs_path=None,
               header_paths=None, source_paths=None, extra_args=None):
    def rel(p):
        try:
            return os.path.relpath(os.path.abspath(p), REPO_ROOT)
        except ValueError:
            return p
    def rel_extra_arg(e):
        if e.startswith('-I') and len(e) > 2:
            return f"-I{rel(e[2:])}"
        return e
    parts = [f"--provider {cfg.key}", f"--yaml {rel(yaml_path)}"]
    if sigs_path is not None:
        parts.append(f"--sigs {rel(sigs_path)}")
    else:
        for h in header_paths or []:
            parts.append(f"--header {rel(h)}")
        for s in source_paths or []:
            parts.append(f"--source {rel(s)}")
        for e in extra_args or []:
            parts.append(f"--extra-arg={rel_extra_arg(e)}")
    parts.append(f"--emit-out {rel(emit_out)}")
    if emit_cpp_out is not None:
        parts.append(f"--emit-cpp-out {rel(emit_cpp_out)}")
    lines = ["python3 shared/tracelogging/scripts/tracelogging_curated_codegen.py \\"]
    for i, p in enumerate(parts):
        lines.append(f"    {p}" + (" \\" if i < len(parts) - 1 else ""))
    return lines


def _comment_block(lines):
    out = ["/* " + lines[0]]
    for l in lines[1:]:
        out.append(" * " + l if l else " *")
    out.append(" */")
    return "\n".join(out) + "\n"


def emit_banner(cfg, yaml_path, sha256, regen_cmd, role_lines):
    return _comment_block([
        f"AUTO-GENERATED by tracelogging_curated_codegen.py from "
        f"{os.path.basename(yaml_path)}.",
        "Do not edit by hand — regenerate instead (see command below).",
        "",
        f"SHA256({os.path.basename(yaml_path)}) at generation: {sha256}",
        f"Provider: {cfg.provider_name}",
        "",
        *role_lines,
        "",
        "Regenerate with:",
        *(f"  {l}" for l in regen_cmd),
    ])


_EMIT_H_ROLE_LINES = [
    "Per-API typed emit-helper DECLARATIONS for curated parameter capture.",
    "TraceLogging backend: definitions in the companion .cpp use",
    "TraceLoggingWrite() against provider {p}. Signatures are byte-compatible",
    "with the classic LTTng backend so call sites are backend-agnostic.",
]

_EMIT_CPP_ROLE_LINES = [
    "Out-of-line TraceLoggingWrite definitions for the curated emit helpers.",
    "Also defines the TRACELOGGING_DEFINE_PROVIDER and the refcounted",
    "register/unregister lifecycle wrappers (same TU as the writes, since the",
    "provider symbol is a linker-section token). Real bodies when the enable",
    "macro is set; no-op stubs otherwise.",
]


# ---------------------------------------------------------------------------
# clang-format post-process (HIP only)
# ---------------------------------------------------------------------------
def _clang_format(text, style_file):
    clang_format = shutil.which('clang-format')
    if not clang_format:
        print("WARN: clang-format not found; leaving output un-formatted.",
              file=sys.stderr)
        return text
    r = subprocess.run([clang_format, f'--style=file:{style_file}'],
                       input=text, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"WARN: clang-format failed (rc={r.returncode}): {r.stderr}",
              file=sys.stderr)
        return text
    return r.stdout


# ---------------------------------------------------------------------------
# Resolution (delegates to the classic pipeline verbatim)
# ---------------------------------------------------------------------------
def load_sigs_by_api(yaml_path, sigs_path, header_paths, source_paths, extra_args):
    if sigs_path is not None:
        with open(sigs_path) as f:
            raw = json.load(f)
        return {api: (entry['params'] if isinstance(entry, dict) else entry)
                for api, entry in raw.items()}
    apis = parse_yaml_file(yaml_path)
    header_decls = parse_declarations(header_paths, source_paths, extra_args)
    return compute_sidecar(apis, header_decls)


def resolve_return_kinds(cfg, apis, sigs_path, header_paths, source_paths, extra_args):
    if sigs_path is not None:
        with open(sigs_path) as f:
            raw = json.load(f)
        return {api['api']: (raw.get(api['api'], {}).get('return_kind', 'STATUS')
                             if isinstance(raw.get(api['api']), dict) else 'STATUS')
                for api in apis}
    header_decls = parse_declarations(header_paths, source_paths, extra_args)
    return {api['api']: resolve_curated_return_kind(
                cfg.key, api['api'], header_decls.get(api['api'], []),
                [a['name'] for a in api['args']])
            for api in apis}


def resolve_apis(yaml_path, sigs_path, header_paths, source_paths, extra_args):
    raw_apis = parse_yaml_file(yaml_path)
    if sigs_path is not None:
        bad = [a['api'] for a in raw_apis if a['args'] and 'type' not in a['args'][0]]
        if bad:
            sys.exit(
                f"ERROR: --sigs mode requires the fully-explicit "
                f"{{name, type, dir}} YAML shape; compact-shape api(s): {bad}")
        return raw_apis
    header_decls = parse_declarations(header_paths, source_paths, extra_args)
    return expand_compact_apis(raw_apis, header_decls)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
def generate(cfg, apis, yaml_path, emit_out_path, sigs_by_api, return_kinds,
             regen_cmd):
    with open(yaml_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    emit_header_basename = os.path.basename(emit_out_path)

    role_h = [l.replace('{p}', cfg.provider_name) for l in _EMIT_H_ROLE_LINES]
    emit_text = emit_emit_h(
        cfg, apis, emit_banner(cfg, yaml_path, sha256, regen_cmd, role_h),
        return_kinds, sigs_by_api=sigs_by_api)
    emit_cpp_text = emit_emit_cpp(
        cfg, apis, emit_banner(cfg, yaml_path, sha256, regen_cmd, _EMIT_CPP_ROLE_LINES),
        return_kinds, emit_header_basename, sigs_by_api=sigs_by_api)

    if cfg.clang_format_emit:
        style_file = os.path.join(REPO_ROOT, 'projects', 'clr', '.clang-format')
        emit_text = _clang_format(emit_text, style_file)
        emit_cpp_text = _clang_format(emit_cpp_text, style_file)
    return emit_text, emit_cpp_text


def _diff(a_text, b_text, a_name, b_name):
    return ''.join(difflib.unified_diff(
        a_text.splitlines(keepends=True), b_text.splitlines(keepends=True),
        fromfile=a_name, tofile=b_name))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--provider', required=True, choices=sorted(PROVIDERS))
    ap.add_argument('--yaml', required=True)
    ap.add_argument('--sigs', default=None)
    ap.add_argument('--header', action='append', default=[])
    ap.add_argument('--source', action='append', default=[])
    ap.add_argument('--extra-arg', action='append', default=[])
    ap.add_argument('--emit-out', required=True)
    ap.add_argument('--emit-cpp-out', default=None)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    if args.sigs and (args.header or args.source):
        ap.error('--sigs is mutually exclusive with --header and --source')
    if not args.sigs and not args.header:
        ap.error('one of --sigs or --header is required')
    if args.source and not args.header:
        ap.error('--source requires --header')

    cfg = PROVIDERS[args.provider]
    try:
        sigs_by_api = load_sigs_by_api(args.yaml, args.sigs, args.header,
                                       args.source, args.extra_arg)
        apis = resolve_apis(args.yaml, args.sigs, args.header, args.source,
                            args.extra_arg)
        return_kinds = resolve_return_kinds(cfg, apis, args.sigs, args.header,
                                            args.source, args.extra_arg)
    except (AmbiguousDeclarationError, AmbiguousInferenceError, ParseError,
            BudgetError, UnsupportedReturnTypeError) as e:
        sys.exit(f"ERROR: {e}")

    regen_cmd = _regen_cmd(cfg, args.yaml, args.emit_out,
                           emit_cpp_out=args.emit_cpp_out, sigs_path=args.sigs,
                           header_paths=args.header, source_paths=args.source,
                           extra_args=args.extra_arg)
    emit_text, emit_cpp_text = generate(
        cfg, apis, args.yaml, args.emit_out, sigs_by_api, return_kinds, regen_cmd)

    outputs = [(args.emit_out, emit_text)]
    if args.emit_cpp_out:
        outputs.append((args.emit_cpp_out, emit_cpp_text))

    if args.check:
        rc = 0
        for out_path, new_text in outputs:
            old_text = ''
            if os.path.exists(out_path):
                with open(out_path) as f:
                    old_text = f.read()
            if old_text != new_text:
                rc = 1
                print(f"DRIFT: {out_path} does not match generator output:",
                      file=sys.stderr)
                print(_diff(old_text, new_text, out_path,
                            f"{out_path} (generated)"), file=sys.stderr)
        if rc == 0:
            print("OK: outputs match generator output", file=sys.stderr)
        sys.exit(rc)

    for out_path, new_text in outputs:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(new_text)
    print(", ".join(f"wrote {p} ({len(t)} B)" for p, t in outputs),
          file=sys.stderr)


if __name__ == '__main__':
    main()
