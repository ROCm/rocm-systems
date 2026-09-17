#!/usr/bin/env python3
"""Codegen: curated_apis.yaml (+ real HIP/HSA header signatures) -> the two
generated LTTng curated-args headers:

  - rocm_<provider>_curated_tp.h        (LTTNG_UST_TRACEPOINT_EVENT defs)
  - rocm_trace_emit_curated.h           (per-API emit-helper declarations)
  - rocm_trace_emit_curated.cpp         (out-of-line emit-helper definitions;
                                         emitted when --emit-cpp-out is given)

Signatures for OUT-handle helper generation come from a live libclang
parse of the real headers (--header, may repeat; --source for
implementation-only exported wrappers; --extra-arg for clang flags) — the
same parsing/duplicate-resolution logic lttng_curated_verify.py uses for its
drift gate, reused here rather than duplicated. `--sigs
<path>` (a pre-computed {api: [{name, c_type}, ...]} JSON file) is also
accepted as an alternative signature source, mainly for tests/synthetic
fixtures that don't want to depend on libclang; exactly one of --sigs or
--header must be given.

Usage (HIP):
    python3 lttng_curated_codegen.py \\
        --provider hip \\
        --yaml       projects/clr/hipamd/scripts/curated_apis.yaml \\
        --header     projects/hip/include/hip/hip_runtime_api.h \\
        --extra-arg=-D__HIP_PLATFORM_AMD__=1 \\
        --tp-out     projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h \\
        --emit-out   projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h

Usage (HSA):
    python3 lttng_curated_codegen.py \\
        --provider hsa \\
        --yaml       projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml \\
        --header     projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h \\
        --header     projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h \\
        --header     projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_api_trace.h \\
        --tp-out     projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h \\
        --emit-out   projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h

CI / pre-commit usage — verify the checked-in headers are still exactly
what the generator produces from the checked-in YAML + the real headers:

    python3 lttng_curated_codegen.py --provider hip --check \\
        --yaml ... --header ... [--extra-arg ...] --tp-out ... --emit-out ...

`--check` generates in memory, diffs the result against the on-disk
`--tp-out` / `--emit-out` files, prints a unified diff on mismatch, and
exits 1. It never overwrites the target files.

`--dump-resolved <path>` writes the live-resolved signature sidecar JSON
({api: [{name, c_type}, ...]}) to `path` and exits, without generating
tp.h/emit.h. This is the mechanism other consumers (e.g. the HIP curated
coverage test harness, which needs each OUT arg's real C type to declare
correctly-typed scratch slots) use instead of reading a checked-in JSON
cache. Requires --yaml and --header (not --sigs).
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
sys.path.insert(0, HERE)
from lttng_curated_lib import (parse_yaml_file, PAYLOAD_BUDGET,
                               validate_api, ParseError, BudgetError,
                               TYPE_EXPANSION, DIR_EXPANSION)
from lttng_curated_verify import (parse_declarations, compute_sidecar,
                                   expand_compact_apis,
                                   resolve_curated_return_kind,
                                  AmbiguousDeclarationError,
                                  AmbiguousInferenceError,
                                  UnsupportedReturnTypeError)

# ---------------------------------------------------------------------------
# Combined-event return-field info (schema v1). Each curated API now emits ONE
# combined `<api>` event fired twice per call with a `phase` discriminator; the
# event carries a per-API-return-typed field on its LAST chunk (unless VOID).
#
#   phase discriminator: ENTER=0, EXIT=1 (int32).
#   return field by curated return kind (see
#   lttng_curated_verify.resolve_curated_return_kind):
#     STATUS -> retstatus int32   (lttng_ust_field_integer)
#     PTR    -> retptr    uint64  (lttng_ust_field_integer_hex)
#     U64    -> retval    uint64  (lttng_ust_field_integer)
#     I64    -> retval    int64   (lttng_ust_field_integer)
#     U32    -> retval    uint32  (lttng_ust_field_integer)
#     VOID   -> (no return field)
# ---------------------------------------------------------------------------
PHASE_ENTER = 0
PHASE_EXIT = 1

# (field_name, tp/emit C type, lttng field macro, emit-helper param type).
#
# STATUS/PTR/VOID/U64/I64/U32 are the return kinds named in the schema-v1
# design. Two more kinds surface from the real full inventory via
# lttng_curated_verify.classify_curated_return_kind() and must be handled too:
#   - I32:    HIP's one plain-`int`-returning curated API
#             (hipGetStreamDeviceId). Captured as a signed 32-bit `retval`.
#   - STRUCT: a by-value struct return (HIP's hipCreateChannelDesc). There is
#             no meaningful scalar return field for a struct, so it is treated
#             like VOID for the combined-event return field (no return field,
#             exit helper takes no return parameter) — matching how v4 gave
#             STRUCT wrappers a direct _args emission with no typed exit event.
RETURN_FIELD_INFO = {
    'STATUS': ('retstatus', 'int32_t',  'lttng_ust_field_integer',     'int32_t'),
    'PTR':    ('retptr',    'uint64_t', 'lttng_ust_field_integer_hex', 'uint64_t'),
    'U64':    ('retval',    'uint64_t', 'lttng_ust_field_integer',     'uint64_t'),
    'I64':    ('retval',    'int64_t',  'lttng_ust_field_integer',     'int64_t'),
    'U32':    ('retval',    'uint32_t', 'lttng_ust_field_integer',     'uint32_t'),
    'I32':    ('retval',    'int32_t',  'lttng_ust_field_integer',     'int32_t'),
    'VOID':   None,
    'STRUCT': None,
}

# Repo root, used to locate the clang-format style file for HIP's emit.h
# post-process pass. shared/lttng/scripts -> shared/lttng -> shared -> root.
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

# ---------------------------------------------------------------------------
# Per-DSL-type emit-side info: (lttng field macro, emit-side C type used for
# both the TP_ARGS/field declaration AND the cast target, cast expression
# template applied to the raw arg name at the do_tracepoint() call site).
# ---------------------------------------------------------------------------
TYPE_INFO = {
    'handle':     ('lttng_ust_field_integer_hex', 'uint64_t', '(uint64_t)(uintptr_t)({arg})'),
    'ptr':        ('lttng_ust_field_integer_hex', 'uint64_t', '(uint64_t)(uintptr_t)({arg})'),
    'device_ptr': ('lttng_ust_field_integer_hex', 'uint64_t', '(uint64_t)({arg})'),
    'size':       ('lttng_ust_field_integer',     'uint64_t', '(uint64_t)({arg})'),
    'int32':      ('lttng_ust_field_integer',     'int32_t',  '(int32_t)({arg})'),
    'uint32':     ('lttng_ust_field_integer',     'uint32_t', '(uint32_t)({arg})'),
    'int64':      ('lttng_ust_field_integer',     'int64_t',  '(int64_t)({arg})'),
    'uint64':     ('lttng_ust_field_integer',     'uint64_t', '(uint64_t)({arg})'),
    'float':      ('lttng_ust_field_float',       'float',    '(float)({arg})'),
    'enum':       ('lttng_ust_field_integer',     'int32_t',  '(int32_t)({arg})'),
    # bool canonicalizes to a 0/1 uint32_t so the wire representation is
    # storage-rep-independent (spec: C bool's in-memory size/rep varies).
    'bool':       ('lttng_ust_field_integer',     'uint32_t', '(uint32_t)(!!({arg}))'),
    'cstring':    ('lttng_ust_field_string',      'const char*', '({arg} ? {arg} : "")'),
    # dim3 / dim3_packed are expanded specially in emit_tp_event()/emit_helper().
}

# Helper formal-parameter C type for each DSL type (the type the emit
# helper's IN-direction parameter is declared with; OUT direction uses a
# pointer-to-this-type, or a sidecar-derived real type for handle/ptr/
# device_ptr — see out_helper_emit()).
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
    'bool':        'int',        # C bool promotes to int at the call boundary
    'cstring':     'const char*',
    'dim3':        'dim3',
    'dim3_packed': 'dim3',       # helper takes dim3, packs internally
}


def out_helper_emit(arg, real_c_type=None):
    """Return (helper_param_type, deref_expr_template) for an OUT arg.

    Numeric/scalar OUT types get a plain T* helper param and a `*p` deref.
    Pointer/handle/device_ptr OUT types need the real, provider-resolved
    C type from the verifier's signature sidecar because the correct deref
    shape differs by provider:
      - HIP: `hipStream_t` is a typedef'd pointer -> `hipStream_t*` param,
        deref is `*p` (one level).
      - HSA: `hsa_signal_t` is `{ uint64_t handle; }` -> `hsa_signal_t*`
        param, deref must read `p->handle`, NOT `*p` (which would
        reinterpret the struct's raw bytes).
      - HSA: `hsa_queue_t**` (pointer-to-pointer) -> deref is `*p` (the
        pointee IS the handle).
    `real_c_type` is None only when no sidecar entry exists for this arg;
    callers should always have one in Phase 1 (sigs is a required input).
    """
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
        # No sidecar entry — fall back to the HIP-style shape.
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
    tp_provider: str        # LTTng provider symbol: 'rocm_hip' / 'rocm_hsa'
    status_type: str        # e.g. 'hipError_t' / 'hsa_status_t'
    status_success: str     # e.g. 'hipSuccess' / 'HSA_STATUS_SUCCESS'
    emit_includes: str      # literal include block for emit.h (see below)
    clang_format_emit: bool # whether to run clang-format on emit.h output


HIP_EMIT_INCLUDES = """\
/* Force the AMD platform define so the host-only HIP runtime header is
 * self-contained. rocclr internal TUs that pull in rocm_trace_emit.h
 * (e.g. device/rocm/rocvirtual.cpp) don't set this themselves. This
 * file is built only into libamdhip64; there is no NVIDIA path. */
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>
"""

HSA_EMIT_INCLUDES = """\
/* HSA headers: the rocr-runtime source layout has these flat in
 * runtime/hsa-runtime/inc/ (not under an hsa/ subdir), and that directory
 * is on the include path while building libhsa-runtime64. The
 * angle-bracket <hsa/hsa.h> form is what consumers see after install;
 * this internal tracepoint emit header is only included from within
 * rocr-runtime's own compile units, so use the quoted form so the
 * in-tree build works. */
#include "hsa.h"
#include "hsa_ext_amd.h"
"""

PROVIDERS = {
    'hip': ProviderConfig(
        key='hip', tp_provider='rocm_hip',
        status_type='hipError_t', status_success='hipSuccess',
        emit_includes=HIP_EMIT_INCLUDES, clang_format_emit=True,
    ),
    'hsa': ProviderConfig(
        key='hsa', tp_provider='rocm_hsa',
        status_type='hsa_status_t', status_success='HSA_STATUS_SUCCESS',
        emit_includes=HSA_EMIT_INCLUDES, clang_format_emit=False,
    ),
}


# ---------------------------------------------------------------------------
# Codegen: tp.h
# ---------------------------------------------------------------------------
def event_name(api_name, chunk_index):
    """Return the generated event name for one ordered API argument chunk.

    Schema v1: the combined event is named exactly `<api>` (`<api>_N` for
    later chunks) — the historical `_args` suffix is dropped."""
    suffix = '' if chunk_index == 0 else f'_{chunk_index + 1}'
    return f'{api_name}{suffix}'


def _arg_field_count(arg):
    """Payload field count one resolved arg expands to (type * direction)."""
    return TYPE_EXPANSION.get(arg['type'], 1) * DIR_EXPANSION[arg['dir']]


def event_field_chunks_v5(api, return_kind):
    """Partition a combined-event API's args into ordered chunks.

    Every chunk reserves one field for the leading `phase` discriminator;
    the LAST chunk additionally reserves one field for the return value
    (unless the API is VOID-returning). Both reservations count against
    the per-event PAYLOAD_BUDGET, so the arg budget per chunk shrinks
    relative to the generic (pre-curation) layout. Arguments are kept intact (an unpacked dim3's
    three fields never straddle a chunk boundary); a new chunk starts only
    between arguments.

    Because the return field lands on whichever chunk turns out to be
    last, this is a two-pass partition: first pack args reserving only
    `phase` per chunk, then re-check whether the final chunk still fits
    once the return field is added and, if not, spill the return field
    into a fresh trailing chunk (carrying `phase` + the return field
    only). This mirrors lttng_curated_lib.event_field_chunks() but is
    v5-aware.
    """
    has_return = RETURN_FIELD_INFO.get(return_kind) is not None
    phase_reserve = 1
    chunks = []
    current = []
    current_fields = phase_reserve
    for arg in api['args']:
        fields = _arg_field_count(arg)
        if fields + phase_reserve > PAYLOAD_BUDGET:
            raise BudgetError(
                f"{api['api']} arg {arg['name']}: expands to {fields} fields, "
                f"exceeds the per-event budget of {PAYLOAD_BUDGET} once the "
                f"phase discriminator is reserved")
        if current and current_fields + fields > PAYLOAD_BUDGET:
            chunks.append(current)
            current = []
            current_fields = phase_reserve
        current.append(arg)
        current_fields += fields
    chunks.append(current)  # may be [] for a zero-arg api
    # Spill the return field into a fresh trailing chunk if the last arg
    # chunk is already full.
    if has_return and current_fields + 1 > PAYLOAD_BUDGET:
        chunks.append([])
    return chunks


def _return_field_lines(return_kind):
    """Return (tp_args_tokens, tp_field_lines) for the return field, or
    ([], []) for a VOID (no-return) API."""
    info = RETURN_FIELD_INFO.get(return_kind)
    if info is None:
        return [], []
    field_name, c_type, field_macro, _param = info
    return ([c_type, field_name],
            [f'{field_macro}({c_type}, {field_name}, {field_name})'])


def emit_tp_event(tp_provider, api_name, args, chunk_index, return_kind,
                  is_last_chunk):
    """Emit one LTTNG_UST_TRACEPOINT_EVENT block for a combined-event chunk.

    Every chunk carries a leading `phase` (int32) field; the return-value
    field is emitted only on the last chunk (unless VOID)."""
    name = event_name(api_name, chunk_index)

    tp_args = ['int32_t', 'phase']
    tp_fields = ['lttng_ust_field_integer(int32_t, phase, phase)']

    for a in args:
        nm = a['name']
        ty = a['type']
        if ty == 'dim3':
            for axis in ('x', 'y', 'z'):
                tp_args += ['uint32_t', f'{nm}_{axis}']
                tp_fields.append(
                    f'lttng_ust_field_integer(uint32_t, {nm}_{axis}, {nm}_{axis})')
        elif ty == 'dim3_packed':
            tp_args += ['uint64_t', nm]
            tp_fields.append(f'lttng_ust_field_integer_hex(uint64_t, {nm}, {nm})')
        else:
            field_macro, c_type, _ = TYPE_INFO[ty]
            tp_args += [c_type, nm]
            if field_macro == 'lttng_ust_field_string':
                tp_fields.append(f'{field_macro}({nm}, {nm})')
            elif field_macro == 'lttng_ust_field_float':
                tp_fields.append(f'{field_macro}(float, {nm}, {nm})')
            else:
                tp_fields.append(f'{field_macro}({c_type}, {nm}, {nm})')

    if is_last_chunk:
        ret_args, ret_fields = _return_field_lines(return_kind)
        tp_args += ret_args
        tp_fields += ret_fields

    args_str = ', '.join(tp_args)

    # Field lines are emitted flush-left (column 0) inside
    # LTTNG_UST_TP_FIELDS(...) — matches the established layout of the
    # checked-in headers. Built via explicit line-joining (not
    # textwrap.dedent on a pre-substituted multi-line value) so every
    # field line lands at the same column regardless of its position in
    # the list; dedent's "first line gets the template's prefix, later
    # lines keep only their own literal prefix" behavior is NOT what we
    # want here.
    lines = [
        "LTTNG_UST_TRACEPOINT_EVENT(",
        f"    {tp_provider}, {name},",
        f"    LTTNG_UST_TP_ARGS({args_str}),",
        "    LTTNG_UST_TP_FIELDS(",
    ]
    lines.extend(tp_fields)
    lines.append("    )")
    lines.append(")")
    return "\n".join(lines) + "\n"


def _tp_header_comment():
    """Schema note emitted at the top of the event section (replaces the
    old shared-lifecycle event block, which schema v1 removes)."""
    return """/* Schema v1: each curated API is ONE combined event `<api>` (`<api>_N`
 * for high-arity chunk overflow), fired TWICE per call. The leading
 * `phase` field discriminates the record: 0 = ENTER (IN args populated,
 * OUT args and return value 0), 1 = EXIT (IN args 0, OUT args populated,
 * return value populated). Enabling the single event name turns on BOTH
 * records at the LTTng level. Per-event identity (vpid, vtid, timestamp)
 * comes from LTTng-UST channel contexts; consumers pair ENTER/EXIT by
 * (vpid, vtid, timestamp, event-name). */
"""


def emit_tp_h(cfg, apis, banner, return_kinds):
    """Assemble the full tp.h. Every chunk in `out` ends with EXACTLY one
    trailing '\\n'; joining with an extra '\\n' separator turns that into
    a single blank line between chunks (and between events), matching the
    established layout, without accidentally doubling up when a chunk's
    own template already included a trailing blank line."""
    guard = f"_{cfg.tp_provider.upper()}_CURATED_TP_H"
    # Banner and include-guard open are adjacent (no blank line between the
    # closing `*/` and `#if`) — standard convention, so fold them into one
    # chunk rather than letting the blank-line-separator rule apply here.
    out = [banner + f"#if !defined({guard}) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)\n"
                     f"#define {guard}\n"]
    needs_dim3_pack = any(a['type'] == 'dim3_packed' for api in apis for a in api['args'])
    if needs_dim3_pack:
        out.append('/* dim3_packed encoding is defined in rocm_dim3_pack.h, included by\n'
                    ' * the emit-helper header that includes us transitively. */\n')
    out.append(_tp_header_comment())
    for api in apis:
        return_kind = return_kinds[api['api']]
        chunks = event_field_chunks_v5(api, return_kind)
        for chunk_index, args in enumerate(chunks):
            out.append(emit_tp_event(cfg.tp_provider, api['api'], args, chunk_index,
                                     return_kind, chunk_index == len(chunks) - 1))
    # Historical layout has one extra blank line before the closing #endif
    # (present in every checked-in revision of this file); reproduced here.
    out.append(f"\n#endif /* {guard} */\n")
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Codegen: emit.h (schema v1 — combined-event enter/exit helper pair)
# ---------------------------------------------------------------------------
def _zero_exprs_for_arg(arg):
    """Field-count-matched zero expression(s) for an arg written as 0 in the
    phase where its direction is not being captured (IN fields at EXIT, OUT
    fields at ENTER)."""
    return ['0'] * _arg_field_count(arg)


def _in_field_exprs(arg):
    """Populated field expression(s) for an IN arg (helper param `<name>`)."""
    nm, ty = arg['name'], arg['type']
    if ty == 'dim3':
        return [f"(uint32_t){nm}.x", f"(uint32_t){nm}.y", f"(uint32_t){nm}.z"]
    if ty == 'dim3_packed':
        return [f"{nm}_packed"]
    _, _, cast_tmpl = TYPE_INFO[ty]
    return [cast_tmpl.format(arg=nm)]


def _emit_do_tracepoint_calls(cfg, name, chunks, return_kind, phase_value,
                              field_exprs_by_arg, ret_expr):
    """Render the per-chunk do_tracepoint body shared by the enter and exit
    helpers. `field_exprs_by_arg` maps an arg name to its already-
    materialized field-value expression list for this phase; `ret_expr` is
    the return field's value expression (or None).

    For a single-chunk API the enclosing helper already guards on the sole
    event's enabled-state, so the call is emitted unconditionally. Multi-
    chunk APIs re-check each chunk's own event so only enabled chunks fire.
    """
    single = len(chunks) == 1
    calls = []
    last_index = len(chunks) - 1
    for index, chunk in enumerate(chunks):
        event = event_name(name, index)
        chunk_exprs = [f"(int32_t){phase_value}"]
        for arg in chunk:
            chunk_exprs.extend(field_exprs_by_arg[arg['name']])
        if index == last_index and ret_expr is not None:
            chunk_exprs.append(ret_expr)
        if single:
            do_tp_args = ',\n                '.join(chunk_exprs)
            do_tp_call = f'{cfg.tp_provider}, {event},\n                {do_tp_args}'
            calls.append(textwrap.dedent(f"""\
                    lttng_ust_do_tracepoint({do_tp_call});""").rstrip())
        else:
            do_tp_args = ',\n                    '.join(chunk_exprs)
            do_tp_call = f'{cfg.tp_provider}, {event},\n                    {do_tp_args}'
            calls.append(textwrap.dedent(f"""\
                    if (lttng_ust_tracepoint_enabled({cfg.tp_provider}, {event})) {{
                        lttng_ust_do_tracepoint({do_tp_call});
                    }}""").rstrip())
    return '\n'.join(calls)


def _enabled_expr(cfg, name, num_chunks):
    return ' || '.join(
        f'lttng_ust_tracepoint_enabled({cfg.tp_provider}, {event_name(name, index)})'
        for index in range(num_chunks))


def emit_enter_helper(cfg, api, return_kind):
    """Emit the active-mode `rocm_trace_emit_<api>_enter(<IN params>)` helper.

    Fires the combined event(s) at phase=ENTER: IN args populated, OUT arg
    fields written as 0, return field (if any) written as 0."""
    name = api['api']
    args = api['args']
    chunks = event_field_chunks_v5(api, return_kind)

    formal_params = []
    setups = []
    field_exprs_by_arg = {}
    for a in args:
        nm, ty, dr = a['name'], a['type'], a['dir']
        if dr == 'IN':
            formal_params.append(f"{HELPER_PARAM_TYPE[ty]} {nm}")
            if ty == 'dim3_packed':
                setups.append(f"        const uint64_t {nm}_packed = ROCM_DIM3_PACK({nm});")
            field_exprs_by_arg[nm] = _in_field_exprs(a)
        else:
            field_exprs_by_arg[nm] = _zero_exprs_for_arg(a)

    ret_expr = '0' if RETURN_FIELD_INFO.get(return_kind) is not None else None
    body_inner = _emit_do_tracepoint_calls(
        cfg, name, chunks, return_kind, PHASE_ENTER, field_exprs_by_arg, ret_expr)
    if setups:
        body_inner = '\n'.join(setups) + '\n' + body_inner
    formal_str = ',\n    '.join(formal_params) if formal_params else 'void'
    return textwrap.dedent(f"""\
        void rocm_trace_emit_{name}_enter(
            {formal_str}) {{
            if (rocm_trace_disabled()) return;
            if ({_enabled_expr(cfg, name, len(chunks))}) {{
        {body_inner}
            }}
        }}
        """)


def emit_exit_helper(cfg, api, return_kind, sigs=None):
    """Emit the active-mode combined-event EXIT helper.

    Signature (schema v1):
      - STATUS: rocm_trace_emit_<api>_exit(<OUT ptr params...>, <status_type>
        status). `status` is both the return field (retstatus) and the
        success gate for OUT-param deref. STATUS is the only kind that ever
        has OUT args in the curated inventory.
      - PTR/U64/I64/U32/I32: rocm_trace_emit_<api>_exit(<retval-type> retval)
        — no OUT args exist for these kinds, so no success gate is needed.
      - VOID/STRUCT: rocm_trace_emit_<api>_exit(void) — no return field.

    Fires the combined event(s) at phase=EXIT: IN arg fields written as 0,
    OUT args populated (success-gated deref, matching v4 semantics), return
    field populated from the trailing param."""
    name = api['api']
    args = api['args']
    chunks = event_field_chunks_v5(api, return_kind)
    sig_by_name = {s['name']: s['c_type'] for s in sigs} if sigs else {}
    ret_info = RETURN_FIELD_INFO.get(return_kind)
    has_return = ret_info is not None
    has_out = any(a['dir'] == 'OUT' for a in args)
    # OUT args only ever occur on STATUS-returning APIs in the curated
    # inventory, and their success gate is the hipError_t/hsa_status_t
    # `status` param. Guard against a future non-STATUS OUT API landing here.
    if has_out and return_kind != 'STATUS':
        raise SystemExit(
            f"{name}: OUT args on a {return_kind}-returning API are not "
            f"supported by the schema-v1 exit helper (no success status "
            f"available to gate the OUT deref)")

    formal_params = []
    setups = []
    field_exprs_by_arg = {}

    for a in args:
        nm, ty, dr = a['name'], a['type'], a['dir']
        if dr == 'OUT':
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
                field_exprs_by_arg[nm] = [f"{nm}_val"]
            else:
                setups.append(textwrap.dedent(f"""\
                            const auto {nm}_val =
                                ({gate}{nm}_out_ptr != NULL)
                                    ? *{nm}_out_ptr : 0;""").rstrip())
                _, _, cast_tmpl = TYPE_INFO[ty]
                field_exprs_by_arg[nm] = [cast_tmpl.format(arg=f"{nm}_val")]
        else:
            field_exprs_by_arg[nm] = _zero_exprs_for_arg(a)

    ret_expr = None
    if has_return:
        field_name, _ctype, _macro, param_type = ret_info
        if return_kind == 'STATUS':
            # STATUS is also the OUT success gate; the trailing param is the
            # provider status type, named `status`.
            formal_params.append(f"{cfg.status_type} status")
            ret_expr = "(int32_t)status"
        else:
            formal_params.append(f"{param_type} {field_name}")
            ret_expr = field_name

    body_inner = _emit_do_tracepoint_calls(
        cfg, name, chunks, return_kind, PHASE_EXIT, field_exprs_by_arg, ret_expr)
    if setups:
        body_inner = '\n'.join(setups) + '\n' + body_inner
    formal_str = ',\n    '.join(formal_params) if formal_params else 'void'
    return textwrap.dedent(f"""\
        void rocm_trace_emit_{name}_exit(
            {formal_str}) {{
            if (rocm_trace_disabled()) return;
            if ({_enabled_expr(cfg, name, len(chunks))}) {{
        {body_inner}
            }}
        }}
        """)


def _enter_formals(cfg, api):
    """Bare (unnamed) formal-parameter type list for the no-op enter helper."""
    formals = []
    for a in api['args']:
        if a['dir'] == 'IN':
            if a['type'] in ('dim3', 'dim3_packed'):
                formals.append('dim3')
            else:
                formals.append(HELPER_PARAM_TYPE[a['type']])
    return formals


def _exit_formals(cfg, api, return_kind, sigs=None):
    """Bare (unnamed) formal-parameter type list for the no-op exit helper.
    Must byte-match the active-mode exit helper's parameter types."""
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
            formals.append(ret_info[3])  # param_type
    return formals


def emit_noop_enter_helper(cfg, api):
    formals = _enter_formals(cfg, api)
    formal_str = ', '.join(formals)
    return f"void rocm_trace_emit_{api['api']}_enter({formal_str}) {{}}\n"


def emit_noop_exit_helper(cfg, api, return_kind, sigs=None):
    formals = _exit_formals(cfg, api, return_kind, sigs=sigs)
    formal_str = ', '.join(formals)
    return f"void rocm_trace_emit_{api['api']}_exit({formal_str}) {{}}\n"


def emit_decl_enter_helper(cfg, api):
    """Ordinary (non-static, non-inline) forward declaration for the enter
    helper, emitted into the thin declarations-only header. The parameter
    types match the active-mode helper's named formals exactly (declarations
    carry types only, so they are valid for both the real-body and no-op
    definitions in the .cpp)."""
    formals = _enter_formals(cfg, api)
    formal_str = ', '.join(formals) if formals else 'void'
    return f"void rocm_trace_emit_{api['api']}_enter({formal_str});\n"


def emit_decl_exit_helper(cfg, api, return_kind, sigs=None):
    """Ordinary forward declaration for the exit helper (see
    emit_decl_enter_helper)."""
    formals = _exit_formals(cfg, api, return_kind, sigs=sigs)
    formal_str = ', '.join(formals) if formals else 'void'
    return f"void rocm_trace_emit_{api['api']}_exit({formal_str});\n"


def emit_emit_h(cfg, apis, banner, return_kinds, sigs_by_api=None):
    """The thin declarations-only emit header.

    Every curated API's enter/exit helper is an ordinary (non-static,
    non-inline) forward declaration. The definitions — real bodies gated
    on the enable macro, no-op stubs otherwise — live out-of-line in the
    companion rocm_trace_emit_curated.cpp (see emit_emit_cpp). Call sites
    include only this header and call the helpers unconditionally; a body
    is always linked in from the .cpp in both LTTng-enabled and disabled
    builds."""
    macro_guard = f"ROCM_{cfg.key.upper()}_TRACE_EMIT_CURATED_H_"

    out = [banner]
    out.append(f"#ifndef {macro_guard}\n#define {macro_guard}\n\n"
               f"#include <stdint.h>\n#include <stddef.h>\n")
    out.append(cfg.emit_includes)

    needs_dim3 = any(a['type'] in ('dim3', 'dim3_packed') for api in apis for a in api['args'])
    if needs_dim3:
        out.append('#include "rocm_dim3_pack.h"\n')

    out.append('\n')
    for api in apis:
        sigs = (sigs_by_api or {}).get(api['api'])
        return_kind = return_kinds[api['api']]
        out.append(emit_decl_enter_helper(cfg, api))
        out.append(emit_decl_exit_helper(cfg, api, return_kind, sigs=sigs))

    out.append(f"\n#endif  /* {macro_guard} */\n")
    return ''.join(out)


def emit_emit_cpp(cfg, apis, banner, return_kinds, emit_header_basename,
                  sigs_by_api=None):
    """The out-of-line definitions TU for the curated emit helpers.

    Carries the same #if enable-macro / #else structure the header used to,
    but with ordinary (non-inline) function definitions inside each branch:
    the real tracepoint bodies when the enable macro is set, no-op stubs
    otherwise. Compiled unconditionally (self-guarded) so the no-op stubs
    are always available to link against in an LTTng-disabled build."""
    enable_macro = 'HIP_ENABLE_LTTNG_UST' if cfg.key == 'hip' else 'HSA_ENABLE_LTTNG_UST'

    out = [banner]
    out.append(f'#include "{emit_header_basename}"\n')

    out.append(f"\n#if defined({enable_macro}) && {enable_macro}\n\n"
               f"#include <atomic>\n#include \"{cfg.tp_provider}_tp.h\"\n")

    out.append(f"""
extern std::atomic<bool> {cfg.tp_provider}_trace_g_disabled;
#ifndef ROCM_TRACE_DISABLED_DEFINED
#define ROCM_TRACE_DISABLED_DEFINED
static inline bool rocm_trace_disabled(void) {{
    return {cfg.tp_provider}_trace_g_disabled.load(std::memory_order_relaxed);
}}
#endif

""")

    for api in apis:
        sigs = (sigs_by_api or {}).get(api['api'])
        return_kind = return_kinds[api['api']]
        out.append(emit_enter_helper(cfg, api, return_kind))
        out.append('\n')
        out.append(emit_exit_helper(cfg, api, return_kind, sigs=sigs))
        out.append('\n')

    out.append(f"\n#else  /* {enable_macro} not defined — all helpers are no-ops */\n\n")
    for api in apis:
        sigs = (sigs_by_api or {}).get(api['api'])
        return_kind = return_kinds[api['api']]
        out.append(emit_noop_enter_helper(cfg, api))
        out.append(emit_noop_exit_helper(cfg, api, return_kind, sigs=sigs))

    out.append(f"\n#endif  /* {enable_macro} */\n")
    return ''.join(out)


# ---------------------------------------------------------------------------
# Banners
# ---------------------------------------------------------------------------
def _regen_cmd(cfg, yaml_path, tp_out, emit_out, emit_cpp_out=None,
                sigs_path=None,
                header_paths=None, source_paths=None, extra_args=None):
    """Render the exact regeneration command with paths relative to the
    repo root. Normalizing here (rather than echoing back whatever the
    caller passed verbatim) keeps the embedded banner command stable
    across absolute-path vs. relative-path invocations of the generator
    itself — otherwise --check would report spurious "drift" any time
    the tool was re-run from a different working directory.

    Exactly one signature source is rendered: `sigs_path` (file-based,
    mainly for tests) XOR `header_paths`/`extra_args` (live libclang
    parse — the normal, production path)."""
    def rel(p):
        try:
            return os.path.relpath(os.path.abspath(p), REPO_ROOT)
        except ValueError:
            return p
    def rel_extra_arg(e):
        # Normalize `-I<path>` args the same way header/yaml/sigs paths
        # are normalized, so the displayed banner stays stable regardless
        # of whether the caller passed an absolute or relative -I path
        # (only -I is a path-shaped clang flag among the ones this
        # project uses; other flags like -D... pass through unchanged).
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
    parts.append(f"--tp-out {rel(tp_out)}")
    parts.append(f"--emit-out {rel(emit_out)}")
    if emit_cpp_out is not None:
        parts.append(f"--emit-cpp-out {rel(emit_cpp_out)}")
    lines = ["python3 shared/lttng/scripts/lttng_curated_codegen.py \\"]
    for i, p in enumerate(parts):
        lines.append(f"    {p}" + (" \\" if i < len(parts) - 1 else ""))
    return lines


def _comment_block(lines):
    """Render `lines` (a list of already-unindented text lines, '' for a
    blank comment line) as a `/* ... */` C comment block, one line at a
    time — no textwrap.dedent (multi-line f-string substitution silently
    breaks its indentation math; see emit_tp_event's comment)."""
    out = ["/* " + lines[0]]
    for l in lines[1:]:
        out.append(" * " + l if l else " *")
    out.append(" */")
    return "\n".join(out) + "\n"


def tp_banner(cfg, apis, yaml_path, sha256, regen_cmd):
    return _comment_block([
        f"AUTO-GENERATED by lttng_curated_codegen.py from {os.path.basename(yaml_path)}.",
        "Do not edit by hand — regenerate instead (see command below).",
        "",
        f"SHA256({os.path.basename(yaml_path)}) at generation: {sha256}",
        f"Provider: {cfg.tp_provider}",
        f"API count: {len(apis)}",
        "",
        "Regenerate with:",
        *(f"  {l}" for l in regen_cmd),
    ])


def emit_banner(cfg, yaml_path, sha256, regen_cmd, role_lines):
    return _comment_block([
        f"AUTO-GENERATED by lttng_curated_codegen.py from {os.path.basename(yaml_path)}.",
        "Do not edit by hand — regenerate instead (see command below).",
        "",
        f"SHA256({os.path.basename(yaml_path)}) at generation: {sha256}",
        "",
        *role_lines,
        "",
        "Regenerate with:",
        *(f"  {l}" for l in regen_cmd),
    ])


_EMIT_H_ROLE_LINES = [
    "Per-API typed emit-helper DECLARATIONS for curated parameter capture.",
    "The out-of-line definitions (real tracepoint bodies when LTTng is",
    "enabled, no-op stubs otherwise) live in the companion",
    "rocm_trace_emit_curated.cpp. Every helper takes (<captured-args...>,",
    "<status_type> status); status is the call's success result, used to",
    "gate OUT-param deref. All-IN APIs accept it but mark it unused.",
]

_EMIT_CPP_ROLE_LINES = [
    "Out-of-line definitions for the per-API typed emit helpers declared in",
    "rocm_trace_emit_curated.h. Both the real tracepoint bodies (enable",
    "macro set) and the no-op stubs (enable macro unset) are compiled here,",
    "chosen by the same #if as before; the header is now declarations only.",
]


# ---------------------------------------------------------------------------
# clang-format post-process (HIP emit.h only — see ProviderConfig.clang_format_emit)
# ---------------------------------------------------------------------------
def _clang_format(text, style_file):
    clang_format = shutil.which('clang-format')
    if not clang_format:
        print("WARN: clang-format not found on PATH; leaving emit.h "
              "un-formatted (raw codegen output).", file=sys.stderr)
        return text
    r = subprocess.run([clang_format, f'--style=file:{style_file}'],
                        input=text, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"WARN: clang-format failed (rc={r.returncode}): {r.stderr}\n"
              f"Leaving emit.h un-formatted.", file=sys.stderr)
        return text
    return r.stdout


def load_sigs_by_api(yaml_path, sigs_path, header_paths, source_paths, extra_args):
    """Resolve the {api: [{name, c_type}, ...]} signature map from
    exactly one source: a pre-computed JSON file (`sigs_path`, mainly
    for tests) or a live libclang parse of `header_paths` (production
    path — reuses lttng_curated_verify.py's parse_headers() +
    compute_sidecar(), the one place duplicate-declaration resolution
    happens)."""
    if sigs_path is not None:
        with open(sigs_path) as f:
            raw = json.load(f)
        # A sigs entry may be either the plain [{name, c_type}, ...] list or
        # a {"params": [...], "return_kind": "X"} mapping (so libclang-
        # independent test fixtures can also declare a per-API return kind
        # for the v5 combined-event return field). Normalize to the list form
        # here; return-kind extraction happens in resolve_return_kinds().
        return {api: (entry['params'] if isinstance(entry, dict) else entry)
                for api, entry in raw.items()}
    apis = parse_yaml_file(yaml_path)
    header_decls = parse_declarations(header_paths, source_paths, extra_args)
    return compute_sidecar(apis, header_decls)


def resolve_return_kinds(cfg, apis, sigs_path, header_paths, source_paths, extra_args):
    """Resolve every API's curated return kind (STATUS/PTR/VOID/U64/I64/U32
    — see lttng_curated_verify.resolve_curated_return_kind), the v5
    combined-event return-field discriminator.

    Live-header mode uses the real libclang-resolved result type (the same
    path the migrator and coverage return-kinds gate use). `--sigs` mode
    (test fixtures) reads an optional `return_kind` from the sigs JSON
    entry, defaulting to STATUS."""
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
    """Resolve curated_apis.yaml into the fully-expanded, budget-checked
    {api, args: [{name, type, dir}, ...]} representation the rest of
    codegen (emit_tp_h/emit_emit_h) operates on.

    Compact-schema entries (real production YAML) need real per-arg C
    types to infer DSL types from — that means a live libclang --header
    parse (see lttng_curated_verify.expand_compact_apis()). `--sigs`
    mode (a pre-computed {api: [{name, c_type}]} JSON, with no
    canonical-type/is_enum info) can only be used with the fully-explicit
    {name, type, dir} YAML shape (test fixtures), since it lacks what
    inference needs; a compact-schema entry combined with --sigs is a
    clear usage error, not silently-wrong output."""
    raw_apis = parse_yaml_file(yaml_path)
    if sigs_path is not None:
        # A zero-arg api (e.g. hipDeviceSynchronize) needs no type
        # resolution at all regardless of which internal shape
        # parse_yaml_file() happened to route it through, so only
        # non-empty compact-shape apis (args present, but no 'type' key)
        # are actually a --sigs-mode usage error.
        bad = [a['api'] for a in raw_apis if a['args'] and 'type' not in a['args'][0]]
        if bad:
            sys.exit(
                f"ERROR: --sigs mode requires the fully-explicit "
                f"{{name, type, dir}} YAML shape (no live header available "
                f"to infer the compact schema's types from); compact-shape "
                f"api(s) found: {bad}")
        return raw_apis
    header_decls = parse_declarations(header_paths, source_paths, extra_args)
    return expand_compact_apis(raw_apis, header_decls)


# ---------------------------------------------------------------------------
# Top-level generation entry point (shared by normal mode and --check mode)
# ---------------------------------------------------------------------------
def generate(cfg, apis, yaml_path, tp_out_path, emit_out_path, sigs_by_api,
             return_kinds, regen_cmd):
    """Return (tp_text, emit_text, emit_cpp_text). `apis` must already be
    the fully-expanded/resolved representation (see resolve_apis())."""
    with open(yaml_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    emit_header_basename = os.path.basename(emit_out_path)

    tp_text = emit_tp_h(cfg, apis, tp_banner(cfg, apis, yaml_path, sha256, regen_cmd),
                        return_kinds)
    emit_text = emit_emit_h(
        cfg, apis,
        emit_banner(cfg, yaml_path, sha256, regen_cmd, _EMIT_H_ROLE_LINES),
        return_kinds, sigs_by_api=sigs_by_api)
    emit_cpp_text = emit_emit_cpp(
        cfg, apis,
        emit_banner(cfg, yaml_path, sha256, regen_cmd, _EMIT_CPP_ROLE_LINES),
        return_kinds, emit_header_basename, sigs_by_api=sigs_by_api)

    if cfg.clang_format_emit:
        style_file = os.path.join(REPO_ROOT, 'projects', 'clr', '.clang-format')
        emit_text = _clang_format(emit_text, style_file)
        emit_cpp_text = _clang_format(emit_cpp_text, style_file)

    return tp_text, emit_text, emit_cpp_text


def _diff(a_text, b_text, a_name, b_name):
    return ''.join(difflib.unified_diff(
        a_text.splitlines(keepends=True), b_text.splitlines(keepends=True),
        fromfile=a_name, tofile=b_name))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--provider', required=True, choices=sorted(PROVIDERS))
    ap.add_argument('--yaml', required=True)
    ap.add_argument('--sigs', default=None,
                    help='Signature sidecar JSON ({api: [{name, c_type}, '
                         '...]}) loaded from a file, rather than live-parsed. '
                         'Mutually exclusive with --header; intended for '
                         'tests/synthetic fixtures that should not depend on '
                         'libclang. Production use should prefer --header.')
    ap.add_argument('--header', action='append', default=[],
                    help='Header file to live-parse via libclang for '
                          'provider-correct OUT-handle signatures (may be '
                          'repeated, e.g. HSA needs hsa.h + hsa_ext_amd.h + '
                          'hsa_api_trace.h + hsa_table_interface.h). '
                          'Mutually exclusive with --sigs.')
    ap.add_argument('--source', action='append', default=[],
                    help='Wrapper implementation TU used as the authoritative '
                    'signature source for implementation-only exported APIs. '
                    'May be repeated; requires --header.')
    ap.add_argument('--extra-arg', action='append', default=[],
                    help='Extra clang arg for --header parsing (e.g. '
                         '-I/path/to/include, -D__HIP_PLATFORM_AMD__=1).')
    ap.add_argument('--dump-resolved', default=None,
                    help='Write the live-resolved signature sidecar JSON to '
                         'this path and exit, without generating tp.h/emit.h. '
                         'Requires --yaml and --header (not --sigs).')
    ap.add_argument('--tp-out')
    ap.add_argument('--emit-out')
    ap.add_argument('--emit-cpp-out', default=None,
                    help='Also emit the out-of-line emit-helper definitions '
                         'to this .cpp path. The --emit-out header becomes '
                         'declarations only; this .cpp carries the real '
                         'tracepoint bodies and the no-op stubs (same #if '
                         'gating as before). Compiled unconditionally into '
                         'the runtime. Optional (omit for legacy '
                         'header-only fixture generation).')
    ap.add_argument('--check', action='store_true',
                    help='Do not write output. Generate to memory, diff '
                         'against the existing --tp-out/--emit-out (and '
                         '--emit-cpp-out, if given) files, print a unified '
                         'diff and exit 1 on any mismatch.')
    args = ap.parse_args()

    if args.sigs and (args.header or args.source):
        ap.error('--sigs is mutually exclusive with --header and --source')
    if not args.sigs and not args.header:
        ap.error('one of --sigs or --header is required')
    if args.source and not args.header:
        ap.error('--source requires --header')

    cfg = PROVIDERS[args.provider]

    if args.dump_resolved:
        if not args.header:
            ap.error('--dump-resolved requires --header (live parse), not --sigs')
        try:
            sidecar = load_sigs_by_api(args.yaml, None, args.header, args.source,
                                       args.extra_arg)
        except AmbiguousDeclarationError as e:
            sys.exit(f"ERROR: {e}")
        os.makedirs(os.path.dirname(args.dump_resolved) or '.', exist_ok=True)
        with open(args.dump_resolved, 'w') as f:
            json.dump(sidecar, f, indent=2, sort_keys=True)
        print(f"wrote {args.dump_resolved} ({len(sidecar)} APIs)", file=sys.stderr)
        return

    if not args.tp_out or not args.emit_out:
        ap.error('--tp-out and --emit-out are required unless --dump-resolved is given')

    try:
        sigs_by_api = load_sigs_by_api(args.yaml, args.sigs, args.header, args.source,
                                       args.extra_arg)
        apis = resolve_apis(args.yaml, args.sigs, args.header, args.source,
                            args.extra_arg)
        return_kinds = resolve_return_kinds(cfg, apis, args.sigs, args.header,
                                            args.source, args.extra_arg)
    except (AmbiguousDeclarationError, AmbiguousInferenceError, ParseError, BudgetError,
            UnsupportedReturnTypeError) as e:
        sys.exit(f"ERROR: {e}")
    regen_cmd = _regen_cmd(cfg, args.yaml, args.tp_out, args.emit_out,
                            emit_cpp_out=args.emit_cpp_out,
                            sigs_path=args.sigs, header_paths=args.header,
                            source_paths=args.source, extra_args=args.extra_arg)
    tp_text, emit_text, emit_cpp_text = generate(
        cfg, apis, args.yaml, args.tp_out, args.emit_out,
        sigs_by_api, return_kinds, regen_cmd)

    outputs = [(args.tp_out, tp_text), (args.emit_out, emit_text)]
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
                print(f"DRIFT: {out_path} does not match generator output:", file=sys.stderr)
                print(_diff(old_text, new_text, out_path, f"{out_path} (generated)"),
                      file=sys.stderr)
        if rc == 0:
            print("OK: " + " and ".join(p for p, _ in outputs)
                  + " match generator output", file=sys.stderr)
        sys.exit(rc)

    for out_path, new_text in outputs:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(new_text)
    print(", ".join(f"wrote {p} ({len(t)} B)" for p, t in outputs), file=sys.stderr)


if __name__ == '__main__':
    main()
