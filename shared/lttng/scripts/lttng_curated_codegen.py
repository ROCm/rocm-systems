#!/usr/bin/env python3
"""Codegen: curated_apis.yaml (+ curated_apis_sigs.json) -> the two generated
LTTng curated-args headers:

  - rocm_<provider>_curated_tp.h        (LTTNG_UST_TRACEPOINT_EVENT defs)
  - rocm_trace_emit_curated.h           (per-API static-inline emit helpers)

Usage (HIP):
    python3 lttng_curated_codegen.py \\
        --provider hip \\
        --yaml     projects/clr/hipamd/scripts/curated_apis.yaml \\
        --sigs     projects/clr/hipamd/scripts/curated_apis_sigs.json \\
        --tp-out   projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h \\
        --emit-out projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h

Usage (HSA):
    python3 lttng_curated_codegen.py \\
        --provider hsa \\
        --yaml     projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml \\
        --sigs     projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis_sigs.json \\
        --tp-out   projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h \\
        --emit-out projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h

CI / pre-commit usage — verify the checked-in headers are still exactly
what the generator produces from the checked-in YAML + sigs cache:

    python3 lttng_curated_codegen.py --provider hip --check \\
        --yaml ... --sigs ... --tp-out ... --emit-out ...

`--check` generates in memory, diffs the result against the on-disk
`--tp-out` / `--emit-out` files, prints a unified diff on mismatch, and
exits 1. It never overwrites the target files.
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
from lttng_curated_lib import parse_yaml_file

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
def emit_tp_event(tp_provider, api):
    """Emit one LTTNG_UST_TRACEPOINT_EVENT block for `api`."""
    name = api['api']
    args = api['args']

    tp_args = []
    tp_fields = []

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
        f"    {tp_provider}, {name}_args,",
        f"    LTTNG_UST_TP_ARGS({args_str}),",
        "    LTTNG_UST_TP_FIELDS(",
    ]
    lines.extend(tp_fields)
    lines.append("    )")
    lines.append(")")
    return "\n".join(lines) + "\n"


def emit_tp_h(cfg, apis, banner):
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
    for api in apis:
        out.append(emit_tp_event(cfg.tp_provider, api))
    # Historical layout has one extra blank line before the closing #endif
    # (present in every checked-in revision of this file); reproduced here.
    out.append(f"\n#endif /* {guard} */\n")
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Codegen: emit.h
# ---------------------------------------------------------------------------
def emit_helper(cfg, api, sigs=None):
    """Emit one active-mode static-inline helper function."""
    name = api['api']
    args = api['args']
    sig_by_name = {s['name']: s['c_type'] for s in sigs} if sigs else {}

    formal_params = []
    cast_exprs = []
    deref_setups = []
    has_out = any(a['dir'] == 'OUT' for a in args)

    for a in args:
        nm, ty, dr = a['name'], a['type'], a['dir']
        if dr == 'IN':
            ptype = HELPER_PARAM_TYPE[ty]
            formal_params.append(f"{ptype} {nm}")
            if ty == 'dim3':
                cast_exprs.extend([f"(uint32_t){nm}.x", f"(uint32_t){nm}.y", f"(uint32_t){nm}.z"])
            elif ty == 'dim3_packed':
                deref_setups.append(f"        const uint64_t {nm}_packed = ROCM_DIM3_PACK({nm});")
                cast_exprs.append(f"{nm}_packed")
            else:
                _, _, cast_tmpl = TYPE_INFO[ty]
                cast_exprs.append(cast_tmpl.format(arg=nm))
        elif dr == 'OUT':
            real_c = sig_by_name.get(nm)
            ptype, deref_tmpl = out_helper_emit(a, real_c)
            formal_params.append(f"{ptype} {nm}_out_ptr")
            if ty in ('ptr', 'handle', 'device_ptr'):
                deref_expr = deref_tmpl.format(p=f"{nm}_out_ptr")
                deref_setups.append(textwrap.dedent(f"""\
                            const uint64_t {nm}_val =
                                (status == {cfg.status_success} && {nm}_out_ptr != NULL)
                                    ? (uint64_t)({deref_expr}) : 0ULL;""").rstrip())
                cast_exprs.append(f"{nm}_val")
            else:
                deref_setups.append(textwrap.dedent(f"""\
                            const auto {nm}_val =
                                (status == {cfg.status_success} && {nm}_out_ptr != NULL)
                                    ? *{nm}_out_ptr : 0;""").rstrip())
                _, _, cast_tmpl = TYPE_INFO[ty]
                cast_exprs.append(cast_tmpl.format(arg=f"{nm}_val"))
        else:
            raise SystemExit(f"INOUT in {name} reached codegen — should be rejected by parser")

    formal_params.append(f"{cfg.status_type} status")
    if not has_out:
        formal_params[-1] = f"{cfg.status_type} /*status*/ /* unused: all-IN API */"

    formal_str = ',\n    '.join(formal_params)
    cast_str = ',\n            '.join(cast_exprs) if cast_exprs else ''
    do_tp_args = f"\n            {cast_str}" if cast_str else ''
    do_tp_call = (f"{cfg.tp_provider}, {name}_args,{do_tp_args}" if cast_str
                  else f"{cfg.tp_provider}, {name}_args")
    setups = '\n'.join(deref_setups)

    body_inner = (setups + '\n' if setups else '') + textwrap.dedent(f"""\
                lttng_ust_do_tracepoint({do_tp_call});""")

    return textwrap.dedent(f"""\
        static inline void rocm_trace_emit_{name}_args(
            {formal_str}) {{
            if (rocm_trace_disabled()) return;
            if (lttng_ust_tracepoint_enabled({cfg.tp_provider}, {name}_args)) {{
        {body_inner}
            }}
        }}
        """)


def emit_noop_helper(cfg, api, sigs=None):
    """No-op helper for <PROVIDER>_ENABLE_LTTNG_UST=0 builds. Signature must
    byte-match the active-mode helper (unnamed params — no body references
    them, so no -Wunused-parameter mitigation is needed)."""
    name = api['api']
    args = api['args']
    sig_by_name = {s['name']: s['c_type'] for s in sigs} if sigs else {}
    formals = []
    for a in args:
        if a['dir'] == 'OUT':
            real_c = sig_by_name.get(a['name'])
            formals.append(out_helper_param_type(a, real_c))
        elif a['type'] in ('dim3', 'dim3_packed'):
            formals.append('dim3')
        else:
            formals.append(HELPER_PARAM_TYPE[a['type']])
    formals.append(cfg.status_type)
    formal_str = ', '.join(formals)
    return f"static inline void rocm_trace_emit_{name}_args({formal_str}) {{}}\n"


def emit_emit_h(cfg, apis, banner, sigs_by_api=None):
    enable_macro = 'HIP_ENABLE_LTTNG_UST' if cfg.key == 'hip' else 'HSA_ENABLE_LTTNG_UST'
    macro_guard = f"ROCM_{cfg.key.upper()}_TRACE_EMIT_CURATED_H_"

    out = [banner]
    out.append(f"#ifndef {macro_guard}\n#define {macro_guard}\n\n"
               f"#include <stdint.h>\n#include <stddef.h>\n")
    out.append(cfg.emit_includes)

    needs_dim3 = any(a['type'] in ('dim3', 'dim3_packed') for api in apis for a in api['args'])
    if needs_dim3:
        out.append('#include "rocm_dim3_pack.h"\n')

    out.append(f"\n#if defined({enable_macro}) && {enable_macro}\n\n"
               f"#include <atomic>\n#include \"{cfg.tp_provider}_curated_tp.h\"\n")

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
        out.append(emit_helper(cfg, api, sigs=sigs))
        out.append('\n')

    out.append(f"\n#else  /* {enable_macro} not defined — all helpers are no-ops */\n\n")
    for api in apis:
        sigs = (sigs_by_api or {}).get(api['api'])
        out.append(emit_noop_helper(cfg, api, sigs=sigs))

    out.append(f"\n#endif  /* {enable_macro} */\n\n#endif  /* {macro_guard} */\n")
    return ''.join(out)


# ---------------------------------------------------------------------------
# Banners
# ---------------------------------------------------------------------------
def _regen_cmd(cfg, yaml_path, sigs_path, tp_out, emit_out):
    """Render the exact regeneration command with paths relative to the
    repo root. Normalizing here (rather than echoing back whatever the
    caller passed verbatim) keeps the embedded banner command stable
    across absolute-path vs. relative-path invocations of the generator
    itself — otherwise --check would report spurious "drift" any time
    the tool was re-run from a different working directory."""
    def rel(p):
        try:
            return os.path.relpath(os.path.abspath(p), REPO_ROOT)
        except ValueError:
            return p
    return [
        "python3 shared/lttng/scripts/lttng_curated_codegen.py \\",
        f"    --provider {cfg.key} \\",
        f"    --yaml {rel(yaml_path)} \\",
        f"    --sigs {rel(sigs_path)} \\",
        f"    --tp-out {rel(tp_out)} \\",
        f"    --emit-out {rel(emit_out)}",
    ]


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


def emit_banner(cfg, yaml_path, sha256, regen_cmd):
    return _comment_block([
        f"AUTO-GENERATED by lttng_curated_codegen.py from {os.path.basename(yaml_path)}.",
        "Do not edit by hand — regenerate instead (see command below).",
        "",
        f"SHA256({os.path.basename(yaml_path)}) at generation: {sha256}",
        "",
        "Per-API typed emit helpers for curated parameter capture. Every",
        "helper takes (<captured-args...>, <status_type> status); status is",
        "the call's success result, used to gate OUT-param deref. All-IN",
        "APIs accept it but mark it unused.",
        "",
        "Regenerate with:",
        *(f"  {l}" for l in regen_cmd),
    ])


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


# ---------------------------------------------------------------------------
# Top-level generation entry point (shared by normal mode and --check mode)
# ---------------------------------------------------------------------------
def generate(cfg, yaml_path, sigs_path, tp_out_path, emit_out_path):
    """Return (tp_text, emit_text)."""
    apis = parse_yaml_file(yaml_path)
    with open(yaml_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    with open(sigs_path) as f:
        sigs_by_api = json.load(f)

    regen_cmd = _regen_cmd(cfg, yaml_path, sigs_path, tp_out_path, emit_out_path)

    tp_text = emit_tp_h(cfg, apis, tp_banner(cfg, apis, yaml_path, sha256, regen_cmd))
    emit_text = emit_emit_h(cfg, apis, emit_banner(cfg, yaml_path, sha256, regen_cmd),
                             sigs_by_api=sigs_by_api)

    if cfg.clang_format_emit:
        style_file = os.path.join(REPO_ROOT, 'projects', 'clr', '.clang-format')
        emit_text = _clang_format(emit_text, style_file)

    return tp_text, emit_text


def _diff(a_text, b_text, a_name, b_name):
    return ''.join(difflib.unified_diff(
        a_text.splitlines(keepends=True), b_text.splitlines(keepends=True),
        fromfile=a_name, tofile=b_name))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--provider', required=True, choices=sorted(PROVIDERS))
    ap.add_argument('--yaml', required=True)
    ap.add_argument('--sigs', required=True,
                    help='Verifier signature sidecar JSON '
                         '({api: [{name, c_type}, ...]}) — required in '
                         'Phase 1; provides provider-correct OUT-handle '
                         'helper signatures and real C parameter types.')
    ap.add_argument('--tp-out', required=True)
    ap.add_argument('--emit-out', required=True)
    ap.add_argument('--check', action='store_true',
                    help='Do not write output. Generate to memory, diff '
                         'against the existing --tp-out/--emit-out files, '
                         'print a unified diff and exit 1 on any mismatch.')
    args = ap.parse_args()

    cfg = PROVIDERS[args.provider]
    tp_text, emit_text = generate(cfg, args.yaml, args.sigs, args.tp_out, args.emit_out)

    if args.check:
        rc = 0
        for out_path, new_text in ((args.tp_out, tp_text), (args.emit_out, emit_text)):
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
            print(f"OK: {args.tp_out} and {args.emit_out} match generator output",
                  file=sys.stderr)
        sys.exit(rc)

    os.makedirs(os.path.dirname(args.tp_out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.emit_out) or '.', exist_ok=True)
    with open(args.tp_out, 'w') as f:
        f.write(tp_text)
    with open(args.emit_out, 'w') as f:
        f.write(emit_text)
    print(f"wrote {args.tp_out} ({len(tp_text)} B), "
          f"{args.emit_out} ({len(emit_text)} B)", file=sys.stderr)


if __name__ == '__main__':
    main()
