#!/usr/bin/env python3
"""Rewrite generic/curated HIP wrappers to schema-v1 combined-event calls.

Schema v1 replaces the generic pair (shared ``rocm_trace_emit_hip_api_enter`` +
per-API ``<api>_args`` + shared ``hip_api_exit_*``) with ONE combined event
fired twice per call. Each curated wrapper therefore emits:

  - ``rocm_trace_emit_<api>_enter(<IN arg casts>);`` (phase=ENTER) up front,
    right after the IN-value snapshots it depends on, carrying the
    ``/* __ROCM_CURATED__: <api> */`` sentinel; and
  - the combined EXIT record via the ``ROCM_TRACE_RET_*_CURATED`` return
    macro, which now expands to ``rocm_trace_emit_<api>_exit(<OUT args>,
    status)`` (see hip_table_interface.cpp).

IN arguments flow to the ``_enter`` call; OUT arguments (safe, audited output
pointers on the small set of hand-curated wrappers) flow to the return macro,
which forwards them to the ``_exit`` helper for success-gated deref. All-IN
wrappers use the ``_NOARGS`` return-macro variant (their EXIT record carries
only phase + status).

The tool is idempotent: it accepts wrappers still in the generic shape (shared
enter + ``_args`` macro) or already in combined-event form, and normalizes both to
the combined-event shape.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED_SCRIPTS = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'shared',
                                               'lttng', 'scripts'))
sys.path.insert(0, SHARED_SCRIPTS)

from lttng_curated_lib import parse_yaml_file
from lttng_curated_verify import (expand_compact_apis, parse_declarations,
                                   compute_sidecar, resolve_curated_return_kind,
                                   resolve_declaration)
# The codegen owns the exact OUT-param pointer type each generated
# rocm_trace_emit_<api>_exit(...) helper uses; reuse it (rather than
# re-deriving) so the cast the wrapper emits on each OUT arg names EXACTLY
# that type and the call type-checks.
from lttng_curated_codegen import out_helper_param_type
from clang import cindex


# IN-arg cast expression applied to the __rocm_in_<name> snapshot local. Must
# match the generated <api>_enter helper's parameter type (see
# lttng_curated_codegen.HELPER_PARAM_TYPE).
IN_CAST = {
    'handle': '(uint64_t)(uintptr_t)({arg})',
    'ptr': '(const void*)(uintptr_t)({arg})',
    'device_ptr': '(uint64_t)({arg})',
    'size': '({arg})',
    'int32': '({arg})',
    'uint32': '({arg})',
    'int64': '({arg})',
    'uint64': '({arg})',
    'float': '({arg})',
    'enum': '(int32_t)({arg})',
    'bool': '({arg})',
    'cstring': '(const char*)({arg})',
    'dim3': '({arg})',
    'dim3_packed': '({arg})',
}

# generic shared enter, still present in un-migrated wrappers.
SHARED_ENTER_RE = re.compile(r'rocm_trace_emit_hip_api_enter\s*\(\s*__func__\s*\)\s*;')
# per-API combined-event enter, present in already-migrated wrappers (idempotent rerun).
GENERIC_RET_RE = re.compile(
    r'ROCM_TRACE_RET_(STATUS|PTR|VOID)(?!_CURATED)\s*\(\s*([^;]+?)\s*\)\s*;',
    flags=re.DOTALL)
# Curated return macro (any kind), captured or NOARGS, already present.
CURATED_RET_RE = re.compile(
    r'ROCM_TRACE_RET_(STATUS|PTR|VOID|I32)_CURATED(_NOARGS)?\s*\(\s*([^;]+?)\s*\)\s*;',
    flags=re.DOTALL)
# legacy direct shared void exit, for the pure-void wrappers with no return macro.
DIRECT_VOID_EXIT_RE = re.compile(
    r'rocm_trace_emit_hip_api_exit_void\s*\(\s*__func__\s*\)\s*;')
# legacy direct <api>_args helper call (STRUCT/void wrappers in non-migrated TUs).
DIRECT_ARGS_RE_TMPL = r'rocm_trace_emit_{name}_args\s*\(\s*([^;]*?)\s*\)\s*;'
# One IN-value snapshot line: `[indent]auto const __rocm_in_X = ...;[ws][nl]`.
_SNAPSHOT_LINE_RE = re.compile(
    r'[ \t]*auto\s+const\s+__rocm_in_[A-Za-z0-9_]+\s*=\s*[^;]+;[ \t]*\n?')


def sentinel(name):
    return f'/* __ROCM_CURATED__: {name} */'


def find_wrapper_body(text, name):
    """Return the byte offsets (start, end) of the first definition of `name`."""
    pattern = re.compile(r'\b' + re.escape(name) + r'\s*\(')
    for match in pattern.finditer(text):
        index = match.end() - 1
        depth = 0
        while index < len(text):
            if text[index] == '(':
                depth += 1
            elif text[index] == ')':
                depth -= 1
                if depth == 0:
                    break
            index += 1
        if depth != 0:
            continue
        index += 1
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != '{':
            continue
        start = index
        depth = 0
        while index < len(text):
            if text[index] == '{':
                depth += 1
            elif text[index] == '}':
                depth -= 1
                if depth == 0:
                    return start, index + 1
            index += 1
    return None


def _source_parameters(source_path, api_names, extra_args):
    """Return {api_name: [param_spelling, ...]} for every wrapper DEFINED in
    `source_path`, in declaration order, read from the real implementation TU
    via libclang.

    The curated YAML uses public-header parameter names, but a few wrappers
    spell a positional parameter differently in their implementation (e.g.
    hipChooseDeviceR0600's public `prop` is `properties` in the impl). The IN
    snapshot's RHS must use the IMPLEMENTATION-local name so the generated
    wrapper actually compiles; the __rocm_in_<name> local and the _enter helper
    ABI keep the public/YAML name. Only definitions physically located in this
    TU are returned (a definition elsewhere must not shadow this file's)."""
    arguments = ['-x', 'c++', '-std=c++17'] + list(extra_args)
    index = cindex.Index.create()
    translation_unit = index.parse(source_path, args=arguments)
    parameters = {}
    source_realpath = os.path.realpath(source_path)
    for node in translation_unit.cursor.walk_preorder():
        try:
            kind = node.kind
        except ValueError:
            continue
        if (kind != cindex.CursorKind.FUNCTION_DECL or not node.is_definition()
                or node.spelling not in api_names or node.location.file is None
                or os.path.realpath(node.location.file.name) != source_realpath):
            continue
        parameters[node.spelling] = [arg.spelling for arg in node.get_arguments()]
    return parameters


def in_snapshots_and_casts(api, source_param_names=None):
    """Return (snapshot_decls, enter_cast_exprs) for the IN args of `api`.

    Snapshots capture each IN param's value up front (``auto const
    __rocm_in_<yaml_name> = <impl_name>;``); the enter helper is then called
    with the matching cast expressions over those snapshots. `source_param_names`
    maps the YAML/public arg name to the IMPLEMENTATION-local parameter name
    (positional), so an aliased param (prop -> properties) reads the value that
    actually exists in the wrapper body. When absent (no impl override known),
    the YAML name is used on both sides, as before.
    """
    source_param_names = source_param_names or {}
    decls = []
    casts = []
    for arg in api['args']:
        if arg['dir'] != 'IN':
            continue
        yaml_name = arg['name']
        local = f'__rocm_in_{yaml_name}'
        rhs = source_param_names.get(yaml_name, yaml_name)
        decls.append(f'auto const {local} = {rhs};')
        casts.append(IN_CAST[arg['type']].format(arg=local))
    return decls, casts


def _normalize_c_type(t):
    """Collapse whitespace so `void **` and `void**` compare equal."""
    return re.sub(r'\s+', '', t or '')


# Canonical pointee type of each generated exit-helper OUT param pointer type.
# The scalar/float/bool helper params are fixed builtins; comparing this
# against the OUT arg's own libclang canonical type decides whether a cast is
# actually needed. This lets `unsigned int*` (canonical `unsigned int *`)
# satisfy a `uint32_t*` param with NO cast — matching what the hand-curated reference
# wrappers did — while a typedef'd-enum arg (canonical is the enum type, not
# `int`) is genuinely different from `int32_t*` and DOES get an `(int32_t*)`
# cast.
_HELPER_PARAM_CANON = {
    'size': 'unsigned long',      # size_t
    'uint32': 'unsigned int',
    'uint64': 'unsigned long',
    'int32': 'int',
    'int64': 'long',
    'float': 'float',
    'enum': 'int',
    'bool': 'int',
}


def out_cast_exprs(api, sig_by_name, canon_by_name):
    """Return the EXIT return-macro OUT-argument expressions, in YAML order,
    each cast to the EXACT pointer type the codegen's generated
    rocm_trace_emit_<api>_exit(...) helper declares for that OUT param.

    `sig_by_name` maps arg-name -> real C (typedef) type; `canon_by_name`
    maps arg-name -> its libclang CANONICAL type. out_helper_param_type()
    (imported from the codegen) computes the helper's OUT param type from the
    DSL type + real C type exactly as the generated helper does:
      enum->int32_t*, size->size_t*, uint32->uint32_t*, uint64->uint64_t*,
      int32->int32_t*, int64->int64_t*, float->float*, bool->int*, and
      ptr/handle/device_ptr-> the sidecar real C type (or void** fallback).

    To reproduce the hand-curated reference wrappers byte-for-byte, the cast is
    OMITTED whenever the OUT arg's raw pointer already has (canonically) the
    helper's param type -- i.e. the ptr/handle/void** cases AND the
    scalar/int cases where the source's canonical type matches the param's
    canonical type (e.g. `unsigned int*`==`uint32_t*`, `int*`==`int32_t*`).
    A cast is emitted only when a genuine representational conversion is
    needed -- notably a typedef'd-enum OUT arg (`hipStreamCaptureStatus*` ->
    `(int32_t*)`)."""
    exprs = []
    for arg in api['args']:
        if arg['dir'] != 'OUT':
            continue
        real_c = sig_by_name.get(arg['name'])
        param_type = out_helper_param_type(arg, real_c)
        ty = arg['type']
        needs_cast = True
        if ty in ('ptr', 'handle', 'device_ptr'):
            # Helper param type IS the real C type (or void** fallback); the
            # raw pointer already matches -> no cast (generic-shape behavior).
            needs_cast = real_c is None or \
                _normalize_c_type(param_type) != _normalize_c_type(real_c)
        else:
            # Scalar/enum/float/bool: cast only if the arg's canonical pointee
            # type differs from the helper param's canonical pointee type.
            arg_canon = canon_by_name.get(arg['name'], '')
            arg_pointee = _normalize_c_type(_strip_trailing_star(arg_canon))
            param_pointee = _normalize_c_type(_HELPER_PARAM_CANON.get(ty, ''))
            needs_cast = arg_pointee != param_pointee
        if needs_cast:
            exprs.append(f'({param_type})({arg["name"]})')
        else:
            exprs.append(arg['name'])
    return exprs


def _strip_trailing_star(canon):
    """Return the pointee spelling of a canonical pointer type: strip exactly
    one trailing `*` (and surrounding whitespace)."""
    s = (canon or '').strip()
    if s.endswith('*'):
        return s[:-1].strip()
    return s


# Wrapper bodies are indented two spaces; keep the injected enter region at
# that column and wrap the enter-call argument list one cast per line at a
# deeper indent. Emitting each snapshot and each enter-call argument on its
# own physical line keeps every line far under hip_prof_gen.py's per-line
# REC_MAX_LEN (~1024) budget, even for the 14-arg launch APIs whose single-
# line form previously reached ~1246 chars.
_BODY_INDENT = '  '
_ENTER_ARG_INDENT = '      '


def build_enter_region(api, source_param_names=None):
    """Build the replacement text for the leading enter region: IN snapshots
    (one per line) followed by the per-API enter call whose arguments are
    wrapped one per line, carrying the curated sentinel after the closing
    ``);``. Kept multi-line so no physical line exceeds hip_prof_gen.py's
    REC_MAX_LEN; the region is still recognized (and thus regenerated
    idempotently) on a rerun because the strip + enter-region detection match
    across newlines. `source_param_names` maps YAML arg name -> impl-local
    parameter name for the snapshot RHS (see in_snapshots_and_casts)."""
    decls, casts = in_snapshots_and_casts(api, source_param_names)
    name = api['api']
    lines = [f'{_BODY_INDENT}{decl}' for decl in decls]
    if casts:
        lines.append(f'{_BODY_INDENT}rocm_trace_emit_{name}_enter(')
        for i, cast in enumerate(casts):
            terminator = ');' if i == len(casts) - 1 else ','
            lines.append(f'{_ENTER_ARG_INDENT}{cast}{terminator}')
        # Sentinel trails the closing `);` so the enter-region detection in
        # migrate_source() (which searches for the sentinel just after the
        # enter call) stays valid on reruns.
        lines[-1] = f'{lines[-1]} {sentinel(name)}'
    else:
        lines.append(f'{_BODY_INDENT}rocm_trace_emit_{name}_enter(); {sentinel(name)}')
    # The region replaces a fragment that began at the (already-consumed)
    # position of the old enter/shared-enter token, so lead with the body
    # indent but do NOT add a trailing newline (the original text following
    # the consumed region — e.g. `\n  TRY;` — is preserved by the caller).
    return '\n'.join(lines)


def curated_return_macro(kind, name, expression, outs):
    """Build the schema-v1 curated return-macro call. OUT args (if any) are
    forwarded; all-IN wrappers use the _NOARGS variant."""
    args = ', '.join(outs)
    if kind in ('STATUS', 'I32', 'VOID'):
        macro = f'ROCM_TRACE_RET_{kind}_CURATED' + ('' if outs else '_NOARGS')
        return f'{macro}({name}, {expression}' + (f', {args}' if outs else '') + ');'
    if kind == 'PTR':
        macro = 'ROCM_TRACE_RET_PTR_CURATED' + ('' if outs else '_NOARGS')
        return (f'{macro}({name}, auto, {expression}' +
                (f', {args}' if outs else '') + ');')
    raise AssertionError(kind)


def migrate_source(source_path, apis, return_kinds, sigs_by_api, canon_by_api,
                   header_params_by_api, extra_args):
    with open(source_path) as file:
        source = file.read()

    # Resolve the IMPLEMENTATION-local parameter spellings for every wrapper
    # DEFINED in this TU, then map (positionally) the YAML/public arg name to
    # the impl-local name. Used for the IN-snapshot RHS so aliased params
    # (e.g. hipChooseDeviceR0600's prop -> properties) still compile on rerun.
    impl_params = _source_parameters(source_path, set(apis), extra_args)
    source_param_names_by_api = {}
    for name, params in impl_params.items():
        header_params = header_params_by_api.get(name)
        if not header_params or len(header_params) != len(params):
            # Count mismatch or unknown header decl: fall back to identity
            # (YAML name on both sides) rather than mis-aliasing positionally.
            continue
        source_param_names_by_api[name] = {
            header_name: impl_name
            for (header_name, *_rest), impl_name in zip(header_params, params)
        }

    edits = []
    changed = 0
    for name, api in apis.items():
        body = find_wrapper_body(source, name)
        if body is None:
            continue
        body_start, body_end = body
        body_text = source[body_start:body_end]
        if sentinel(name) not in body_text and not SHARED_ENTER_RE.search(body_text):
            # Not a wrapper this tool owns (no curation marker, no shared
            # enter to migrate).
            continue
        return_kind = return_kinds[name]
        sig_by_name = {s['name']: s['c_type']
                       for s in (sigs_by_api.get(name) or [])}
        canon_by_name = canon_by_api.get(name, {})
        source_param_names = source_param_names_by_api.get(name)
        # OUT args forwarded to the EXIT return macro, each cast to the exact
        # pointer type the generated _exit helper expects (see out_cast_exprs).
        outs = out_cast_exprs(api, sig_by_name, canon_by_name)
        wrapper_edits = []

        # Replace the entire leading enter region (any pre-existing IN-snapshot
        # lines + the shared-generic-or-per-API-combined enter call + trailing sentinel)
        # with a freshly-built, deterministically-formatted multi-line region.
        # Consuming the whole contiguous block in one edit (rather than
        # separately stripping snapshots) keeps the output format independent
        # of whatever whitespace the previous run produced, so reruns are
        # stable and no physical line exceeds hip_prof_gen.py's REC_MAX_LEN.
        shared = SHARED_ENTER_RE.search(body_text)
        per_api_enter = re.compile(
            r'rocm_trace_emit_' + re.escape(name) + r'_enter\s*\([^;]*\)\s*;')
        sentinel_re = re.compile(
            r'/\*\s*__ROCM_CURATED__:\s*' + re.escape(name) + r'\s*\*/')
        enter_region = build_enter_region(api, source_param_names)

        if shared:
            enter_match = shared
        else:
            enter_match = per_api_enter.search(body_text)
            if enter_match is None:
                raise SystemExit(
                    f'{source_path}: {name}: no shared or per-API enter found')

        # The generic shape places the shared enter + sentinel first, then the IN
        # snapshots on following lines; the combined-event shape places snapshots first,
        # then the multi-line per-API enter + sentinel. Consume the WHOLE
        # contiguous cluster spanning every IN-snapshot line and the enter
        # call + sentinel (in either order), then re-emit one freshly-
        # formatted, deterministically-indented region. Doing snapshot removal
        # and the enter rewrite as a single edit guarantees no stray snapshot
        # survives (the previous single-edit bug duplicated them).
        region_start = enter_match.start()
        region_end = enter_match.end()
        # Extend end past the trailing sentinel.
        sm = sentinel_re.search(body_text, region_end, region_end + 120)
        if sm and body_text[region_end:sm.start()].strip() == '':
            region_end = sm.end()

        # Ranges of every IN-snapshot line in the body. Absorb any that are
        # contiguous with the current region across whitespace only.
        snapshot_ranges = [
            (m.start(), m.end())
            for m in _SNAPSHOT_LINE_RE.finditer(body_text)]
        grew = True
        while grew:
            grew = False
            for s, e in snapshot_ranges:
                if e <= region_start and body_text[e:region_start].strip() == '':
                    region_start = s
                    grew = True
                elif s >= region_end and body_text[region_end:s].strip() == '':
                    region_end = e
                    grew = True

        # Extend region_start left over same-line leading whitespace.
        line_start = body_text.rfind('\n', 0, region_start) + 1
        if body_text[line_start:region_start].strip() == '':
            region_start = line_start

        # Normalize spacing before the next statement (TRY;, do {, or the
        # return macro): consume trailing horizontal whitespace and one
        # optional newline + indent, then re-emit exactly "\n  ".
        trail = re.compile(r'[ \t]*\n?[ \t]*')
        tm = trail.match(body_text, region_end)
        if tm:
            region_end = tm.end()

        wrapper_edits.append(
            (body_start + region_start, body_start + region_end,
             enter_region + '\n' + _BODY_INDENT))

        # 3. Rewrite the return path.
        generic_matches = list(GENERIC_RET_RE.finditer(body_text))
        curated_matches = list(CURATED_RET_RE.finditer(body_text))
        direct_args_re = re.compile(DIRECT_ARGS_RE_TMPL.format(name=re.escape(name)),
                                    flags=re.DOTALL)
        direct_args = list(direct_args_re.finditer(body_text))
        if generic_matches:
            for m in generic_matches:
                _kind, expression = m.group(1), m.group(2)
                wrapper_edits.append(
                    (body_start + m.start(), body_start + m.end(),
                     curated_return_macro(return_kind, name, expression, outs)))
        elif curated_matches:
            for m in curated_matches:
                # m.group(3) is the full arg list of the existing curated
                # macro (api[, ptr_type], expr[, captured...]). Recover the
                # call expression and rebuild for the combined-event exit (OUT-only capture).
                expression = _extract_call_expression(m.group(3), return_kind)
                wrapper_edits.append(
                    (body_start + m.start(), body_start + m.end(),
                     curated_return_macro(return_kind, name, expression, outs)))
        elif direct_args:
            # Pure-void / STRUCT wrappers with a direct _args helper call and
            # (for void) a direct shared void exit. Replace with an enter is
            # already handled above; here rewrite the direct emission to the
            # exit helper.
            for m in direct_args:
                exit_call = (f'rocm_trace_emit_{name}_exit();'
                             if return_kind in ('VOID', 'STRUCT')
                             else f'rocm_trace_emit_{name}_exit(hipSuccess);')
                # STRUCT/void _args helpers took (IN casts..., hipSuccess);
                # combined-event exit takes only OUT args (none here) + optional status.
                if outs:
                    exit_call = (f'rocm_trace_emit_{name}_exit('
                                 + ', '.join(outs) + ');'
                                 if return_kind in ('VOID', 'STRUCT')
                                 else f'rocm_trace_emit_{name}_exit('
                                 + ', '.join(outs) + ', hipSuccess);')
                wrapper_edits.append(
                    (body_start + m.start(), body_start + m.end(), exit_call))
            for m in DIRECT_VOID_EXIT_RE.finditer(body_text):
                wrapper_edits.append(
                    (body_start + m.start(), body_start + m.end(), ''))
        elif re.search(r'\brocm_trace_emit_' + re.escape(name) + r'_exit\s*\(', body_text):
            # Idempotent rerun: the return path is already the combined-event direct
            # rocm_trace_emit_<api>_exit(...) emission (STRUCT/void wrappers
            # with no return macro to rewrite). Leave it as-is; only the enter
            # region above is re-normalized.
            pass
        else:
            raise SystemExit(
                f'{source_path}: {name}: no recognized return/emit path found')

        edits.extend(wrapper_edits)
        changed += 1

    edits.sort(key=lambda edit: edit[0], reverse=True)
    output = source
    for start, end, replacement in edits:
        output = output[:start] + replacement + output[end:]
    with open(source_path, 'w') as file:
        file.write(output)
    return changed, len(edits)


def _extract_call_expression(macro_args, return_kind):
    """Extract the wrapped call expression from an existing curated return
    macro's argument text. The layout is:
        <api>, <expr>[, captured...]                    (STATUS/I32/VOID)
        <api>, <ptr_type>, <expr>[, captured...]        (PTR)
    Split on top-level commas (respecting parens/brackets) and take the
    expression element."""
    parts = _split_top_level(macro_args)
    if return_kind == 'PTR':
        # api, ptr_type, expr, ...
        return parts[2].strip()
    return parts[1].strip()


def _split_top_level(text):
    """Split `text` on commas not enclosed in (), [], <>-free parens."""
    parts = []
    depth = 0
    current = []
    for ch in text:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current))
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True,
                        help='Wrapper source to migrate; may be repeated.')
    parser.add_argument('--curated-yaml', required=True)
    parser.add_argument('--header', action='append', required=True)
    parser.add_argument('--extra-arg', action='append', default=[])
    args = parser.parse_args()

    raw_apis = parse_yaml_file(args.curated_yaml)
    declarations = parse_declarations(args.header, args.source, args.extra_arg)
    expanded = expand_compact_apis(raw_apis, declarations)
    apis = {api['api']: api for api in expanded}
    # Real per-arg C types (the codegen/verifier signature sidecar), used to
    # cast OUT args to the generated _exit helper's exact param types.
    sigs_by_api = compute_sidecar(expanded, declarations)
    # Per-arg canonical libclang types, used to decide when an OUT-arg cast is
    # actually required (canonical mismatch) vs redundant (matches the generic form). Also
    # keep the resolved (public-header) parameter tuples so migrate_source()
    # can positionally align them with each impl TU's local parameter names.
    canon_by_api = {}
    header_params_by_api = {}
    for name, api in apis.items():
        params = resolve_declaration(name, declarations[name],
                                     [arg['name'] for arg in api['args']])
        canon_by_api[name] = {p[0]: p[2] for p in params}
        header_params_by_api[name] = params
    return_kinds = {
        name: resolve_curated_return_kind(
            'hip', name, declarations[name], [arg['name'] for arg in api['args']])
        for name, api in apis.items()
    }
    wrappers = edits = 0
    for source_path in args.source:
        count, edit_count = migrate_source(source_path, apis, return_kinds,
                                           sigs_by_api, canon_by_api,
                                           header_params_by_api, args.extra_arg)
        wrappers += count
        edits += edit_count
        print(f'{source_path}: migrated {count} wrappers ({edit_count} edits)', file=sys.stderr)
    print(f'migrated {wrappers} wrapper bodies ({edits} edits total)', file=sys.stderr)


if __name__ == '__main__':
    main()
