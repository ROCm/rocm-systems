#!/usr/bin/env python3
"""Rewrite HSA's curated table-interface wrappers to schema-v1 combined events.

Schema v1 folds the generic triple (shared ``hsa_api_enter`` + per-API
``<api>_args`` + shared ``hsa_api_exit_*``) into ONE combined event fired
twice per call. Each curated HSA wrapper therefore:

  - snapshots its IN args (``auto const __rocm_in_<name> = <src>;``) and emits
    the ENTER record up front via ``rocm_trace_emit_<api>_enter(<IN casts>);``,
    carrying the ``/* __ROCM_CURATED__: <api> */`` sentinel; and
  - emits the matching EXIT record through the curated return macro
    (``ROCR_TRACE_API_RET_*_CURATED_HSA``), which now expands to
    ``rocm_trace_emit_<api>_exit(<OUT args...>, <return value>)`` (see
    hsa_table_interface.cpp).

IN args flow to ``_enter``; the small set of hand-curated OUT pointers flow to
the return macro (and thus ``_exit``) for success-gated deref. All-IN wrappers
use the ``_NOARGS`` return-macro variant.

The pure-void ``hsa_table_interface_init`` wrapper has no return statement for
a return macro to wrap, so it emits ``_enter``/``_exit`` directly.

The migrator is idempotent: generic-shape wrappers (shared enter + ``_args``) and
already-combined-event wrappers are both normalized to the combined-event shape.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lttng_curated_lib import parse_yaml_file
from lttng_curated_verify import (expand_compact_apis, parse_headers,
                                   resolve_curated_return_kind,
                                   resolve_declaration)
from clang import cindex


RETURN_MACROS = {
    'STATUS': 'ROCR_TRACE_API_RET_STATUS_CURATED_HSA',
    'PTR': 'ROCR_TRACE_API_RET_PTR_CURATED_HSA',
    'VOID': 'ROCR_TRACE_API_RET_VOID_CURATED_HSA',
    'U64': 'ROCR_TRACE_API_RET_U64_CURATED_HSA',
    'I64': 'ROCR_TRACE_API_RET_I64_CURATED_HSA',
    'U32': 'ROCR_TRACE_API_RET_U32_CURATED_HSA',
}

SHARED_ENTER_RE = re.compile(
    r'rocm_trace_emit_hsa_api_enter\s*\(\s*__func__\s*\)\s*;')
# generic curated return macro (captured IN+OUT, or _NOARGS).
CURATED_RETURN_RE = re.compile(
    r'ROCR_TRACE_API_RET_(STATUS|PTR|VOID|U64|I64|U32)_CURATED_HSA(_NOARGS)?\s*'
    r'\(\s*([^;]+?)\s*\)\s*;', flags=re.DOTALL)
DIRECT_VOID_EXIT_RE = re.compile(
    r'rocm_trace_emit_hsa_api_exit_void\s*\(\s*__func__\s*\)\s*;')
# One IN-value snapshot line: `[indent]auto const __rocm_in_X = ...;[ws][nl]`.
_SNAPSHOT_LINE_RE = re.compile(
    r'[ \t]*auto\s+const\s+__rocm_in_[A-Za-z0-9_]+\s*=\s*[^;]+;[ \t]*\n?')


def _find_wrapper_body(text, name):
    """Return the byte range of ``name``'s definition body, or None."""
    pattern = re.compile(r'\b' + re.escape(name) + r'\s*\(')
    for match in pattern.finditer(text):
        pos = match.end() - 1
        parens = 0
        while pos < len(text):
            if text[pos] == '(':
                parens += 1
            elif text[pos] == ')':
                parens -= 1
                if parens == 0:
                    break
            pos += 1
        if parens != 0:
            continue
        pos += 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text) or text[pos] != '{':
            continue
        begin = pos
        braces = 0
        while pos < len(text):
            if text[pos] == '{':
                braces += 1
            elif text[pos] == '}':
                braces -= 1
                if braces == 0:
                    return begin, pos + 1
            pos += 1
    return None


def _capture_expression(arg, c_type):
    """Cast an IN snapshot to the generated enter helper's parameter type."""
    name = f'__rocm_in_{arg["name"]}'
    kind = arg['type']
    if kind == 'handle':
        if '*' in c_type:
            return f'(uint64_t)(uintptr_t)({name})'
        return f'(uint64_t)(({name}).handle)'
    if kind == 'ptr':
        return f'(const void*)(uintptr_t)({name})'
    if kind == 'device_ptr':
        return f'(uint64_t)({name})'
    if kind == 'enum':
        return f'(int32_t)({name})'
    if kind == 'cstring':
        return f'(const char*)({name})'
    return f'({name})'


def _source_parameters(source_path, api_names, extra_args):
    """Return source wrapper parameter names in declaration order."""
    arguments = ['-x', 'c++', '-std=c++17'] + list(extra_args)
    source_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(source_path))))
    arguments += ['-I', source_root, '-I', os.path.join(source_root, 'inc')]
    index = cindex.Index.create()
    translation_unit = index.parse(source_path, args=arguments)
    parameters = {}
    source_realpath = os.path.realpath(source_path)
    for node in translation_unit.cursor.walk_preorder():
        if (node.kind != cindex.CursorKind.FUNCTION_DECL or not node.is_definition()
                or node.spelling not in api_names or node.location.file is None
                or os.path.realpath(node.location.file.name) != source_realpath):
            continue
        parameters[node.spelling] = [(argument.spelling, argument.type.spelling)
                                     for argument in node.get_arguments()]
    missing = sorted(api_names - parameters.keys())
    if missing:
        raise SystemExit(f'ERROR: source declarations not found for {missing}')
    return parameters


def _in_snapshots_and_casts(api, parameter_types):
    """Return (snapshot_decls, enter_cast_exprs) for the IN args of `api`.

    The snapshot RHS uses the IMPLEMENTATION-local parameter name (which may
    differ from the public/YAML name, e.g. hsa_soft_queue_create's completion
    signal), while the __rocm_in_<name> local keeps the YAML name so the
    downstream _enter helper ABI/field names stay stable."""
    decls = []
    casts = []
    for arg in api['args']:
        if arg['dir'] != 'IN':
            continue
        source_name, c_type = parameter_types[arg['name']]
        decls.append(f'auto const __rocm_in_{arg["name"]} = {source_name};')
        casts.append(_capture_expression(arg, c_type))
    return decls, casts


def _out_names(api):
    return [arg['name'] for arg in api['args'] if arg['dir'] == 'OUT']


# Wrapper bodies are indented two spaces; mirror the HIP migrator's multi-line
# enter-region layout (one snapshot per line, one _enter(...) argument per
# line) so every physical line stays well under 1024 chars and the output is
# deterministic/idempotent.
_BODY_INDENT = '  '
_ENTER_ARG_INDENT = '      '


def _build_enter_region(api, parameter_types):
    """Build the deterministically-formatted, multi-line enter region: IN
    snapshots (one per line) followed by the per-API enter call whose args are
    wrapped one per line, carrying the curated sentinel after the closing
    ``);``. No trailing newline (the caller re-emits ``\\n  `` after)."""
    decls, casts = _in_snapshots_and_casts(api, parameter_types)
    name = api['api']
    lines = [f'{_BODY_INDENT}{decl}' for decl in decls]
    if casts:
        lines.append(f'{_BODY_INDENT}rocm_trace_emit_{name}_enter(')
        for i, cast in enumerate(casts):
            terminator = ');' if i == len(casts) - 1 else ','
            lines.append(f'{_ENTER_ARG_INDENT}{cast}{terminator}')
        lines[-1] = f'{lines[-1]} /* __ROCM_CURATED__: {name} */'
    else:
        lines.append(f'{_BODY_INDENT}rocm_trace_emit_{name}_enter(); '
                     f'/* __ROCM_CURATED__: {name} */')
    return '\n'.join(lines)


def _replacement_macro(api, return_kind, expression):
    """Build the schema-v1 curated return-macro call for one return site.
    Only OUT args are forwarded (IN args go to the enter call)."""
    outs = _out_names(api)
    macro = RETURN_MACROS[return_kind]
    if not outs:
        macro = macro + '_NOARGS'
    if return_kind == 'PTR':
        prefix = f'{api["api"]}, auto, {expression}'
    else:
        prefix = f'{api["api"]}, {expression}'
    if outs:
        return f'{macro}({prefix}, {", ".join(outs)});'
    return f'{macro}({prefix});'


def _table_init_epilogue(api):
    """Direct combined-event emit for the sole pure-void wrapper
    (hsa_table_interface_init: all-IN, VOID -> exit takes no arg)."""
    return '\n  rocm_trace_emit_hsa_table_interface_init_exit();\n'


def _extract_call_expression(macro_args, return_kind):
    """Recover the wrapped call expression from an existing generic curated
    return macro's argument list."""
    parts = _split_top_level(macro_args)
    if return_kind == 'PTR':
        return parts[2].strip()  # api, ptr_type, expr, ...
    return parts[1].strip()      # api, expr, ...


def _split_top_level(text):
    parts, depth, current = [], 0, []
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


def migrate(source_path, yaml_path, header_paths, extra_args):
    raw_apis = parse_yaml_file(yaml_path)
    header_declarations = parse_headers(header_paths, extra_args)
    apis = expand_compact_apis(raw_apis, header_declarations)
    signatures = {
        api['api']: resolve_declaration(
            api['api'], header_declarations[api['api']],
            [arg['name'] for arg in api['args']])
        for api in apis
    }
    return_kinds = {
        api['api']: resolve_curated_return_kind(
            'hsa', api['api'], header_declarations[api['api']],
            [arg['name'] for arg in api['args']])
        for api in apis
    }
    source_parameters = _source_parameters(source_path,
                                           {api['api'] for api in apis},
                                           extra_args)

    with open(source_path) as f:
        source = f.read()

    edits = []
    changed = []
    for api in apis:
        name = api['api']
        return_kind = return_kinds[name]
        body_range = _find_wrapper_body(source, name)
        if body_range is None:
            raise SystemExit(f'ERROR: {name}: wrapper body not found in {source_path}')
        start, end = body_range
        body = source[start:end]
        header_params = signatures[name]
        source_params = source_parameters[name]
        if len(header_params) != len(source_params):
            raise SystemExit(
                f'ERROR: {name}: header/source parameter count differs '
                f'({len(header_params)} != {len(source_params)})')
        parameter_types = {
            header_name: (source_name, c_type)
            for (header_name, c_type, _canon, _is_enum), (source_name, _source_type)
            in zip(header_params, source_params)
        }

        expected_names = {arg['name'] for arg in api['args']}
        missing = expected_names - parameter_types.keys()
        if missing:
            raise SystemExit(f'ERROR: {name}: signature lacks curated args {sorted(missing)}')

        wrapper_edits = []

        # 1+2. Replace the ENTIRE leading enter region in ONE edit: every
        # pre-existing IN-snapshot line, the shared-generic-or-per-API-combined enter
        # call, and ALL adjacent `__ROCM_CURATED__: <name>` sentinels (even
        # split multi-line ones) for this API. Consuming the whole contiguous
        # cluster in a single replacement (rather than separate overlapping
        # per-snapshot deletions + a fixed-window sentinel search) is what
        # makes the migrator idempotent and eliminates duplicate/stale
        # sentinels — this mirrors the HIP migrator's whole-cluster approach.
        enter_region = _build_enter_region(api, parameter_types)
        # Sentinel comment for THIS api, tolerating internal whitespace/newlines
        # (a formatter may have wrapped `/* __ROCM_CURATED__:\n  <name> */`).
        sentinel_re = re.compile(
            r'/\*\s*__ROCM_CURATED__:\s*' + re.escape(name) + r'\s*\*/')
        per_api_enter = re.compile(
            r'rocm_trace_emit_' + re.escape(name) + r'_enter\s*\([^;]*\)\s*;')
        shared = SHARED_ENTER_RE.search(body)
        pa = per_api_enter.search(body)
        if shared:
            enter_match = shared
        elif pa:
            enter_match = pa
        else:
            raise SystemExit(f'ERROR: {name}: no shared or per-API enter found')

        region_start = enter_match.start()
        region_end = enter_match.end()

        # Precompute all snapshot-line ranges and all sentinel ranges (for this
        # API) once, then grow the region to absorb any that are contiguous
        # with it across whitespace only — in either direction, repeatedly.
        snapshot_ranges = [(m.start(), m.end())
                           for m in _SNAPSHOT_LINE_RE.finditer(body)]
        sentinel_ranges = [(m.start(), m.end())
                           for m in sentinel_re.finditer(body)]
        absorb = snapshot_ranges + sentinel_ranges
        grew = True
        while grew:
            grew = False
            for s, e in absorb:
                if e <= region_start and body[e:region_start].strip() == '':
                    region_start = s
                    grew = True
                elif s >= region_end and body[region_end:s].strip() == '':
                    region_end = e
                    grew = True

        # Extend region_start left over same-line leading whitespace.
        line_start = body.rfind('\n', 0, region_start) + 1
        if body[line_start:region_start].strip() == '':
            region_start = line_start

        # Normalize spacing before the next statement: consume trailing
        # horizontal whitespace + one optional newline + indent, re-emit "\n  ".
        trail = re.compile(r'[ \t]*\n?[ \t]*')
        tm = trail.match(body, region_end)
        if tm:
            region_end = tm.end()

        wrapper_edits.append((start + region_start, start + region_end,
                              enter_region + '\n' + _BODY_INDENT))

        # 3. Rewrite the return path.
        curated_matches = list(CURATED_RETURN_RE.finditer(body))
        if name == 'hsa_table_interface_init':
            # Pure-void wrapper: replace the old direct _args + shared void
            # exit with the direct _exit() emission (idempotent).
            direct_args_re = re.compile(
                r'\n?[ \t]*rocm_trace_emit_hsa_table_interface_init_args\s*\([^;]*\)\s*;',
                flags=re.DOTALL)
            handled = False
            for m in direct_args_re.finditer(body):
                wrapper_edits.append((start + m.start(), start + m.end(),
                                      '\n  rocm_trace_emit_hsa_table_interface_init_exit();'))
                handled = True
            for m in DIRECT_VOID_EXIT_RE.finditer(body):
                wrapper_edits.append((start + m.start(), start + m.end(), ''))
                handled = True
            direct_exit_re = re.compile(
                r'rocm_trace_emit_hsa_table_interface_init_exit\s*\(\s*\)\s*;')
            if not handled and not direct_exit_re.search(body):
                wrapper_edits.append((end - 1, end - 1, _table_init_epilogue(api)))
        elif curated_matches:
            for m in curated_matches:
                _kind, _noargs, macro_args = m.groups()
                expression = _extract_call_expression(macro_args, return_kind)
                wrapper_edits.append(
                    (start + m.start(), start + m.end(),
                     _replacement_macro(api, return_kind, expression)))
        else:
            raise SystemExit(f'ERROR: {name}: no recognized return macro found')

        edits.extend(wrapper_edits)
        changed.append(name)

    for s, e, replacement in sorted(edits, reverse=True):
        source = source[:s] + replacement + source[e:]
    with open(source_path, 'w') as f:
        f.write(source)
    print(f'WROTE: {len(changed)} curated wrapper updates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--curated-yaml', required=True)
    parser.add_argument('--header', action='append', required=True)
    parser.add_argument('--extra-arg', action='append', default=[])
    args = parser.parse_args()
    migrate(args.source, args.curated_yaml, args.header, args.extra_arg)


if __name__ == '__main__':
    main()
