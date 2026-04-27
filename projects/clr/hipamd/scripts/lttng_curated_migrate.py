#!/usr/bin/env python3
"""Curated-args migrator: applies curated transforms to already-migrated source.

Idempotent: skips wrappers that already have the __ROCM_CURATED__ sentinel.

For each wrapper named in --curated-yaml whose body in --source contains
the provider's ENTER snippet but NOT the curated sentinel:
  1. Insert sentinel + IN-locals right after the ENTER_SNIPPET.
  2. Rewrite the wrapper's existing macro family (ROCM_TRACE_RET_* for HIP,
     ROCR_TRACE_API_RET_* for HSA) to the matching _CURATED / _CURATED_HSA
     (and _NOARGS) variant.

Provider-agnostic: takes --provider hip|hsa and selects:
- enter-helper regex (rocm_trace_emit_hip_api_enter vs rocm_trace_emit_hsa_api_enter)
- existing macro family regex
- emitted curated macro names
- default include flags (HIP needs -D__HIP_PLATFORM_AMD__=1)

Usage:
    python3 lttng_curated_migrate.py \\
        --provider hip \\
        --source path/to/hip_table_interface.cpp \\
        --curated-yaml path/to/curated_apis.yaml \\
        --include-path /opt/rocm/include
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lttng_curated_lib import parse_yaml_file
from clang import cindex


# Cast templates per DSL type (must match HELPER_PARAM_TYPE in
# lttng_curated_codegen.py — the helper's formal-param types). Mirrors
# the table in lttng_migrate.py so the two scripts produce identical
# call-site expressions.
_CURATED_IN_CAST_TMPL = {
    'handle':      '(uint64_t)(uintptr_t)({arg})',
    # `ptr` covers everything from `void*` to function pointers to
    # struct pointers (e.g. `hipMemcpy3DParms*`). Cast through
    # `(const void*)(uintptr_t)` so all pointer flavors — including
    # function pointers — convert without a C++ type-checker complaint.
    'ptr':         '(const void*)(uintptr_t)({arg})',
    'device_ptr':  '(uint64_t)({arg})',
    'size':        '({arg})',
    'int32':       '({arg})',
    'uint32':      '({arg})',
    'int64':       '({arg})',
    'uint64':      '({arg})',
    'float':       '({arg})',
    'enum':        '(int32_t)({arg})',
    'bool':        '({arg})',
    # `cstring` is `const char*`; a plain (const char*) cast handles
    # any const variants of the source.
    'cstring':     '(const char*)({arg})',
    'dim3':        '({arg})',
    'dim3_packed': '({arg})',
}

# Provider-specific overrides for the IN-cast templates. HIP handles
# are pointer typedefs (e.g. `hipStream_t = ihipStream_t*`), so the
# uintptr_t cast in the default table above works. HSA handles are
# value-type structs with a single `.handle` uint64_t member; reading
# that member is the canonical way to get the integer-valued handle
# (cf. HSA spec §2.3 "opaque handles"), and the (uintptr_t) cast on
# the struct itself is not legal.
_CURATED_IN_CAST_OVERRIDES = {
    'hsa': {
        'handle': '(uint64_t)(({arg}).handle)',
    },
}


def _captured_arg_expr(arg, ident, provider='hip'):
    """Cast `ident` (typically '__rocm_in_<name>' or the OUT param name)
    to the helper's formal-param type per the DSL `arg`."""
    overrides = _CURATED_IN_CAST_OVERRIDES.get(provider, {})
    tmpl = overrides.get(arg['type'])
    if tmpl is None:
        tmpl = _CURATED_IN_CAST_TMPL.get(arg['type'], '({arg})')
    return tmpl.format(arg=ident)


# Provider-specific configuration. Adding a new provider means adding one
# entry here and updating the --provider choices.
PROVIDER_CONFIG = {
    'hip': {
        'enter_re': re.compile(
            r'__rocm_corr\s*=\s*rocm_trace_next_corr_id\(\)\s*;'
            r'\s*rocm_trace_emit_hip_api_enter\([^)]*\)\s*;'),
        # Match ONLY the non-curated forms — must not re-match _CURATED on a
        # second pass. Use a negative lookahead.
        'ret_re': re.compile(
            r'ROCM_TRACE_RET_(STATUS|PTR|VOID)(?!_CURATED)\s*\(\s*([^;]+?)\s*\)\s*;',
            flags=re.DOTALL),
        'curated_status':         'ROCM_TRACE_RET_STATUS_CURATED',
        'curated_status_noargs':  'ROCM_TRACE_RET_STATUS_CURATED_NOARGS',
        'curated_ptr':            'ROCM_TRACE_RET_PTR_CURATED',
        'curated_ptr_noargs':     'ROCM_TRACE_RET_PTR_CURATED_NOARGS',
        'curated_void':           'ROCM_TRACE_RET_VOID_CURATED',
        'curated_void_noargs':    'ROCM_TRACE_RET_VOID_CURATED_NOARGS',
        'default_extra_args': ['-D__HIP_PLATFORM_AMD__=1'],
    },
    'hsa': {
        'enter_re': re.compile(
            r'__rocm_corr\s*=\s*rocm_trace_next_corr_id\(\)\s*;'
            r'\s*rocm_trace_emit_hsa_api_enter\([^)]*\)\s*;'),
        'ret_re': re.compile(
            r'ROCR_TRACE_API_RET_(STATUS|PTR|VOID)(?!_CURATED)\s*\(\s*([^;]+?)\s*\)\s*;',
            flags=re.DOTALL),
        'curated_status':         'ROCR_TRACE_API_RET_STATUS_CURATED_HSA',
        'curated_status_noargs':  'ROCR_TRACE_API_RET_STATUS_CURATED_HSA_NOARGS',
        'curated_ptr':            'ROCR_TRACE_API_RET_PTR_CURATED_HSA',
        'curated_ptr_noargs':     'ROCR_TRACE_API_RET_PTR_CURATED_HSA_NOARGS',
        'curated_void':           'ROCR_TRACE_API_RET_VOID_CURATED_HSA',
        'curated_void_noargs':    'ROCR_TRACE_API_RET_VOID_CURATED_HSA_NOARGS',
        'default_extra_args': [],
    },
}


def _sentinel_re(api):
    return re.compile(rf'/\* __ROCM_CURATED__: {re.escape(api)} \*/')


def find_wrapper_body(text, fn_name):
    """Return (body_start, body_end) byte offsets of the wrapper body, or
    None. body_start points at the `{`, body_end is one past the `}`.

    Only matches a definition whose `(` is followed (after balanced parens)
    by a `{`. Skips forward declarations.
    """
    pat = re.compile(r'\b' + re.escape(fn_name) + r'\s*\(')
    for m in pat.finditer(text):
        # Walk balanced parens from the opening '('.
        i = m.end() - 1  # at '('
        depth = 0
        while i < len(text):
            c = text[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            continue
        # After the closing ')', skip whitespace + qualifiers; require '{'.
        j = i + 1
        while j < len(text) and text[j] in ' \t\n\r':
            j += 1
        if j >= len(text) or text[j] != '{':
            continue  # forward decl or trailing-attr decl, not a definition
        # Walk balanced braces.
        body_start = j
        bdepth = 0
        k = j
        while k < len(text):
            if text[k] == '{':
                bdepth += 1
            elif text[k] == '}':
                bdepth -= 1
                if bdepth == 0:
                    return (body_start, k + 1)
            k += 1
    return None


def overlay(provider, source_path, yaml_path, include_path, extra_includes=None):
    cfg = PROVIDER_CONFIG[provider]
    apis = {a['api']: a for a in parse_yaml_file(yaml_path)}

    # Use libclang to get param-type info for wrapper signatures.
    args = ['-x', 'c++', '-std=c++17', '-I', include_path] + cfg['default_extra_args']
    if extra_includes:
        for p in extra_includes:
            args += ['-I', p]
    idx = cindex.Index.create()
    tu = idx.parse(source_path, args=args)
    param_types = {}  # fn_name -> {pname: ptype}
    for n in tu.cursor.walk_preorder():
        if (n.kind == cindex.CursorKind.FUNCTION_DECL and n.is_definition()
                and n.spelling in apis):
            param_types[n.spelling] = {a.spelling: a.type.spelling
                                       for a in n.get_arguments()}

    with open(source_path, 'r') as f:
        src = f.read()
    edits = []  # (start, end, replacement)
    for fn, api in apis.items():
        if _sentinel_re(fn).search(src):
            print(f'  {fn}: already overlaid; skip', file=sys.stderr)
            continue
        body = find_wrapper_body(src, fn)
        if body is None:
            print(f'  {fn}: body not found in {source_path}; skip',
                  file=sys.stderr)
            continue
        bstart, bend = body
        body_text = src[bstart:bend]
        # Find provider-specific ENTER snippet in body to anchor the
        # sentinel insertion.
        em = cfg['enter_re'].search(body_text)
        if em is None:
            print(f'  {fn}: ENTER snippet not found in body; skip',
                  file=sys.stderr)
            continue
        insert_off = bstart + em.end()

        # Build sentinel + IN-locals.
        sentinel = f' /* __ROCM_CURATED__: {fn} */'
        in_locals = []
        ptypes = param_types.get(fn, {})
        for a in api['args']:
            if a['dir'] == 'IN':
                pname = a['name']
                if pname not in ptypes:
                    sys.exit(f'{fn}: arg {pname!r} not in wrapper params '
                             f'{list(ptypes)}')
                # Use 'auto const' rather than the libclang-resolved type:
                # when system headers are not available, libclang may
                # resolve typedefs like size_t to a narrower type (e.g.
                # int), which would silently truncate the value at the
                # assignment line. C++14+ type deduction preserves the
                # wrapper's actual parameter type.
                in_locals.append(f' auto const __rocm_in_{pname} = {pname};')
        insertion = sentinel + ''.join(in_locals)
        edits.append((insert_off, insert_off, insertion))

        # Rewrite the existing macro family to its _CURATED variant.
        for m in cfg['ret_re'].finditer(body_text):
            cls, expr = m.group(1), m.group(2)
            macro_start = bstart + m.start(0)
            macro_end = bstart + m.end(0)
            # Build captured-args list. IN args may need a cast (e.g.
            # handle → uint64_t) to match the helper's formal-param type.
            # OUT args are passed by-pointer (helper deref's), no cast.
            captured = []
            for a in api['args']:
                if a['dir'] == 'IN':
                    captured.append(
                        _captured_arg_expr(a, f'__rocm_in_{a["name"]}',
                                           provider=provider))
                elif a['dir'] == 'OUT':
                    # OUT args are passed by-pointer; helper deref's the
                    # pointer to read the value at exit time. Most OUT
                    # types use the source's pointer-to-T directly (e.g.
                    # `unsigned int *` for OUT uint32 → helper `uint32_t*`
                    # is layout-compatible). For OUT enum the helper
                    # signature is `int32_t*` while the source is
                    # `<some_typedef>*`, so cast to keep the C++ type
                    # checker happy. Same for OUT bool (`int*`).
                    if a['type'] == 'enum':
                        captured.append(f'(int32_t*)({a["name"]})')
                    elif a['type'] == 'bool':
                        captured.append(f'(int*)({a["name"]})')
                    else:
                        captured.append(a['name'])
            if cls == 'STATUS':
                if captured:
                    repl = (f'{cfg["curated_status"]}({fn}, {expr}, '
                            f'__rocm_corr, {", ".join(captured)});')
                else:
                    repl = (f'{cfg["curated_status_noargs"]}({fn}, {expr}, '
                            f'__rocm_corr);')
            elif cls == 'PTR':
                if captured:
                    repl = (f'{cfg["curated_ptr"]}({fn}, auto, {expr}, '
                            f'__rocm_corr, {", ".join(captured)});')
                else:
                    repl = (f'{cfg["curated_ptr_noargs"]}({fn}, auto, {expr}, '
                            f'__rocm_corr);')
            elif cls == 'VOID':
                if captured:
                    repl = (f'{cfg["curated_void"]}({fn}, {expr}, '
                            f'__rocm_corr, {", ".join(captured)});')
                else:
                    repl = (f'{cfg["curated_void_noargs"]}({fn}, {expr}, '
                            f'__rocm_corr);')
            else:
                continue
            edits.append((macro_start, macro_end, repl))

    if not edits:
        print('no edits to apply', file=sys.stderr)
        return 0
    edits.sort(key=lambda e: e[0], reverse=True)
    out = list(src)
    for start, end, repl in edits:
        out[start:end] = repl
    with open(source_path, 'w') as f:
        f.write(''.join(out))
    print(f'applied {len(edits)} edits', file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--provider', required=True,
                    choices=sorted(PROVIDER_CONFIG))
    ap.add_argument('--source', required=True)
    ap.add_argument('--curated-yaml', required=True)
    ap.add_argument('--include-path', default='/opt/rocm/include')
    ap.add_argument('--extra-include', action='append', default=[],
                    help='Additional -I path; may be repeated. Useful '
                         'for HSA wrappers that #include "inc/..." paths '
                         'relative to the project source root.')
    args = ap.parse_args()
    sys.exit(overlay(args.provider, args.source, args.curated_yaml,
                     args.include_path, args.extra_include))


if __name__ == '__main__':
    main()
