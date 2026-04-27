#!/usr/bin/env python3
"""LTTng coverage-gate Python helpers.

Driven by lttng_coverage_gate.sh. Each subcommand corresponds to a
distinct gate the shell script invokes; keeping them here (rather than
embedded as heredocs) keeps the shell side short and the Python side
linter-/test-friendly.

Subcommands:
  list-curated  Print one curated API name per line from curated_apis.yaml.
                Used by the shell script to feed the curated-vs-inventory
                set diff that runs before the body-content scan.

  body          Body-content gate (defense-in-depth). For each migrated
                symbol name listed in --names-file, locate its function
                body under --src-dir and require:
                  - >=1 rocm_trace_emit_hip_api_enter call
                  - AND (>=1 ROCM_TRACE_RET_*/ROCR_TRACE_API_RET_* macro
                    OR    >=1 rocm_trace_emit_hip_api_exit_* call)
                Symbols whose definition lives in a TU not touched by
                the migrator (no '__rocm_corr' marker in any candidate
                body) are reported as 'not-migrated-but-listed' and do
                not fail the gate; the symbol-coverage gate vouches for
                them upstream.

  curated       Curated-args body-content gate. For each API listed in
                --yaml, locate the wrapper body that contains the
                '/* __ROCM_CURATED__: <name> */' sentinel and require:
                  - a curated _CURATED macro invocation
                  - if the API has at least one IN/INOUT arg, at least
                    one '__rocm_in_' local in the body

Exit codes (per subcommand):
  0  PASS
  1  FAIL  -- gate found a gap
  2  USAGE -- bad arguments / missing files
"""
import argparse
import os
import re
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_cpp_files(src_dir):
    """Return list of (path, text) tuples for every .cpp under src_dir."""
    out = []
    for root, _, fnames in os.walk(src_dir):
        for fn in fnames:
            if fn.endswith('.cpp'):
                p = os.path.join(root, fn)
                with open(p, 'rb') as fh:
                    out.append((p, fh.read().decode('utf-8', errors='replace')))
    return out


def _iter_function_bodies(text, name):
    """Yield each function-definition body whose signature begins with
    `name(...)` and is followed by '{ ... }'. There may be multiple
    candidates (forward decls + definitions, helper TUs); the caller
    picks among them.
    """
    pat = re.compile(r'\b' + re.escape(name) + r'\s*\(')
    for m in pat.finditer(text):
        depth = 0
        i = m.end() - 1  # at '('
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
                                yield text[j:k + 1]
                                break
                        k += 1
                    break
            i += 1


def _find_first_body(text, name):
    """Return the first body for `name` in `text`, or None."""
    for body in _iter_function_bodies(text, name):
        return body
    return None


# ---------------------------------------------------------------------------
# Subcommand: list-curated
# ---------------------------------------------------------------------------

def gate_list_curated(args):
    if not os.path.isfile(args.yaml):
        print(f'ERROR: yaml not found: {args.yaml}', file=sys.stderr)
        return 2
    # Local import: lttng_curated_lib lives next to this script.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from lttng_curated_lib import parse_yaml_file
    for a in parse_yaml_file(args.yaml):
        print(a['api'])
    return 0


# ---------------------------------------------------------------------------
# Subcommand: body
# ---------------------------------------------------------------------------

_BODY_ENTER_RE = re.compile(r'rocm_trace_emit_hip_api_enter\s*\(')
_BODY_EXIT_MACRO_RE = re.compile(
    r'\b(ROCM_TRACE_RET_[A-Z0-9_]+|ROCR_TRACE_API_RET_[A-Z0-9_]+)\s*\(')
_BODY_EXIT_FN_RE = re.compile(r'rocm_trace_emit_hip_api_exit_[a-z0-9_]+\s*\(')

# Presence in a candidate body => this is the migrated wrapper (vs. a
# forward decl or pre-migration helper).
_BODY_MIGRATION_MARKER = '__rocm_corr'


def gate_body(args):
    if not os.path.isdir(args.src_dir):
        print(f'ERROR: src-dir not found: {args.src_dir}', file=sys.stderr)
        return 2
    if not os.path.isfile(args.names_file):
        print(f'ERROR: names-file not found: {args.names_file}', file=sys.stderr)
        return 2

    with open(args.names_file) as f:
        names = [ln.strip() for ln in f if ln.strip()]

    files = _collect_cpp_files(args.src_dir)

    bad = []
    not_found = []
    unmigrated = []
    for n in names:
        # Find every candidate body and pick the one that contains the
        # migration marker. If no body has it, the symbol is in the
        # inventory but its source-of-truth definition lives in a TU not
        # touched by the migrator -- skip the body-content check (the
        # symbol-coverage gate above already vouched for it).
        chosen = None
        for _path, txt in files:
            for body in _iter_function_bodies(txt, n):
                if _BODY_MIGRATION_MARKER in body:
                    chosen = body
                    break
            if chosen:
                break
        if chosen is None:
            unmigrated.append(n)
            continue
        has_enter = bool(_BODY_ENTER_RE.search(chosen))
        has_exit = bool(_BODY_EXIT_MACRO_RE.search(chosen)
                        or _BODY_EXIT_FN_RE.search(chosen))
        if not (has_enter and has_exit):
            bad.append((n, has_enter, has_exit))

    for n, e, x in bad:
        print(f'  body-content miss: {n}: enter={e} exit={x}')
    print(f'BODY: {len(bad)} body-content gaps; '
          f'{len(unmigrated)} not-migrated-but-listed; '
          f'{len(not_found)} not_found')
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Subcommand: curated
# ---------------------------------------------------------------------------

# Regex matcher for all six HIP curated-macro variants plus the HSA mirror
# (so the same gate works for hsa_table_interface.cpp).
_CURATED_MACRO_RE = re.compile(
    r'(?:ROCM_TRACE_RET|ROCR_TRACE_API_RET)_(?:STATUS|PTR|VOID)_CURATED'
    r'(?:_HSA)?(?:_NOARGS)?\s*\(')


def gate_curated(args):
    if not os.path.isdir(args.src_dir):
        print(f'ERROR: src-dir not found: {args.src_dir}', file=sys.stderr)
        return 2
    if not os.path.isfile(args.yaml):
        print(f'ERROR: yaml not found: {args.yaml}', file=sys.stderr)
        return 2

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from lttng_curated_lib import parse_yaml_file

    apis = parse_yaml_file(args.yaml)
    files = _collect_cpp_files(args.src_dir)

    failures = []
    for api in apis:
        name = api['api']
        sentinel = f'/* __ROCM_CURATED__: {name} */'
        body = None
        for _path, text in files:
            b = _find_first_body(text, name)
            if b and sentinel in b:
                body = b
                break
        if body is None:
            failures.append(
                f'{name}: no wrapper body found containing sentinel {sentinel!r}')
            continue
        if not _CURATED_MACRO_RE.search(body):
            failures.append(
                f'{name}: sentinel present but no _CURATED macro invocation')
            continue
        # IN-local check, only when the API has at least one IN/INOUT arg.
        has_in = any(a['dir'] in ('IN', 'INOUT') for a in api['args'])
        if has_in and '__rocm_in_' not in body:
            failures.append(
                f'{name}: has IN args but no __rocm_in_ locals in body')

    if failures:
        for f in failures:
            print(f'  CURATED FAIL: {f}')
        return 1
    print(f'CURATED: {len(apis)} curated APIs verified')
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='gate', required=True)

    p_list = sub.add_parser(
        'list-curated', help='Print curated API names from a YAML file.')
    p_list.add_argument('--yaml', required=True)
    p_list.set_defaults(func=gate_list_curated)

    p_body = sub.add_parser('body', help='Body-content gate.')
    p_body.add_argument('--src-dir', required=True)
    p_body.add_argument('--names-file', required=True)
    p_body.set_defaults(func=gate_body)

    p_cur = sub.add_parser('curated', help='Curated-args body-content gate.')
    p_cur.add_argument('--src-dir', required=True)
    p_cur.add_argument('--yaml', required=True)
    p_cur.set_defaults(func=gate_curated)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
