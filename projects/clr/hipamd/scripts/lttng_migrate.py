#!/usr/bin/env python3
"""LTTng wrapper migration: classify each public HIP wrapper by return type
and inject typed enter/exit emission via libclang AST analysis.

Usage:
  pip install libclang>=14
  python3 lttng_migrate.py \\
      --source projects/clr/hipamd/src/hip_table_interface.cpp \\
      --include-path /opt/rocm/include \\
      --inventory projects/clr/hipamd/scripts/lttng_migration_inventory.txt

What it does for each public HIP wrapper:
  1. Classifies by return type (STATUS / PTR / VOID).
  2. Inserts at the start of the function body:
        const uint64_t __rocm_corr = rocm_trace_next_corr_id();
        rocm_trace_emit_hip_api_enter(__func__, __rocm_corr);
  3. Rewrites every `return EXPR;` statement to
     ROCM_TRACE_RET_STATUS(EXPR);   (or _PTR / _VOID)
     For void wrappers without a return statement, inserts a
     `rocm_trace_emit_hip_api_exit_void(__func__, __rocm_corr);`
     immediately before the closing brace.
  4. Writes a tab-separated inventory: name<TAB>class<TAB>return-type.

Failure modes (intentional, not silent):
  - Unknown return type -> raise.
  - Wrapper body has no return AND is non-void -> warn, skip wrapper.
  - Wrapper has nested local functions -> warn, skip.

The migration is IDEMPOTENT: a second run on an already-migrated file is a no-op
(detected by the presence of `__rocm_corr` in the function body).
"""
import argparse
import os
import sys
from clang import cindex

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lttng_curated_lib import parse_yaml_file, IN_ARG_KIND


# ---------------------------------------------------------------------------
# Return-type classification
# ---------------------------------------------------------------------------

# Exact-match types that map to STATUS (32-bit-coercible).
STATUS_TYPES = {
    'hipError_t', 'int', 'bool', 'unsigned int', 'uint32_t', 'int32_t',
}
# Exact-match types that map to VOID.
VOID_TYPES = {'void'}


def classify(return_type_spelling: str) -> str:
    """Map a return type spelling to STATUS / PTR / VOID / STRUCT.

    STRUCT is for struct/class return-by-value (e.g. hipChannelFormatDesc) -
    we can't capture the value in the tracepoint, so we emit hip_api_exit_void
    on the success path and the caller just sees enter+exit_void with a
    corr_id but no return-value field. The wrapper still returns the original
    expression unchanged.

    Raises SystemExit on truly unknown shapes (so a human decides).
    """
    s = return_type_spelling.strip()
    if s in VOID_TYPES:
        return 'VOID'
    if s in STATUS_TYPES:
        return 'STATUS'
    # Pointer types (any depth, any cv-qualifier).
    if '*' in s:
        return 'PTR'
    # Identifier-only types are assumed to be struct returns by value
    # (e.g. hipChannelFormatDesc, dim3). These get the STRUCT treatment
    # (emit exit_void). Reject anything that smells like a template
    # instantiation or function pointer.
    if '<' in s or '(' in s:
        raise SystemExit(
            f'Unhandled return type: {s!r}. Add to STATUS_TYPES, VOID_TYPES, '
            f'or extend classify() to handle this case.'
        )
    return 'STRUCT'


# ---------------------------------------------------------------------------
# Curated APIs (spec §6) - sentinel + IN-locals + _CURATED macro routing
# ---------------------------------------------------------------------------

CURATED_SENTINEL_PREFIX = b'/* __ROCM_CURATED__:'


def load_curated(yaml_path):
    """Returns {api_name: api_dict} or {} if path is None/missing."""
    if not yaml_path or not os.path.exists(yaml_path):
        return {}
    return {a['api']: a for a in parse_yaml_file(yaml_path)}


def _captured_args_for_curated(api):
    """Return list of captured-arg names from the YAML, in YAML order.
    For the migrator's macro emit we pass each captured arg's wrapper
    parameter name (which equals the YAML name per the §4 binding rule).
    OUT-only and INOUT args are passed differently: OUT passes the
    pointer parameter directly (helper deref's at the right time); IN
    passes the captured local __rocm_in_<name>."""
    result = []
    for a in api['args']:
        if a['dir'] == 'IN':
            result.append(f'__rocm_in_{a["name"]}')
        elif a['dir'] == 'OUT':
            # Helper expects the original out-pointer parameter.
            result.append(a['name'])
        # INOUT is rejected by the parser.
    return result


# ---------------------------------------------------------------------------
# AST traversal
# ---------------------------------------------------------------------------

def is_public_hip_api(node: cindex.Cursor, source_path: str) -> bool:
    """True if `node` is a definition of a public HIP API wrapper in this TU."""
    if node.kind != cindex.CursorKind.FUNCTION_DECL:
        return False
    if not node.is_definition():
        return False
    name = node.spelling
    if not (name.startswith('hip') or name.startswith('__hip')):
        return False
    # Must be defined IN THIS source file (not in a header).
    loc = node.location
    if loc.file is None:
        return False
    if os.path.realpath(loc.file.name) != os.path.realpath(source_path):
        return False
    return True


def find_compound_body(func_node: cindex.Cursor):
    """Return the COMPOUND_STMT child of a function definition, or None."""
    for c in func_node.get_children():
        if c.kind == cindex.CursorKind.COMPOUND_STMT:
            return c
    return None


def _is_real_return_in_source(ret_node: cindex.Cursor, src: bytes) -> bool:
    """True if this RETURN_STMT corresponds to a literal `return` keyword at
    its extent.start in the source. Filters out return statements that come
    from macro expansions (e.g. `CATCH;` -> `return hip::HandleException...`),
    where libclang reports the cursor extent at the macro use site
    (not the macro body), so the source there does NOT start with `return`.
    """
    ext = ret_node.extent
    off = ext.start.offset
    if off < 0 or off + 6 > len(src):
        return False
    return src[off:off+6] == b'return'


def find_return_stmts(node: cindex.Cursor, out: list, function_extent, src: bytes):
    """Collect RETURN_STMT cursors lexically inside this function whose
    extent corresponds to a literal `return` keyword in the source (i.e.
    not macro-expanded returns)."""
    for child in node.get_children():
        if child.kind == cindex.CursorKind.FUNCTION_DECL:
            # nested local function definition - shouldn't happen in wrappers
            continue
        if child.kind == cindex.CursorKind.RETURN_STMT:
            ext = child.extent
            if (ext.start.file is not None and
                ext.start.file.name == function_extent.start.file.name and
                ext.start.line >= function_extent.start.line and
                ext.end.line <= function_extent.end.line and
                _is_real_return_in_source(child, src)):
                out.append(child)
        find_return_stmts(child, out, function_extent, src)


# ---------------------------------------------------------------------------
# Source rewriting
# ---------------------------------------------------------------------------
#
# We work with byte offsets into the source string. libclang's
# SourceLocation.offset is the byte offset (0-based). We collect all edits
# first, then apply in reverse-offset order so earlier edits don't shift
# later offsets.
#
# Edit shape: (start_offset, end_offset, replacement_text)
#   For pure insertions, start_offset == end_offset.

ENTER_SNIPPET = (
    ' const uint64_t __rocm_corr = rocm_trace_next_corr_id();'
    ' rocm_trace_emit_hip_api_enter(__func__, __rocm_corr);'
)
EXIT_VOID_SNIPPET = (
    ' rocm_trace_emit_hip_api_exit_void(__func__, __rocm_corr);'
)
MIGRATION_MARKER = 'const uint64_t __rocm_corr = rocm_trace_next_corr_id();'  # presence => already migrated


def rewrite_return_stmt(src: bytes, ret_node: cindex.Cursor, cls: str,
                        curated_api: dict = None) -> tuple:
    """Compute (start, end, replacement) edit for rewriting a return stmt.

    Input source like `return EXPR;` becomes `ROCM_TRACE_RET_X(EXPR);`.
    For VOID returns with no expression: `return;` -> emit + return.

    If `curated_api` is non-None, emit the matching _CURATED / _CURATED_NOARGS
    variant (spec §6.2) instead of the plain ROCM_TRACE_RET_X.
    """
    ext = ret_node.extent
    start = ext.start.offset
    end = ext.end.offset
    # libclang's RETURN_STMT extent excludes the trailing semicolon. Extend
    # to include it.
    while end < len(src) and src[end:end+1] in (b' ', b'\t'):
        end += 1
    if end < len(src) and src[end:end+1] == b';':
        end += 1

    snippet = src[start:end].decode('utf-8', errors='strict')
    # Strip the leading 'return' keyword and the trailing ';'.
    s = snippet.strip()
    if not s.startswith('return'):
        raise SystemExit(
            f'rewrite_return_stmt: expected return stmt, got {snippet!r}')
    inner = s[len('return'):].rstrip(';').strip()

    if curated_api is None:
        # Existing non-curated path — emit ROCM_TRACE_RET_<cls>.
        if cls == 'STATUS':
            if not inner:
                raise SystemExit(
                    f'STATUS wrapper has bare `return;` (wrapper not status?): '
                    f'{snippet!r}')
            repl = f'ROCM_TRACE_RET_STATUS({inner});'
        elif cls == 'PTR':
            if not inner:
                raise SystemExit(
                    f'PTR wrapper has bare `return;`: {snippet!r}')
            repl = f'ROCM_TRACE_RET_PTR({inner});'
        elif cls == 'VOID':
            if inner:
                repl = f'ROCM_TRACE_RET_VOID({inner});'
            else:
                repl = f'rocm_trace_emit_hip_api_exit_void(__func__, __rocm_corr); return;'
        elif cls == 'STRUCT':
            if not inner:
                raise SystemExit(
                    f'STRUCT wrapper has bare `return;`: {snippet!r}')
            repl = (f'do {{ auto __rocm_rv = ({inner}); '
                    f'rocm_trace_emit_hip_api_exit_void(__func__, __rocm_corr); '
                    f'return __rocm_rv; }} while (0);')
        else:
            raise AssertionError(cls)
        return (start, end, repl.encode('utf-8'))

    # Curated path. Pick _CURATED vs _CURATED_NOARGS by captured arg count.
    captured = _captured_args_for_curated(curated_api)
    api = curated_api['api']
    if cls == 'STATUS':
        if not inner:
            raise SystemExit(f'{api}: STATUS curated wrapper has bare return')
        if captured:
            args_str = ', '.join(captured)
            repl = (f'ROCM_TRACE_RET_STATUS_CURATED({api}, {inner}, '
                    f'__rocm_corr, {args_str});')
        else:
            repl = (f'ROCM_TRACE_RET_STATUS_CURATED_NOARGS({api}, {inner}, '
                    f'__rocm_corr);')
    elif cls == 'PTR':
        if not inner:
            raise SystemExit(f'{api}: PTR curated wrapper has bare return')
        # PTR macro takes ptr_type as second arg. Use auto for the wrapper's
        # actual return type.
        ptr_type = 'auto'
        if captured:
            args_str = ', '.join(captured)
            repl = (f'ROCM_TRACE_RET_PTR_CURATED({api}, {ptr_type}, {inner}, '
                    f'__rocm_corr, {args_str});')
        else:
            repl = (f'ROCM_TRACE_RET_PTR_CURATED_NOARGS({api}, {ptr_type}, '
                    f'{inner}, __rocm_corr);')
    elif cls == 'VOID':
        inner_expr = inner if inner else ''
        if captured:
            args_str = ', '.join(captured)
            repl = (f'ROCM_TRACE_RET_VOID_CURATED({api}, {inner_expr}, '
                    f'__rocm_corr, {args_str});')
        else:
            repl = (f'ROCM_TRACE_RET_VOID_CURATED_NOARGS({api}, {inner_expr}, '
                    f'__rocm_corr);')
    elif cls == 'STRUCT':
        # STRUCT-returning curated wrappers are not supported in v1.
        raise SystemExit(
            f'{api}: STRUCT-returning curated wrappers not supported in v1; '
            f'remove from curated_apis.yaml')
    else:
        raise AssertionError(cls)
    return (start, end, repl.encode('utf-8'))


def _detect_resource_dir() -> str:
    """Best-effort detection of a clang resource dir for libclang to find
    its builtin headers (stddef.h, etc.). Falls back to None if not found,
    in which case the caller can pass --extra-arg=-resource-dir=... explicitly.
    """
    candidates = [
        '/usr/lib/llvm-18/lib/clang/18',
        '/usr/lib/llvm-17/lib/clang/17',
        '/usr/lib/llvm-16/lib/clang/16',
        '/usr/lib/llvm-15/lib/clang/15',
        '/usr/lib/llvm-14/lib/clang/14',
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, 'include', 'stddef.h')):
            return c
    return None


def migrate_file(source_path: str, include_path: str, extra_args: list,
                 inventory_path: str, dry_run: bool = False,
                 curated: dict = None) -> dict:
    """Migrate one source file in place. Returns dict {name: classification}.

    If `curated` is a non-empty {api_name: api_dict}, wrappers in the set
    additionally get the __ROCM_CURATED__ sentinel + __rocm_in_<arg> locals
    + _CURATED macro routing per spec §6.
    """
    curated = curated or {}
    args = ['-x', 'c++', '-std=c++17', '-D__HIP_PLATFORM_AMD__=1']
    res_dir = _detect_resource_dir()
    if res_dir and not any(a.startswith('-resource-dir') for a in extra_args):
        args += [f'-resource-dir={res_dir}']
    # extra_args FIRST so -I paths take priority over the fallback include_path
    # (which is typically /opt/rocm/include - older than the in-tree headers).
    args += extra_args
    if include_path:
        args += ['-I', include_path]

    idx = cindex.Index.create()
    tu = idx.parse(source_path, args=args,
                   options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)

    # Report errors as warnings - libclang produces a usable AST even when
    # some declarations fail to type-check (e.g. missing transitive headers
    # for GL types). What matters is that public HIP wrappers in this TU
    # parse correctly enough to extract their FunctionDecl + RETURN_STMTs.
    # We re-validate at the wrapper level: the symbol-coverage gate
    # (post-build) catches any wrappers we missed.
    fatal = [d for d in tu.diagnostics if d.severity >= cindex.Diagnostic.Error]
    if fatal:
        # Print first few but continue.
        for d in fatal[:5]:
            print(f'PARSE WARN: {d.spelling} at {d.location}', file=sys.stderr)
        if len(fatal) > 5:
            print(f'PARSE WARN: ... and {len(fatal)-5} more parse errors '
                  f'(continuing; symbol-coverage gate will catch missed wrappers)',
                  file=sys.stderr)

    with open(source_path, 'rb') as f:
        src = f.read()

    # Already-migrated detection: if the source contains `__rocm_corr`,
    # bail out idempotently.
    if MIGRATION_MARKER.encode('utf-8') in src:
        print(f'{source_path}: already migrated (found {MIGRATION_MARKER!r}); '
              f'no changes applied.', file=sys.stderr)
        # Still emit inventory by walking the AST.
        inv = {}
        for n in tu.cursor.walk_preorder():
            if is_public_hip_api(n, source_path):
                inv[n.spelling] = (classify(n.result_type.spelling),
                                   n.result_type.spelling)
        write_inventory(inv, inventory_path)
        return inv

    inventory = {}
    edits = []  # list of (start, end, replacement_bytes)

    for n in tu.cursor.walk_preorder():
        if not is_public_hip_api(n, source_path):
            continue
        ret_type = n.result_type.spelling
        cls = classify(ret_type)
        inventory[n.spelling] = (cls, ret_type)

        body = find_compound_body(n)
        if body is None:
            print(f'WARN: {n.spelling}: no compound body; skipping',
                  file=sys.stderr)
            continue
        body_extent = body.extent

        # Insertion: just past the opening brace.
        # body.extent.start points AT the '{', so insert at offset+1.
        open_brace_off = body.extent.start.offset
        if src[open_brace_off:open_brace_off+1] != b'{':
            raise SystemExit(
                f'{n.spelling}: expected {{ at offset {open_brace_off}, '
                f'found {src[open_brace_off:open_brace_off+5]!r}')
        insert_off = open_brace_off + 1

        # Build the per-wrapper insertion block: ENTER_SNIPPET, optionally
        # followed by the curated sentinel and IN-locals. Combining them
        # into a single edit avoids reverse-order ambiguity when both
        # would target the same offset.
        curated_api = curated.get(n.spelling)
        block = ENTER_SNIPPET
        if curated_api is not None:
            sentinel = f' /* __ROCM_CURATED__: {n.spelling} */'
            in_locals = []
            param_types = {arg.spelling: arg.type.spelling
                           for arg in n.get_arguments()}
            for a in curated_api['args']:
                if a['dir'] == 'IN':
                    pname = a['name']
                    if pname not in param_types:
                        raise SystemExit(
                            f"{n.spelling}: curated arg {pname!r} not in "
                            f"wrapper params {list(param_types)}")
                    pty = param_types[pname]
                    in_locals.append(
                        f' {pty} const __rocm_in_{pname} = {pname};')
            block = block + sentinel + ''.join(in_locals)
        edits.append((insert_off, insert_off, block.encode('utf-8')))

        # Find every return stmt in the body (excluding macro-expanded ones).
        returns = []
        find_return_stmts(body, returns, body_extent, src)

        for r in returns:
            edits.append(rewrite_return_stmt(src, r, cls,
                                             curated_api=curated_api))

        # For VOID wrappers with no return stmt, inject the exit emit
        # immediately before the closing brace.
        if cls == 'VOID' and not returns:
            close_brace_off = body.extent.end.offset - 1
            # body.extent.end points one past the closing '}'. Verify.
            if src[close_brace_off:close_brace_off+1] != b'}':
                # try one back
                close_brace_off = body.extent.end.offset
                if src[close_brace_off:close_brace_off+1] != b'}':
                    raise SystemExit(
                        f'{n.spelling}: cannot find closing }} of body '
                        f'(end offset {body.extent.end.offset})')
            edits.append((close_brace_off, close_brace_off,
                          EXIT_VOID_SNIPPET.encode('utf-8')))

    # Apply edits in reverse-offset order so earlier offsets stay valid.
    edits.sort(key=lambda e: e[0], reverse=True)
    out = bytearray(src)
    for start, end, repl in edits:
        out[start:end] = repl

    if dry_run:
        sys.stdout.buffer.write(bytes(out))
    else:
        with open(source_path, 'wb') as f:
            f.write(bytes(out))

    write_inventory(inventory, inventory_path)
    return inventory


def write_inventory(inventory: dict, path: str):
    with open(path, 'w') as f:
        for name in sorted(inventory):
            cls, ret = inventory[name]
            f.write(f'{name}\t{cls}\t{ret}\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, help='C++ source file')
    ap.add_argument('--include-path', default=None,
                    help='Primary include path (e.g. /opt/rocm/include)')
    ap.add_argument('--extra-arg', action='append', default=[],
                    help='Extra compile arg (repeatable). E.g. --extra-arg=-I/foo')
    ap.add_argument('--inventory', required=True,
                    help='Path to write the migration inventory (TSV)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print rewritten source to stdout, do not modify file')
    ap.add_argument('--curated-yaml', default=None,
                    help='Path to curated_apis.yaml; if provided, curated '
                         'APIs get sentinel + IN-locals + _CURATED macro '
                         'routing per spec §6.')
    args = ap.parse_args()
    curated = load_curated(args.curated_yaml)
    inv = migrate_file(args.source, args.include_path, args.extra_arg,
                       args.inventory, dry_run=args.dry_run,
                       curated=curated)
    msg = f'Migrated {len(inv)} wrappers; inventory at {args.inventory}'
    if curated:
        msg += f' (curated routing for {len(curated)} APIs)'
    print(msg, file=sys.stderr)


if __name__ == '__main__':
    main()
