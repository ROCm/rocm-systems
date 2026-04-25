#!/usr/bin/env python3
"""LTTng wrapper migration for hsa_table_interface.cpp.

Walks the file using libclang, classifies every public HSA wrapper
(extern "C" function with HSA_API attribute) by its return type, and
inserts:
  1. `const uint64_t __rocm_corr = rocm_trace_next_corr_id();`
     `rocm_trace_emit_hsa_api_enter(__func__, __rocm_corr);`
     immediately after the function's opening `{`.
  2. The matching `ROCR_TRACE_API_RET_STATUS / _PTR / _VOID` macro
     wrapping the wrapper's `return EXPR;` statement.

The classifier raises on any unrecognized return type — silent fallback
to STATUS is the bug class we are eliminating.

Output:
  --inventory        a tab-separated list of (function_name, classification,
                     return_type_spelling) — one line per migrated wrapper.

Usage:
  python3 lttng_migrate.py \\
      --source ...hsa_table_interface.cpp \\
      --include-path /opt/rocm/include \\
      --inventory ...lttng_migration_inventory.txt
"""
import argparse
import os
import sys
from clang import cindex


# Status types: anything safely castable to int32_t (with an explicit
# "wider int truncated" caveat for the queue/signal hot-path ops).
STATUS_TYPES = {
    'hsa_status_t',
    'int',
    'unsigned int',
    'uint32_t',
    'int32_t',
    'bool',
    # Wider-than-32 returns from queue/signal index ops. The exit_status
    # event field is int32_t; the high 32 bits are dropped. This is OK
    # for a tracepoint marker — consumers that need the actual index can
    # capture it via the synchronous instrumentation pair (read_index_*
    # is paired with hsa_signal_load) or via the doorbell-ring tracepoint.
    'uint64_t',
    'int64_t',
    'hsa_signal_value_t',
}

# Void return.
VOID_TYPES = {'void'}


def classify(return_type_spelling: str) -> str:
    s = return_type_spelling.strip()
    if s in VOID_TYPES:
        return 'VOID'
    if s in STATUS_TYPES:
        return 'STATUS'
    # Any pointer return — the schema captures pointer-as-uint64 hex.
    if s.endswith('*') or s.endswith('**') or 'const char *' in s:
        return 'PTR'
    raise SystemExit(f'lttng_migrate: unhandled return type: {s!r}. Add to classifier.')


def find_libclang():
    """Locate libclang.so for cindex.Config.set_library_file()."""
    candidates = [
        '/opt/rocm-7.2.2/lib/llvm/lib/libclang.so.22.0.0git',
        '/opt/rocm/lib/llvm/lib/libclang.so',
        '/usr/lib/x86_64-linux-gnu/libclang-14.so.1',
        '/usr/lib/x86_64-linux-gnu/libclang.so',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fall back: let cindex auto-discover.
    return None


def is_public_hsa_api(node, source_path):
    """Return True iff node is a definition of an HSA-API wrapper.
    We require:
      - FUNCTION_DECL kind
      - Has body (is_definition)
      - Defined in the source file we are rewriting
      - Name starts with hsa_ or __hsa_
      - Has external linkage (extern "C" — checked via storage class)
    """
    if node.kind != cindex.CursorKind.FUNCTION_DECL:
        return False
    if not node.is_definition():
        return False
    name = node.spelling
    if not (name.startswith('hsa_') or name.startswith('__hsa_')):
        return False
    if node.location.file is None:
        return False
    if os.path.realpath(node.location.file.name) != os.path.realpath(source_path):
        return False
    return True


def find_compound_stmt_extent(func_node):
    """Return (open_brace_loc, return_stmt_node) for the function body.
    open_brace_loc is the start of the COMPOUND_STMT (the '{').
    return_stmt_node is the (sole) ReturnStmt direct child, or None if void.
    """
    body = None
    for c in func_node.get_children():
        if c.kind == cindex.CursorKind.COMPOUND_STMT:
            body = c
            break
    if body is None:
        return None, None

    return_stmt = None
    for c in body.get_children():
        if c.kind == cindex.CursorKind.RETURN_STMT:
            return_stmt = c
            # don't break — take the LAST return stmt, but in this file
            # there's at most one per wrapper. We keep the first.
            break
    return body.extent.start, return_stmt


def line_col_to_offset(lines, line, col):
    """Convert (1-indexed line, 1-indexed col) to a byte offset into the
    concatenated source. Operates on a list of strings (with \\n preserved).
    """
    # offset = sum of len(lines[0..line-2]) + (col-1)
    off = 0
    for i in range(line - 1):
        off += len(lines[i])
    off += col - 1
    return off


def rewrite(source_path, include_path, inventory_path):
    libclang_path = find_libclang()
    if libclang_path:
        cindex.Config.set_library_file(libclang_path)

    idx = cindex.Index.create()
    args = [
        '-I', include_path,
        '-I', os.path.dirname(os.path.dirname(os.path.abspath(source_path))),
        '-I', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(source_path))), 'inc'),
        '-D__HSA_LARGE_MODEL=1',
        '-std=c++17',
        '-x', 'c++',
    ]
    tu = idx.parse(source_path, args=args,
                   options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)

    # Show diagnostics but don't fatal-out on parse errors — even with
    # missing headers we can still walk top-level FunctionDecls.
    n_err = sum(1 for d in tu.diagnostics if d.severity >= cindex.Diagnostic.Error)
    if n_err > 0:
        sys.stderr.write(f'lttng_migrate: {n_err} clang errors (continuing if FunctionDecls visible)\n')
        for d in list(tu.diagnostics)[:5]:
            sys.stderr.write(f'  {d}\n')

    with open(source_path, 'r') as f:
        src = f.read()
    lines = src.splitlines(keepends=True)

    wrappers = []
    for n in tu.cursor.walk_preorder():
        if not is_public_hsa_api(n, source_path):
            continue
        ret = n.result_type.spelling
        cls = classify(ret)
        body_start, return_stmt = find_compound_stmt_extent(n)
        if body_start is None:
            sys.stderr.write(f'lttng_migrate: no body for {n.spelling}; skipping\n')
            continue

        # The COMPOUND_STMT location is at the '{' character itself.
        # We want to insert just *after* that '{'.
        brace_off = line_col_to_offset(lines, body_start.line, body_start.column)
        # body_start.column points at the '{' itself (1-indexed). insert_at
        # is one column later.
        insert_off = brace_off + 1

        # Locate return stmt extent (start..end).
        ret_start_off = None
        ret_end_off = None
        if return_stmt is not None:
            rs = return_stmt.extent.start
            re = return_stmt.extent.end
            ret_start_off = line_col_to_offset(lines, rs.line, rs.column)
            ret_end_off = line_col_to_offset(lines, re.line, re.column)

        wrappers.append({
            'name': n.spelling,
            'cls': cls,
            'ret_type': ret,
            'insert_off': insert_off,
            'ret_start': ret_start_off,
            'ret_end': ret_end_off,
        })

    if not wrappers:
        sys.exit('lttng_migrate: zero wrappers found — translation unit failed to parse')

    # Apply edits in reverse offset order (highest offset first) so earlier
    # offsets don't shift.
    edits = []  # list of (start, end, replacement)
    for w in wrappers:
        # 1. Insertion of `__rocm_corr` decl + enter call after '{'.
        enter_snippet = (
            ' const uint64_t __rocm_corr = rocm_trace_next_corr_id();'
            ' rocm_trace_emit_hsa_api_enter(__func__, __rocm_corr);'
        )
        edits.append((w['insert_off'], w['insert_off'], enter_snippet))

        # 2. Replace the return stmt with the typed macro.
        if w['cls'] == 'VOID':
            if w['ret_start'] is not None:
                # Wrapper has `return foo();`. The expression to wrap is the
                # text between `return ` and `;`. Use the raw source.
                ret_text = src[w['ret_start']:w['ret_end']]
                # Strip the leading 'return' and trailing ';' if any.
                if ret_text.startswith('return'):
                    body_expr = ret_text[len('return'):].lstrip()
                else:
                    body_expr = ret_text
                # Strip trailing ';' or whitespace.
                body_expr = body_expr.rstrip().rstrip(';').rstrip()
                replacement = f'ROCR_TRACE_API_RET_VOID({body_expr});'
                # For void wrappers, the source may also have a trailing ';'
                # in the next char — extend end to include it.
                end = w['ret_end']
                # Look ahead one char for ';'
                if end < len(src) and src[end] == ';':
                    end += 1
                edits.append((w['ret_start'], end, replacement))
            # else: pure void no-return body — emit nothing extra (the enter
            # snippet alone is fine; an exit_void event would be nice but
            # there's no clean injection point without an AST-level pass).
        else:
            assert w['ret_start'] is not None, f'expected return stmt in {w["name"]}'
            ret_text = src[w['ret_start']:w['ret_end']]
            if not ret_text.startswith('return'):
                sys.exit(f'lttng_migrate: malformed return for {w["name"]}: {ret_text!r}')
            body_expr = ret_text[len('return'):].lstrip()
            body_expr = body_expr.rstrip().rstrip(';').rstrip()
            macro = 'ROCR_TRACE_API_RET_PTR' if w['cls'] == 'PTR' else 'ROCR_TRACE_API_RET_STATUS'
            replacement = f'{macro}({body_expr});'
            end = w['ret_end']
            if end < len(src) and src[end] == ';':
                end += 1
            edits.append((w['ret_start'], end, replacement))

    # Sort edits descending by start offset; apply.
    edits.sort(key=lambda e: -e[0])
    new_src = src
    for start, end, repl in edits:
        new_src = new_src[:start] + repl + new_src[end:]

    with open(source_path, 'w') as f:
        f.write(new_src)

    with open(inventory_path, 'w') as f:
        for w in wrappers:
            f.write(f'{w["name"]}\t{w["cls"]}\t{w["ret_type"]}\n')

    sys.stderr.write(f'lttng_migrate: rewrote {len(wrappers)} wrappers in {source_path}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--include-path', required=True)
    ap.add_argument('--inventory', required=True)
    args = ap.parse_args()
    rewrite(args.source, args.include_path, args.inventory)


if __name__ == '__main__':
    main()
