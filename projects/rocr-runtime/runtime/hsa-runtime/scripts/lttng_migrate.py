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


# Status types: 32-bit-coercible. NOTE: 64-bit returns (uint64_t /
# int64_t / hsa_signal_value_t) used to live here too -- silently
# truncated to int32_t in the exit_status event. They have moved to
# U64_TYPES / I64_TYPES below so the typed exit_u64 / exit_i64 events
# preserve full 64-bit width on the wire.
STATUS_TYPES = {
    'hsa_status_t',
    'int',
    'unsigned int',
    'uint32_t',
    'int32_t',
    'bool',
}

# Unsigned 64-bit returns -- queue index ops
# (hsa_queue_load_*_index_*, hsa_queue_cas_*_index_*, hsa_queue_add_*_index_*).
U64_TYPES = {'uint64_t'}

# Signed 64-bit returns -- signal value ops
# (hsa_signal_load_*, hsa_signal_cas_*, hsa_signal_exchange_*,
# hsa_signal_wait_*). hsa_signal_value_t is int64_t in the large memory
# model (HSA_LARGE_MODEL=1, which the runtime always builds with).
I64_TYPES = {'int64_t', 'hsa_signal_value_t'}

# Void return.
VOID_TYPES = {'void'}


def classify(return_type_spelling: str) -> str:
    s = return_type_spelling.strip()
    if s in VOID_TYPES:
        return 'VOID'
    if s in STATUS_TYPES:
        return 'STATUS'
    if s in U64_TYPES:
        return 'U64'
    if s in I64_TYPES:
        return 'I64'
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
    # Include paths: the source includes "inc/hsa_api_trace.h" and
    # "core/inc/hsa_api_trace_int.h" — both relative to the hsa-runtime root.
    # source_path is .../hsa-runtime/core/common/hsa_table_interface.cpp;
    # the hsa-runtime root is two levels up.
    src_abs = os.path.abspath(source_path)
    # core/common/hsa_table_interface.cpp -> hsa-runtime
    hsa_runtime_root = os.path.dirname(os.path.dirname(os.path.dirname(src_abs)))
    args = [
        '-I', include_path,
        '-I', hsa_runtime_root,
        '-I', os.path.join(hsa_runtime_root, 'inc'),
        '-I', os.path.join(hsa_runtime_root, 'core', 'inc'),
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

        # Locate the body's closing '}' so we can inject exit_void into
        # pure-void no-return wrappers. body.extent.end points one past the
        # '}'; verify and back up by one.
        body_node = None
        for c in n.get_children():
            if c.kind == cindex.CursorKind.COMPOUND_STMT:
                body_node = c
                break
        body_close_off = None
        if body_node is not None:
            be = body_node.extent.end
            be_off = line_col_to_offset(lines, be.line, be.column)
            # libclang sometimes reports end at the char *after* '}'. Try
            # both positions.
            if be_off > 0 and be_off <= len(src) and src[be_off - 1] == '}':
                body_close_off = be_off - 1
            elif be_off < len(src) and src[be_off] == '}':
                body_close_off = be_off

        wrappers.append({
            'name': n.spelling,
            'cls': cls,
            'ret_type': ret,
            'insert_off': insert_off,
            'ret_start': ret_start_off,
            'ret_end': ret_end_off,
            'body_close': body_close_off,
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

        # 2. Replace the return stmt with the typed macro, OR (for pure-void
        # no-return wrappers) inject an exit_void emit immediately before the
        # body's closing '}' so every void wrapper has a balanced enter/exit
        # pair (fixes C1 -- previously the only no-return wrapper,
        # hsa_table_interface_init, never popped its corr_id slot).
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
            else:
                # Pure void no-return body. Inject exit_void emit just
                # before the closing '}'.
                if w['body_close'] is None:
                    sys.exit(
                        f'lttng_migrate: cannot locate closing }} for '
                        f'pure-void wrapper {w["name"]}; cannot inject exit_void')
                exit_snippet = (
                    ' rocm_trace_emit_hsa_api_exit_void(__func__, __rocm_corr);'
                )
                edits.append((w['body_close'], w['body_close'], exit_snippet))
        else:
            assert w['ret_start'] is not None, f'expected return stmt in {w["name"]}'
            ret_text = src[w['ret_start']:w['ret_end']]
            if not ret_text.startswith('return'):
                sys.exit(f'lttng_migrate: malformed return for {w["name"]}: {ret_text!r}')
            body_expr = ret_text[len('return'):].lstrip()
            body_expr = body_expr.rstrip().rstrip(';').rstrip()
            if w['cls'] == 'PTR':
                macro = 'ROCR_TRACE_API_RET_PTR'
            elif w['cls'] == 'U64':
                macro = 'ROCR_TRACE_API_RET_U64'
            elif w['cls'] == 'I64':
                macro = 'ROCR_TRACE_API_RET_I64'
            else:
                macro = 'ROCR_TRACE_API_RET_STATUS'
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
