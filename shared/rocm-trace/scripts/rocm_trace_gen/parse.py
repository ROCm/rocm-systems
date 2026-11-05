# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Read dispatch tables from the runtime headers with libclang."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from clang import cindex

from .model import CType, Kind, Param, Slot, Table

TK = cindex.TypeKind
CK = cindex.CursorKind

_SIGNED = {TK.CHAR_S, TK.SCHAR, TK.SHORT, TK.INT, TK.LONG, TK.LONGLONG, TK.WCHAR}
_UNSIGNED = {
    TK.CHAR_U,
    TK.UCHAR,
    TK.USHORT,
    TK.UINT,
    TK.ULONG,
    TK.ULONGLONG,
    TK.CHAR16,
    TK.CHAR32,
}
_CHAR = {TK.CHAR_S, TK.CHAR_U, TK.SCHAR, TK.UCHAR}
_FUNCTION = {TK.FUNCTIONPROTO, TK.FUNCTIONNOPROTO}

# How deep pointer targets and record members are described. Depth 0 is the
# parameter itself; its pointee or members are depth 1, and so on.
_MAX_DEPTH = 2


class ParseError(Exception):
    """The headers did not parse, or a configured table is missing."""


def _ctype(t: cindex.Type, depth: int = 0) -> CType:
    spelling = t.spelling
    c = t.get_canonical()
    size = max(c.get_size(), 0)
    kind = c.kind

    if kind == TK.VOID:
        return CType(spelling, Kind.VOID)
    if kind == TK.BOOL:
        return CType(spelling, Kind.BOOL, size)
    if kind in _SIGNED:
        return CType(spelling, Kind.SIGNED, size)
    if kind in _UNSIGNED:
        return CType(spelling, Kind.UNSIGNED, size)
    if kind in (TK.FLOAT, TK.DOUBLE):
        return CType(spelling, Kind.FLOAT, size)
    if kind == TK.ENUM:
        return CType(spelling, Kind.ENUM, size)
    if kind == TK.POINTER:
        target = t.get_pointee()
        target_kind = target.get_canonical().kind
        pointee = _ctype(target, depth + 1) if depth < _MAX_DEPTH else None
        return CType(
            spelling,
            Kind.POINTER,
            size,
            pointee=pointee,
            is_char_pointer=target_kind in _CHAR,
            is_function_pointer=target_kind in _FUNCTION,
        )
    if kind == TK.RECORD:
        members: Tuple[Tuple[str, CType], ...] = ()
        if depth < _MAX_DEPTH and size > 0:
            members = tuple(
                (f.spelling, _ctype(f.type, depth + 1)) for f in c.get_fields()
            )
        return CType(spelling, Kind.RECORD, size, members=members)
    return CType(spelling, Kind.UNSUPPORTED, size)


def _parameters(
    field: cindex.Cursor, proto: cindex.Type
) -> List[Tuple[str, cindex.Type]]:
    """Parameter names and declared types of a table member.

    HIP members are ``t_<api> <api>_fn``: the names come from the typedef. HSA
    members are ``decltype(<api>)* <api>_fn``: the names come from the function
    declaration. A parameter without a name is called ``arg<N>``.
    """
    decl: Optional[cindex.Cursor] = None
    for child in field.get_children():
        if child.kind == CK.TYPE_REF and child.referenced is not None:
            decl = child.referenced
            break
        if child.kind == CK.DECL_REF_EXPR and child.referenced is not None:
            decl = child.referenced
            break
        for sub in child.walk_preorder():
            if sub.kind == CK.DECL_REF_EXPR and sub.referenced is not None:
                decl = sub.referenced
                break
        if decl is not None:
            break

    named: List[cindex.Cursor] = []
    if decl is not None:
        if decl.kind == CK.FUNCTION_DECL:
            named = list(decl.get_arguments())
        else:
            named = [c for c in decl.get_children() if c.kind == CK.PARM_DECL]

    types = list(proto.argument_types())
    if len(named) == len(types):
        return [(p.spelling or f"arg{i}", p.type) for i, p in enumerate(named)]
    return [(f"arg{i}", t) for i, t in enumerate(types)]


def _structs(
    tu: cindex.TranslationUnit, names: Iterable[str]
) -> Dict[str, cindex.Cursor]:
    wanted = set(names)
    found: Dict[str, cindex.Cursor] = {}
    for cursor in tu.cursor.walk_preorder():
        if (
            cursor.kind == CK.STRUCT_DECL
            and cursor.spelling in wanted
            and cursor.is_definition()
            and cursor.spelling not in found
        ):
            found[cursor.spelling] = cursor
    return found


def _table(struct: cindex.Cursor, name: str, size_member: str) -> Table:
    fields = [c for c in struct.get_children() if c.kind == CK.FIELD_DECL]
    if not fields:
        raise ParseError(f"{name} has no members")
    member_names = {f.spelling for f in fields}
    if size_member.split(".")[0] not in member_names:
        raise ParseError(f"{name} has no size member '{size_member}'")

    slots: List[Slot] = []
    reserved: List[str] = []
    for field in fields:
        canonical = field.type.get_canonical()
        if canonical.kind != TK.POINTER:
            continue
        proto = canonical.get_pointee()
        if proto.kind not in _FUNCTION:
            if proto.kind == TK.VOID:
                reserved.append(field.spelling)
            continue
        member = field.spelling
        api = member[:-3] if member.endswith("_fn") else member
        params = tuple(Param(n, _ctype(t)) for n, t in _parameters(field, proto))
        slots.append(Slot(name, member, api, _ctype(proto.get_result()), params))

    if not slots:
        raise ParseError(f"{name} has no function members")
    return Table(name, size_member, tuple(slots), fields[-1].spelling, tuple(reserved))


def parse_tables(
    includes: Sequence[str],
    tables: Sequence[Tuple[str, str]],
    clang_args: Sequence[str],
) -> List[Table]:
    """Parse ``includes`` as one C++ translation unit and read ``tables``.

    ``tables`` lists ``(struct name, size member)`` pairs. Any error diagnostic
    fails the parse: a missing header silently turns types into ``int``.
    """
    source = "".join(f"#include <{inc}>\n" for inc in includes)
    args = ["-x", "c++", "-std=c++17", *clang_args]
    index = cindex.Index.create()
    tu = index.parse(
        "rocm_trace_gen_input.cpp",
        args=args,
        unsaved_files=[("rocm_trace_gen_input.cpp", source)],
        options=cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
    )

    errors = [d for d in tu.diagnostics if d.severity >= cindex.Diagnostic.Error]
    if errors:
        lines = [str(d) for d in errors[:20]]
        if any("stddef.h" in line or "stdarg.h" in line for line in lines):
            lines.append(
                "hint: libclang cannot find its builtin headers; pass "
                '--resource-dir "$(clang -print-resource-dir)"'
            )
        raise ParseError("the headers did not parse:\n  " + "\n  ".join(lines))

    structs = _structs(tu, [name for name, _ in tables])
    missing = [name for name, _ in tables if name not in structs]
    if missing:
        raise ParseError("tables not found in the headers: " + ", ".join(missing))
    return [_table(structs[name], name, size) for name, size in tables]
