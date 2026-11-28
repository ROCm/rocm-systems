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
_PLAIN_CHAR = {TK.CHAR_S, TK.CHAR_U}
_FUNCTION = {TK.FUNCTIONPROTO, TK.FUNCTIONNOPROTO}

# Records nest at most this deep; deeper members are unsupported.
_MAX_NESTING = 8


class ParseError(Exception):
    """The headers did not parse, or a configured table is missing."""


def _member_name(f: cindex.Cursor) -> str:
    """A member's name, or "" for an anonymous struct or union member.

    libclang spells an anonymous member "S::(anonymous union at ...)". A named
    member whose type is an unnamed struct (``struct { ... } flags;``) keeps
    its name: is_anonymous() is true for both, so the spelling decides.
    """
    spelling = f.spelling
    if not spelling or "(anonymous" in spelling or "(unnamed" in spelling:
        return ""
    return spelling


def _ctype(t: cindex.Type, follow_pointer: bool = True, depth: int = 0) -> CType:
    """Describe ``t``. ``follow_pointer`` describes a pointer's target, which
    is wanted for parameters and results (outputs are read through them) but
    not inside records or arrays (whose pointers are recorded as addresses)."""
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
        decl = c.get_declaration()
        underlying = decl.enum_type.get_canonical().kind
        enumerators = tuple(
            (e.spelling, e.enum_value)
            for e in decl.get_children()
            if e.kind == CK.ENUM_CONSTANT_DECL
        )
        return CType(
            spelling,
            Kind.ENUM,
            size,
            enum_signed=underlying in _SIGNED,
            enumerators=enumerators,
        )
    if kind == TK.POINTER:
        target = t.get_pointee()
        if target.kind == TK.INVALID:  # a typedef of a pointer, such as hipStream_t
            target = c.get_pointee()
        target_kind = target.get_canonical().kind
        pointee = _ctype(target, False, depth + 1) if follow_pointer else None
        return CType(
            spelling,
            Kind.POINTER,
            size,
            pointee=pointee,
            is_char_pointer=target_kind in _CHAR,
            is_const_char_pointer=target_kind in _PLAIN_CHAR
            and target.is_const_qualified(),
            is_function_pointer=target_kind in _FUNCTION,
        )
    if kind == TK.RECORD:
        members: Tuple[Tuple[str, CType], ...] = ()
        if size > 0 and depth < _MAX_NESTING:
            members = tuple(
                (_member_name(f), _ctype(f.type, False, depth + 1))
                for f in c.get_fields()
            )
        is_union = c.get_declaration().kind == CK.UNION_DECL
        return CType(spelling, Kind.RECORD, size, members=members, is_union=is_union)
    if kind == TK.CONSTANTARRAY and depth < _MAX_NESTING:
        element = _ctype(c.element_type, False, depth + 1)
        return CType(spelling, Kind.ARRAY, size, element=element, count=c.element_count)
    return CType(spelling, Kind.UNSUPPORTED, size)


def _declaration(
    field: cindex.Cursor, arity: int
) -> Optional[List[Tuple[str, cindex.Type]]]:
    """Parameter names and spelled types from the declaration a table member is
    typed by: HIP's ``t_<api> <api>_fn`` typedef, or the function in HSA's
    ``decltype(<api>)* <api>_fn``. Spelled types keep typedefs such as
    ``size_t``, which the prototype has lost."""
    decl: Optional[cindex.Cursor] = None
    for child in field.get_children():
        for sub in child.walk_preorder():
            if (
                sub.kind in (CK.TYPE_REF, CK.DECL_REF_EXPR)
                and sub.referenced is not None
            ):
                decl = sub.referenced
                break
        if decl is not None:
            break
    if decl is None:
        return None
    if decl.kind == CK.FUNCTION_DECL:
        params = list(decl.get_arguments())
    else:
        params = [c for c in decl.get_children() if c.kind == CK.PARM_DECL]
    if len(params) != arity:
        return None
    return [(p.spelling, p.type) for p in params]


_Signature = List[Tuple[str, str]]  # (parameter name, canonical type)


def _public_functions(tu: cindex.TranslationUnit) -> Dict[str, List[_Signature]]:
    """Parameter names and canonical types of every function with external
    linkage declared at namespace scope (including inside ``extern "C"``), by
    function name."""
    found: Dict[str, List[_Signature]] = {}
    parents = (CK.TRANSLATION_UNIT, CK.LINKAGE_SPEC)
    for c in tu.cursor.walk_preorder():
        if (
            c.kind == CK.FUNCTION_DECL
            and c.semantic_parent is not None
            and c.semantic_parent.kind in parents
            and c.linkage == cindex.LinkageKind.EXTERNAL
            and c.type.kind == TK.FUNCTIONPROTO
        ):
            found.setdefault(c.spelling, []).append(
                [
                    (a.spelling, a.type.get_canonical().spelling)
                    for a in c.get_arguments()
                ]
            )
    return found


def _parameters(
    api: str,
    field: cindex.Cursor,
    proto: cindex.Type,
    public: Dict[str, List[_Signature]],
) -> Tuple[Param, ...]:
    """The entry's parameters, named as RC-API-1 requires: by the public
    function's declaration when the headers have one with the same parameter
    types, else by the table's own declaration; a parameter with no name is
    ``arg_<N>``, 1-based."""
    declaration = _declaration(field, len(list(proto.argument_types())))
    if declaration is not None:
        types = [t for _, t in declaration]
        names = [n for n, _ in declaration]
    else:
        types = list(proto.argument_types())
        names = [""] * len(types)
    canonical = [t.get_canonical().spelling for t in types]
    for candidate in public.get(api, []):
        if [t for _, t in candidate] == canonical:
            names = [n for n, _ in candidate]
            break
    return tuple(
        Param(n, _ctype(t)) if n else Param(f"arg_{i + 1}", _ctype(t), declared=False)
        for i, (n, t) in enumerate(zip(names, types))
    )


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


def _table(
    struct: cindex.Cursor,
    name: str,
    size_member: str,
    public: Dict[str, List[_Signature]],
) -> Table:
    fields = [c for c in struct.get_children() if c.kind == CK.FIELD_DECL]
    if not fields:
        raise ParseError(f"{name} has no members")
    if size_member.split(".")[0] not in {f.spelling for f in fields}:
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
        params = _parameters(api, field, proto, public)
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
    public = _public_functions(tu)
    return [_table(structs[name], name, size, public) for name, size in tables]
