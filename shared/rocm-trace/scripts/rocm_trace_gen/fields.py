# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Map C types to TraceLogging field macros."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .model import CType, Kind

# A by-value struct is recorded as one field per member when it has at most
# this many members, each a number, enum, or pointer (dim3, hipExtent, ...).
MAX_RECORD_MEMBERS = 8


class Unmappable(Exception):
    """A type has no field mapping."""


@dataclass(frozen=True)
class Field:
    """One event field: ``macro(expr, "name")``.

    ``zero`` is the value of the field's type that ``expr`` is replaced with
    when the field is conditional (an output the call did not write).
    """

    macro: str
    expr: str
    name: str
    zero: str

    def render(self) -> str:
        return f'{self.macro}({self.expr}, "{self.name}")'

    def guarded(self, condition: str) -> "Field":
        return Field(
            self.macro,
            f"({condition} ? {self.expr} : {self.zero})",
            self.name,
            self.zero,
        )


def _integer(signed: bool, size: int, expr: str, name: str) -> Field:
    bits = {1: 8, 2: 16, 4: 32, 8: 64}.get(size)
    if bits is None:
        raise Unmappable(f"{size}-byte integer")
    ctype = f"{'' if signed else 'u'}int{bits}_t"
    macro = f"TraceLogging{'Int' if signed else 'UInt'}{bits}"
    return Field(macro, f"static_cast<{ctype}>({expr})", name, f"{ctype}{{0}}")


def scalar(ctype: CType, expr: str, name: str) -> Field:
    """The field for a value that is not a record."""
    kind = ctype.kind
    if kind == Kind.SIGNED:
        return _integer(True, ctype.size, expr, name)
    if kind == Kind.UNSIGNED:
        return _integer(False, ctype.size, expr, name)
    if kind == Kind.ENUM:
        return _integer(True, ctype.size if ctype.size == 8 else 4, expr, name)
    if kind == Kind.BOOL:
        return Field(
            "TraceLoggingBoolean", f"static_cast<uint8_t>({expr})", name, "uint8_t{0}"
        )
    if kind == Kind.FLOAT:
        if ctype.size == 4:
            return Field(
                "TraceLoggingFloat32", f"static_cast<float>({expr})", name, "0.0f"
            )
        if ctype.size == 8:
            return Field(
                "TraceLoggingFloat64", f"static_cast<double>({expr})", name, "0.0"
            )
        raise Unmappable(f"{ctype.size}-byte floating point")
    if kind == Kind.POINTER:
        return Field(
            "TraceLoggingPointer",
            f"reinterpret_cast<const void*>({expr})",
            name,
            "static_cast<const void*>(nullptr)",
        )
    raise Unmappable(f"'{ctype.spelling}'")


def string(expr: str, name: str) -> Field:
    """A NUL-terminated string. TraceLogging records a null pointer as ""."""
    return Field("TraceLoggingString", expr, name, "static_cast<const char*>(nullptr)")


def value(ctype: CType, expr: str, name: str) -> List[Field]:
    """The fields for a value: one, or one per member of a small struct."""
    if ctype.kind != Kind.RECORD:
        return [scalar(ctype, expr, name)]
    if not ctype.members:
        raise Unmappable(f"struct '{ctype.spelling}' (incomplete or too deep)")
    if len(ctype.members) > MAX_RECORD_MEMBERS:
        raise Unmappable(
            f"struct '{ctype.spelling}' ({len(ctype.members)} members; at most "
            f"{MAX_RECORD_MEMBERS} are recorded one per field)"
        )
    fields: List[Field] = []
    for member, mtype in ctype.members:
        if mtype.kind in (Kind.RECORD, Kind.UNSUPPORTED, Kind.VOID):
            raise Unmappable(
                f"struct '{ctype.spelling}' (member '{member}' is '{mtype.spelling}')"
            )
        fields.append(scalar(mtype, f"{expr}.{member}", f"{name}_{member}"))
    return fields
