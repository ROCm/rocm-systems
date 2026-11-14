# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Map C types to TraceLogging field macros.

Every builder takes an optional ``cond``: a C++ condition under which the
value is recorded. When it is false the field records zero (or an empty
string), so an event has the same fields whatever is captured; one event name
must keep one layout, or LTTng corrupts the trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .model import CType, Kind

STRING_CAP = "ROCM_TRACE_STRING_CAP"


class Unmappable(Exception):
    """A type has no field mapping."""


Path = Tuple[Union[str, int], ...]


@dataclass(frozen=True)
class Field:
    """One event field: ``macro(args..., "name")``.

    The rest describes the field for the probe schema: ``ctype`` is the C type
    it records (an array or union for the fixed-array fields), ``path`` the
    members and indices from the parameter to it, ``param`` and ``param_type``
    the parameter (or "result", "exception") and its type, ``role`` one of "argument", "output", "result",
    "exception", and ``capture`` the lowest ROCM_TRACE_CAPTURE level that
    records it ("none", "scalars", "full"), with ``on_success`` for outputs.
    """

    macro: str
    args: Tuple[str, ...]
    name: str
    ctype: Optional[CType] = None
    path: Path = ()
    param: str = ""
    param_type: str = ""
    role: str = ""
    capture: str = "none"
    on_success: bool = False

    def render(self) -> str:
        return f'{self.macro}({", ".join(self.args)}, "{self.name}")'


def _guard(cond: Optional[str], expr: str, zero: str) -> str:
    return expr if cond is None else f"({cond} ? {expr} : {zero})"


def _integer_type(signed: bool, size: int) -> Tuple[str, str]:
    """(C type, TraceLogging type name) of an integer of ``size`` bytes."""
    bits = {1: 8, 2: 16, 4: 32, 8: 64}.get(size)
    if bits is None:
        raise Unmappable(f"{size}-byte integer")
    return f"{'' if signed else 'u'}int{bits}_t", f"{'Int' if signed else 'UInt'}{bits}"


def _scalar_type(ctype: CType) -> Tuple[str, str]:
    """(C type the value is cast to, TraceLogging type name) for a scalar."""
    kind = ctype.kind
    if kind in (Kind.SIGNED, Kind.UNSIGNED):
        return _integer_type(kind == Kind.SIGNED, ctype.size)
    if kind == Kind.ENUM:
        return _integer_type(ctype.enum_signed, ctype.size)
    if kind == Kind.BOOL:
        return "uint8_t", "Boolean"
    if kind == Kind.FLOAT:
        if ctype.size == 4:
            return "float", "Float32"
        if ctype.size == 8:
            return "double", "Float64"
        raise Unmappable(f"{ctype.size}-byte floating point")
    if kind == Kind.POINTER:
        return "const void*", "Pointer"
    raise Unmappable(f"'{ctype.spelling}'")


def scalar(
    ctype: CType, expr: str, name: str, cond: Optional[str] = None, path: Path = ()
) -> Field:
    """The field for a number, enum, bool, or pointer."""
    c_type, tl_type = _scalar_type(ctype)
    if ctype.kind == Kind.POINTER:
        value = f"reinterpret_cast<const void*>({expr})"
        zero = "static_cast<const void*>(nullptr)"
    else:
        value = f"static_cast<{c_type}>({expr})"
        zero = f"{c_type}{{}}"
    return Field(
        f"TraceLogging{tl_type}", (_guard(cond, value, zero),), name, ctype, path
    )


def string(
    expr: str,
    name: str,
    capped: bool = True,
    cond: Optional[str] = None,
    ctype: Optional[CType] = None,
) -> Field:
    """A NUL-terminated string; a null pointer records "". Capped strings
    record at most ROCM_TRACE_STRING_CAP bytes (RC-GEN-8)."""
    pointer = _guard(cond, expr, "static_cast<const char*>(nullptr)")
    if not capped:
        return Field("TraceLoggingString", (pointer,), name, ctype)
    return Field(
        "TraceLoggingCountedString",
        (f"text({pointer})", f"text_size({pointer}, {STRING_CAP})"),
        name,
        ctype,
    )


def _array(
    ctype: CType, expr: str, name: str, cond: Optional[str], path: Path
) -> List[Field]:
    element = ctype.element
    assert element is not None
    # One set of fields per element for structs and nested arrays, and for floating point:
    # the LTTng flavour of TraceLogging has no floating-point array fields.
    if element.kind in (Kind.RECORD, Kind.ARRAY, Kind.FLOAT):
        fields: List[Field] = []
        for i in range(ctype.count):
            fields.extend(
                value(element, f"{expr}[{i}]", f"{name}_{i}", cond, path + (i,))
            )
        return fields
    c_type, tl_type = _scalar_type(element)
    if element.kind == Kind.POINTER:
        # The Windows SDK's TraceLoggingPointerFixedArray takes void* const*; the
        # LTTng flavour's takes void const* const*, to which this converts.
        c_type = "void*"
        pointer = f"reinterpret_cast<void* const*>({expr})"
    else:
        pointer = f"reinterpret_cast<const {c_type}*>({expr})"
    zero = f"zeros<{c_type}, {ctype.count}>()"
    return [
        Field(
            f"TraceLogging{tl_type}FixedArray",
            (_guard(cond, pointer, zero), str(ctype.count)),
            name,
            ctype,
            path,
        )
    ]


def value(
    ctype: CType, expr: str, name: str, cond: Optional[str] = None, path: Path = ()
) -> List[Field]:
    """The fields for a value: one for a scalar, one per component for a
    struct or array (RC-API-1: all components of an aggregate)."""
    if ctype.kind == Kind.ARRAY:
        return _array(ctype, expr, name, cond, path)
    if ctype.kind != Kind.RECORD:
        return [scalar(ctype, expr, name, cond, path)]
    if not ctype.members:
        raise Unmappable(
            f"struct '{ctype.spelling}' (incomplete, or nested too deeply)"
        )
    if ctype.is_union:
        return [_union_bytes(ctype, f"&({expr})", name, cond, path)]
    fields: List[Field] = []
    for member, mtype in ctype.members:
        if mtype.kind in (Kind.UNSUPPORTED, Kind.VOID):
            raise Unmappable(
                f"struct '{ctype.spelling}' (member '{member}' is '{mtype.spelling}')"
            )
        if member:
            fields.extend(
                value(
                    mtype,
                    f"{expr}.{member}",
                    f"{name}_{member}",
                    cond,
                    path + (member,),
                )
            )
        elif mtype.is_union:
            # An anonymous union starts where its first member does.
            first = mtype.members[0][0] if mtype.members else ""
            if not first:
                raise Unmappable(f"struct '{ctype.spelling}' (an unnamed nested union)")
            fields.append(
                _union_bytes(mtype, f"&({expr}.{first})", f"{name}_union", cond, path)
            )
        else:
            # An anonymous struct: its members belong to the enclosing one.
            fields.extend(value(mtype, expr, name, cond, path))
    return fields


def _union_bytes(
    ctype: CType, address: str, name: str, cond: Optional[str], path: Path
) -> Field:
    """A union's bytes: which member is active is unknown."""
    pointer = f"reinterpret_cast<const uint8_t*>({address})"
    zero = f"zeros<uint8_t, {ctype.size}>()"
    return Field(
        "TraceLoggingUInt8FixedArray",
        (_guard(cond, pointer, zero), str(ctype.size)),
        name,
        ctype,
        path,
    )
