# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Types describing the dispatch tables read from the runtime headers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class Kind(Enum):
    """The shape of a C type, as far as event fields are concerned."""

    VOID = "void"
    BOOL = "bool"
    SIGNED = "signed"
    UNSIGNED = "unsigned"
    FLOAT = "float"
    ENUM = "enum"
    POINTER = "pointer"
    RECORD = "record"
    ARRAY = "array"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CType:
    """A parameter, result, member, or array element type.

    ``spelling`` is the type as written in the header (``hipStream_t``,
    ``size_t``); the generated code uses it for declarations. ``size`` is in
    bytes.

    - Enums: ``enum_signed`` is the signedness of the underlying type, and
      ``enumerators`` lists ``(name, value)`` in declaration order.
    - Pointers: ``pointee`` is set for a parameter or result that points to a
      complete type (outputs are read through it); ``is_const_char_pointer``
      marks a ``const char*``, which RC-API-2 records as a string.
    - Records: ``members`` lists ``(name, type)`` pairs, nested records and
      arrays included; an anonymous member has the name "". ``is_union`` marks
      a union, whose members overlap.
    - Arrays: ``element`` and ``count``.
    """

    spelling: str
    kind: Kind
    size: int = 0
    enum_signed: bool = True
    enumerators: Tuple[Tuple[str, int], ...] = ()
    pointee: Optional["CType"] = None
    is_char_pointer: bool = False
    is_const_char_pointer: bool = False
    is_function_pointer: bool = False
    members: Tuple[Tuple[str, "CType"], ...] = ()
    is_union: bool = False
    element: Optional["CType"] = None
    count: int = 0


@dataclass(frozen=True)
class Param:
    """A parameter. ``declared`` is false for one named ``arg_<N>`` because
    its declaration has no name (RC-API-1)."""

    name: str
    ctype: CType
    declared: bool = True


@dataclass(frozen=True)
class Slot:
    """One function entry of a dispatch table."""

    table: str
    member: str
    api: str
    result: CType
    params: Tuple[Param, ...]


@dataclass(frozen=True)
class Table:
    name: str
    size_member: str
    slots: Tuple[Slot, ...]
    last_member: str
    reserved: Tuple[str, ...] = field(default=())
