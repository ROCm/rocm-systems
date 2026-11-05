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
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CType:
    """A parameter or result type.

    ``spelling`` is the type as written in the header (``hipStream_t``,
    ``size_t``); the generated code uses it for declarations. ``size`` is in
    bytes. ``pointee`` is set for pointers whose target is a complete type,
    ``members`` for records, as ``(name, type)`` pairs; a member that is an
    array or other unsupported type has kind ``UNSUPPORTED``.
    """

    spelling: str
    kind: Kind
    size: int = 0
    pointee: Optional["CType"] = None
    is_char_pointer: bool = False
    is_function_pointer: bool = False
    members: Tuple[Tuple[str, "CType"], ...] = ()


@dataclass(frozen=True)
class Param:
    name: str
    ctype: CType


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
