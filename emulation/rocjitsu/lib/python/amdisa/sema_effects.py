# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Context effects hidden behind semantic helper calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InlineOperationEffects:
    reads_scc: bool = False
    writes_scc: bool = False


_NO_EFFECTS = InlineOperationEffects()
_INLINE_BINARY_OP_EFFECTS = {
    'addc': InlineOperationEffects(reads_scc=True, writes_scc=True),
    'subb': InlineOperationEffects(reads_scc=True, writes_scc=True),
    'lshl1_add': InlineOperationEffects(writes_scc=True),
    'lshl2_add': InlineOperationEffects(writes_scc=True),
    'lshl3_add': InlineOperationEffects(writes_scc=True),
    'lshl4_add': InlineOperationEffects(writes_scc=True),
}


def inline_binary_op_effects(name: str | None) -> InlineOperationEffects:
    return _INLINE_BINARY_OP_EFFECTS.get(name or '', _NO_EFFECTS)
