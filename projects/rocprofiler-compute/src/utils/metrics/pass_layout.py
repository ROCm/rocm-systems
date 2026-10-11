# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Counters observed per profiling pass.

Data only, so the shared analysis schema can name the layout type without
depending on metric calculation code.
"""

import re
from dataclasses import dataclass
from typing import FrozenSet, List, Mapping, Tuple

PASS_COLUMN_SEP = "@pass:"

_PASS_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def require_safe_pass_key(pass_key: str) -> str:
    """Reject a pass key that cannot be spliced into a quoted expression."""
    if _PASS_KEY_RE.fullmatch(pass_key) is None:
        raise ValueError(
            f"pass key {pass_key!r} is not safe to splice into an expression"
        )
    return pass_key


@dataclass(frozen=True)
class PassLayout:
    """Counters observed per profiling pass (from results, not perfmon YAML)."""

    pass_keys: Tuple[str, ...]
    counters_by_pass: Mapping[str, FrozenSet[str]]
    duplicated: FrozenSet[str]

    @property
    def has_duplicates(self) -> bool:
        return bool(self.duplicated)

    @property
    def is_multi_pass(self) -> bool:
        return len(self.pass_keys) > 1

    def ordinal(self, pass_key: str) -> int:
        return self.pass_keys.index(pass_key)

    def qualified_column(self, counter: str, pass_key: str) -> str:
        require_safe_pass_key(pass_key)
        return f"{counter}{PASS_COLUMN_SEP}{pass_key}"

    def passes_containing(self, counters: FrozenSet[str]) -> List[str]:
        return [
            pass_key
            for pass_key in self.pass_keys
            if counters <= self.counters_by_pass[pass_key]
        ]

    @classmethod
    def empty(cls) -> "PassLayout":
        return cls(pass_keys=(), counters_by_pass={}, duplicated=frozenset())
