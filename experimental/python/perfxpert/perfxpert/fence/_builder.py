"""FenceBuilder — assemble per-role fence text from slices + YAML excerpts.

Deterministic: build(role, bottleneck, gfx_id) returns bit-identical
output across calls. LRU-cached on the full argument tuple.

Size constraint: every slice file is ≤ 400 lines; the full assembled
fence for a role is ≤ 60 KB (enforced by test_determinism.py).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

_SLICES_DIR = Path(__file__).parent / "slices"

_ROLES = frozenset(
    [
        "root",
        "analysis",
        "recommendation",
        "correctness",
        "compute_specialist",
        "memory_specialist",
        "latency_specialist",
    ]
)


def _load_slice(name: str) -> str:
    path = _SLICES_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"fence slice missing: {path}")
    return path.read_text()


@lru_cache(maxsize=64)
def _build_cached(role: str, bottleneck: Optional[str], gfx_id: Optional[str]) -> str:
    if role not in _ROLES:
        known = ", ".join(sorted(_ROLES))
        raise KeyError(f"unknown fence role {role!r}; known: {known}")

    sections = [_load_slice("always"), _load_slice(role)]

    # Import filters lazily — avoids circular import during package init
    from perfxpert.fence._filters import (
        format_yaml_excerpt,
        get_yaml_keys_for_role,
    )

    for yaml_name, keys in get_yaml_keys_for_role(role).items():
        excerpt = format_yaml_excerpt(yaml_name, keys, gfx_id=gfx_id, bottleneck=bottleneck)
        if excerpt:
            sections.append(excerpt)

    return "\n\n".join(sections)


class FenceBuilder:
    """Assemble fence text for a given agent role."""

    def build(
        self,
        agent_role: str,
        *,
        bottleneck: Optional[str] = None,
        gfx_id: Optional[str] = None,
    ) -> str:
        return _build_cached(agent_role, bottleneck, gfx_id)


__all__ = ["FenceBuilder"]
