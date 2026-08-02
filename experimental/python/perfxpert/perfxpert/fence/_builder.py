"""FenceBuilder — assemble per-role fence text from slices + YAML excerpts.

The canonical slices are ``perfxpert/agents/fence/*.md`` — the same files the
agents load. This module previously kept its own copies under
``perfxpert/fence/slices/``, which drifted from the live fences and omitted
the Trace-Diff specialist entirely, so nothing that used the builder matched
what an agent actually received.

``compose_prompt(role)`` is the single composition path. ``framework.Agent``
calls it at construction time, so the shared ``always.md`` policy reaches
every live prompt exactly once.

Deterministic: build(role, bottleneck, gfx_id) returns bit-identical
output across calls. LRU-cached on the full argument tuple.

Size constraint: the composed prose for a role is ≤ 400 lines (enforced by
framework.Agent and test_fence_size_guardrail.py); the full assembled fence
including YAML excerpts is ≤ 60 KB (enforced by test_determinism.py).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

_SLICES_DIR = Path(__file__).parent.parent / "agents" / "fence"

SHARED_SLICE = "always"

_ROLES = frozenset(
    [
        "root",
        "analysis",
        "recommendation",
        "correctness",
        "compute_specialist",
        "memory_specialist",
        "latency_specialist",
        "diff_specialist",
    ]
)


def known_role(role: str) -> bool:
    return role in _ROLES


def _load_slice(name: str) -> str:
    path = _SLICES_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"fence slice missing: {path}")
    return path.read_text()


@lru_cache(maxsize=16)
def compose_prompt(role: str) -> str:
    """Return the shared fence plus ``role``'s fence — the live agent prompt."""
    if role not in _ROLES:
        known = ", ".join(sorted(_ROLES))
        raise KeyError(f"unknown fence role {role!r}; known: {known}")
    return "\n\n".join([_load_slice(SHARED_SLICE), _load_slice(role)])


@lru_cache(maxsize=64)
def _build_cached(role: str, bottleneck: Optional[str], gfx_id: Optional[str]) -> str:
    sections = [compose_prompt(role)]

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


__all__ = ["FenceBuilder", "compose_prompt", "known_role", "SHARED_SLICE"]
