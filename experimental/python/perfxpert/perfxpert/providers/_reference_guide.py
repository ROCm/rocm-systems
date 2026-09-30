"""Reference-guide loader shim.

The legacy monolithic llm-reference-guide.md is retired. All callers MUST
use the split fence in ``perfxpert/agents/fence/*.md`` via
``perfxpert.agents.fence.load_fence_slice(agent_name)``.

This module is kept only to raise a clear, actionable error for any
remaining callers that still request the monolithic guide.
"""

from __future__ import annotations


class ReferenceGuideNotFoundError(RuntimeError):
    """Raised when the monolithic reference guide is requested.

    Callers should use ``perfxpert.agents.fence.load_fence_slice()`` instead.
    """


def load_reference_guide() -> str:
    """Always raise — the monolithic reference guide is no longer supported.

    Raises:
        ReferenceGuideNotFoundError: always. Use
        ``perfxpert.agents.fence.load_fence_slice(<agent>)`` instead.
    """
    raise ReferenceGuideNotFoundError(
        "The monolithic llm-reference-guide.md is retired. "
        "Use perfxpert.agents.fence.load_fence_slice(<agent>) instead."
    )
