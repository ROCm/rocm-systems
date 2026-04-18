"""Legacy monolithic reference-guide loader.

**Post-Phase-6, the monolithic guide is DELETED in PR 2.** This loader only
functions under PERFXPERT_LEGACY=1 AND ROCINSIGHT_LLM_REFERENCE_GUIDE pointing
to a user-supplied replacement file. New code MUST use the split fence in
agents/fence/*.md (via `perfxpert.agents.fence.load_fence_slice(agent_name)`).
"""

from __future__ import annotations

import os
from pathlib import Path


class ReferenceGuideNotFoundError(RuntimeError):
    """Raised when the monolithic reference guide is requested but unavailable.

    This is the expected state in the agentic default path. Callers should
    use `perfxpert.agents.fence.load_fence_slice()` instead.
    """


def _is_legacy_mode() -> bool:
    return os.environ.get("PERFXPERT_LEGACY") == "1"


def load_reference_guide() -> str:
    """Load the legacy monolithic llm-reference-guide.md content.

    Resolution order (legacy-only):
    1. ROCINSIGHT_LLM_REFERENCE_GUIDE env var → explicit path
    2. (Removed) Package-relative share/llm-reference-guide.md — deleted in Phase 6 PR 2
    3. (Removed) /opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md — deleted in Phase 6 PR 2

    Returns:
        The guide contents as a string.

    Raises:
        ReferenceGuideNotFoundError: always, unless (a) PERFXPERT_LEGACY=1 and
        (b) ROCINSIGHT_LLM_REFERENCE_GUIDE points to an existing file.
    """
    if not _is_legacy_mode():
        raise ReferenceGuideNotFoundError(
            "The monolithic llm-reference-guide.md is retired. "
            "Use perfxpert.agents.fence.load_fence_slice(<agent>) instead, "
            "or set PERFXPERT_LEGACY=1 with ROCINSIGHT_LLM_REFERENCE_GUIDE "
            "pointing to your own copy."
        )

    override = os.environ.get("ROCINSIGHT_LLM_REFERENCE_GUIDE")
    if not override:
        raise ReferenceGuideNotFoundError(
            "Legacy mode requires ROCINSIGHT_LLM_REFERENCE_GUIDE to point to "
            "a user-supplied copy of the legacy monolithic guide. The in-tree "
            "copy was deleted in Phase 6 (see CHANGELOG.md)."
        )
    p = Path(override)
    if not p.exists():
        raise ReferenceGuideNotFoundError(
            f"Legacy monolithic guide not found at {override!r}. "
            "See docs/deprecation/PERFXPERT_LEGACY.md for migration steps."
        )
    return p.read_text(encoding="utf-8")
