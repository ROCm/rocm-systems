"""analyze_run — single-call wrapper over the full Root->Analysis->Rec pipeline.

Blocker 1 (same-brain): opencode / claude / codex / gemini TUI backends
previously spun up their own agent loops and called perfxpert MCP tools
à la carte. The tools are shared, but the decision hierarchy (Root →
Analysis → Recommendation → Specialists) was NOT — each TUI's native
planner was making its own call graph.

This tool exposes the ENTIRE perfxpert decision hierarchy as ONE MCP
call. A backend that calls ``perfxpert_run_root_analysis`` gets the
same structured verdict (narrative + primary_bottleneck + recommendations
+ warnings + metadata) the in-process ``perfxpert analyze`` path would
produce, without the backend having to route through its own planner.

The TUI-side prompt (see ``_bundled/opencode_config/AGENTS.md`` and the
per-backend renderings) instructs the model to call this tool FIRST for
any GPU-performance query. Combined with the existing
``perfxpert_intent_classify`` gate, the two tools together enforce the
same brain across every backend.

Tool class: READ_ONLY — pure aggregator over already-READ_ONLY tools.
Honors ``PERFXPERT_AIRGAP=1`` + the full provider / fallback-chain
ladder because it defers directly to ``agents.runtime.build_session``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def run_root_analysis(
    user_query: str = "Analyze this GPU performance trace.",
    database_path: Optional[str] = None,
    source_dir: Optional[str] = None,
    provider: Optional[str] = None,
    airgap: bool = False,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full perfxpert Root → Analysis → Recommendation pipeline.

    Wraps :func:`perfxpert.agents.runtime.build_session` + ``run_root``
    in a single MCP-visible call so backend TUIs (opencode / claude /
    codex / gemini) use the SAME decision hierarchy as the in-process
    ``perfxpert analyze`` path — not each backend's native planner.

    Args:
        user_query: Free-form user question. Passed as
            ``RootInput.user_query`` so the agent can route by intent.
        database_path: Path to a rocprofiler-sdk ``.db`` file. Optional
            for Tier 0 (source-only) analysis.
        source_dir: Path to a source tree for Tier 0 static scan.
            Optional if ``database_path`` is set.
        provider: Explicit LLM provider name. Falls back to the session
            default when unset. Ignored under airgap.
        airgap: When ``True`` (or ``PERFXPERT_AIRGAP=1``), skip every
            provider call and return the deterministic airgap narrative.
        session_id: Re-use an existing session id. A UUID is generated
            when unset.

    Returns:
        A dict with the documented RootOutput schema keys:
        ``{"narrative", "recommendations", "primary_bottleneck",
        "warnings", "metadata"}``.
    """
    # Lazy import so the MCP registry can import this module without
    # eagerly loading the agents runtime (which pulls pydantic /
    # openai-agents). Tests patch ``build_session`` on this symbol.
    from perfxpert.agents import runtime, schemas

    session = runtime.build_session(
        provider=provider,
        session_id=session_id,
        airgap=airgap if airgap else None,
    )

    payload = schemas.RootInput(
        user_query=user_query,
        database_path=database_path,
        source_dir=source_dir,
        provider=provider,
        airgap=airgap,
        session_id=session.session_id,
    )

    root_output = session.run_root(payload)

    # RootOutput is a frozen Pydantic model; ``model_dump`` gives us the
    # documented schema-shaped dict. Fall back to attribute reads if the
    # test passes a plain object (e.g. a SimpleNamespace).
    if hasattr(root_output, "model_dump"):
        return root_output.model_dump()
    return {
        "narrative": getattr(root_output, "narrative", ""),
        "recommendations": list(getattr(root_output, "recommendations", []) or []),
        "primary_bottleneck": getattr(root_output, "primary_bottleneck", "mixed"),
        "warnings": list(getattr(root_output, "warnings", []) or []),
        "metadata": dict(getattr(root_output, "metadata", {}) or {}),
    }


__all__ = ["run_root_analysis"]
