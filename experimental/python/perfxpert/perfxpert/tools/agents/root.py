"""agent_root — MCP tool wrapping the Layer-0 Root agent.

Root is the user-facing entry point of the perfxpert agent hierarchy.
It reads the user's intent, routes to one of the three Layer-1
decision-makers (Analysis / Recommendation / Correctness), and
assembles a structured verdict with narrative + primary_bottleneck +
recommendations + warnings + metadata.

This wrapper is the ONLY thing every backend (opencode / claude /
codex / gemini) needs to reach the full agent brain: `agent_root`
builds an :class:`AnalysisSession` and dispatches `run_root` through
the same airgap + provider-selection semantics as the in-process
`perfxpert analyze` path.

Tool class: READ_ONLY — pure aggregator over already-READ_ONLY tools.
Honors ``PERFXPERT_AIRGAP=1`` plus the shared provider-selection
semantics from ``agents.runtime.build_session``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def agent_root(
    user_query: str = "Analyze this GPU performance trace.",
    database_path: Optional[str] = None,
    source_dir: Optional[str] = None,
    provider: Optional[str] = None,
    airgap: bool = False,
    session_id: Optional[str] = None,
    analysis_options: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full Root → Analysis → Recommendation pipeline.

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
        analysis_options: CLI-side analysis knobs forwarded through the
            public RootInput surface so batch CLI, library API, and MCP
            all share the same entry-point contract.
        progress_callback: Optional ``Callable[[str], None]`` that receives
            short phase-transition messages (e.g. ``"entering root"``,
            ``"exit root"``) so UI consumers (CLI spinner, MCP streaming
            consumer) can show progress during long LLM calls. ``None``
            disables the feature with zero overhead.
        api_key: Explicit LLM provider API key forwarded through to
            :func:`perfxpert.agents.runtime.build_session`. When set it
            overrides the provider-specific env var for the duration of
            this call (the previous env state is restored on exit). Use
            this path to pass ``--llm-api-key`` from the CLI. Ignored
            under airgap.

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
        progress_callback=progress_callback,
        api_key=api_key,
    )

    payload = schemas.RootInput(
        user_query=user_query,
        database_path=database_path,
        source_dir=source_dir,
        provider=provider,
        airgap=airgap,
        session_id=session.session_id,
        analysis_options=dict(analysis_options or {}),
    )

    try:
        output = session.run_root(payload, progress_callback=progress_callback)
    except TypeError as exc:
        if "progress_callback" in str(exc):
            output = session.run_root(payload)
        else:
            raise

    # RootOutput is a frozen Pydantic model; ``model_dump`` gives us the
    # documented schema-shaped dict. Fall back to attribute reads if the
    # test passes a plain object (e.g. a SimpleNamespace).
    if hasattr(output, "model_dump"):
        return output.model_dump()
    return {
        "narrative": getattr(output, "narrative", ""),
        "recommendations": list(getattr(output, "recommendations", []) or []),
        "primary_bottleneck": getattr(output, "primary_bottleneck", "mixed"),
        "warnings": list(getattr(output, "warnings", []) or []),
        "metadata": dict(getattr(output, "metadata", {}) or {}),
    }


agent_root.__tool_name__ = "agent_root"

__all__ = ["agent_root"]
