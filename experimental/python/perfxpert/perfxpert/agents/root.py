"""Root agent (Layer 0) — user-facing entry point.

Responsibilities:
- Route user intent to exactly one Layer-1 decision-maker
- Delegate to Analysis / Recommendation / Correctness and assemble a
  final narrative driven by their structured outputs (NOT by Root's own
  routing prose).
- Write 2-3 sentence bottleneck-classification prose (absorbed from
  former Bottleneck-Narrator per review N3)
- Manage task backbone via tasks.* tools

Tool allowlist (≤5):
  intent.classify, tasks.next, tasks.create, tasks.update, tasks.close

Handoff whitelist: analysis, recommendation, correctness (Layer 1 only).
Cannot skip directly to Layer 2.

Fence: agents/fence/root.md (≤ 400 lines; per-agent slice of the split fence).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from perfxpert.agents import schemas
from perfxpert.agents.framework import (
    Agent, ToolBinding, run_agent,
)
from perfxpert.runtime import classify_intent
from perfxpert.tools import intent as intent_tool
from perfxpert.tools import tasks as _tasks


_FENCE_PATH = Path(__file__).parent / "fence" / "root.md"


def build_root_agent() -> Agent:
    """Construct the Root agent with its fixed tool allowlist + handoffs.

    tasks.* bindings are EXECUTION-class tools: they mutate the local
    ``.perfxpert/`` task store (via ``perfxpert.tools.tasks``). They are
    intentionally NOT exposed to external MCP clients — the read-only
    invariant for MCP lives in ``mcp_server`` (§5.8).
    """
    tools = [
        ToolBinding(name="intent.classify", fn=intent_tool.classify),
        ToolBinding(name="tasks.next", fn=_tasks.next_task),
        ToolBinding(name="tasks.create", fn=_tasks.create),
        ToolBinding(name="tasks.update", fn=_tasks.update),
        ToolBinding(name="tasks.close", fn=_tasks.close),
    ]
    return Agent(
        name="Root",
        layer=0,
        fence_path=str(_FENCE_PATH) if _FENCE_PATH.exists() else None,
        input_schema=schemas.RootInput,
        output_schema=schemas.RootOutput,
        tools=tools,
        allowed_handoffs=["analysis", "recommendation", "correctness"],
        token_budget=4096,
    )


_INTENT_TO_HANDOFF = {
    "analyze": "analysis",
    "optimize": "recommendation",
    "verify": "correctness",
    "explain": "analysis",   # explain falls back to analysis
    "help": "analysis",
}


# Intents that should trigger a sub-agent delegation (Bug 1). Everything
# routes through Analysis first (findings drive the narrative); optimize
# additionally runs Recommendation; verify additionally runs Correctness.
_INTENTS_WITH_SUBAGENT = frozenset({"analyze", "optimize", "verify", "explain", "help"})


def _emit(progress_callback: Optional[Callable[[str], None]], msg: str) -> None:
    """Fire a progress callback, swallowing any exception it raises."""
    if progress_callback is None:
        return
    try:
        progress_callback(msg)
    except Exception:
        pass


def _format_pct(value: Any) -> Optional[str]:
    """Render a 0–1 or 0–100 value as ``"XX.X%"``; None on missing."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    # Heuristic: values > 1.0 are already in the 0–100 range.
    pct = v if v > 1.0 else v * 100.0
    return f"{pct:.1f}%"


def _build_narrative_from_analysis(
    analysis_out: schemas.AnalysisOutput,
    *,
    source_dir: Optional[str] = None,
) -> str:
    """Synthesize a deterministic narrative paragraph from Analysis findings.

    Used under airgap AND when the LLM did not emit a usable narrative. The
    output references the top kernel by name + its share of GPU time and
    names the primary bottleneck class with the metric evidence that drove
    it.
    """
    bn = analysis_out.primary_bottleneck or "mixed"
    tb = analysis_out.time_breakdown or {}
    hk = list(analysis_out.hot_kernels or [])

    # Top kernel — tolerate the two shapes the pipeline emits:
    #   deterministic: {"name", "percent_of_total", "total_duration", ...}
    #   agent schema:  {"name", "pct", "duration_ns", ...}
    top_sentence = ""
    if hk:
        top = hk[0]
        name = top.get("name") or top.get("kernel") or "unknown_kernel"
        pct = (
            top.get("percent_of_total")
            if top.get("percent_of_total") is not None
            else top.get("pct")
        )
        pct_str = _format_pct(pct) if pct is not None else None
        if pct_str:
            top_sentence = (
                f"The trace is dominated by kernel `{name}` at {pct_str} "
                "of GPU time. "
            )
        else:
            top_sentence = f"Top kernel by time: `{name}`. "

    # Bottleneck evidence.
    kernel_pct = tb.get("kernel_percent") or tb.get("kernel_pct")
    memcpy_pct = tb.get("memcpy_percent") or tb.get("memcpy_pct")
    overhead_pct = tb.get("overhead_percent") or tb.get("api_pct")
    evidence_bits: List[str] = []
    if kernel_pct is not None:
        p = _format_pct(kernel_pct)
        if p:
            evidence_bits.append(f"kernels {p}")
    if memcpy_pct is not None:
        p = _format_pct(memcpy_pct)
        if p:
            evidence_bits.append(f"memcpy {p}")
    if overhead_pct is not None:
        p = _format_pct(overhead_pct)
        if p:
            evidence_bits.append(f"API overhead {p}")
    evidence = ", ".join(evidence_bits)

    verdict_sentence = f"The primary bottleneck is `{bn}`"
    if evidence:
        verdict_sentence += f" (time breakdown: {evidence})."
    else:
        verdict_sentence += "."

    counter_note = ""
    if not analysis_out.counter_data_available:
        counter_note = (
            " Hardware counter data is unavailable — re-profile with `--pmc` "
            "for a confident classification."
        )

    tier0_note = ""
    if source_dir:
        tier0_note = (
            f" Source tree scanned at `{source_dir}`; see the Tier-0 section "
            "for the profiling plan and detected code patterns."
        )

    return (top_sentence + verdict_sentence + counter_note + tier0_note).strip()


def _analysis_out_to_rec_seeds(
    analysis_out: schemas.AnalysisOutput,
) -> List[Dict[str, Any]]:
    """Derive seed recommendations from Analysis findings (top kernels).

    These feed ``recommendations`` so the merged output has at least the
    top-kernel triage items even without an LLM. The deterministic
    ``recommendations_deterministic`` pass in ``build_analysis_payload``
    adds richer items; both are deduped by ``merge_recommendations``.
    """
    seeds: List[Dict[str, Any]] = []
    for k in (analysis_out.hot_kernels or [])[:3]:
        name = k.get("name") or k.get("kernel")
        if not name:
            continue
        pct = (
            k.get("percent_of_total")
            if k.get("percent_of_total") is not None
            else k.get("pct")
        )
        pct_str = _format_pct(pct) if pct is not None else None
        issue = (
            f"Hot kernel `{name}`"
            + (f" — {pct_str} of GPU time" if pct_str else "")
        )
        seeds.append({
            "priority": "INFO",
            "category": "Hotspot",
            "issue": issue,
            "suggestion": (
                f"Inspect `{name}` for the dominant bottleneck "
                f"(`{analysis_out.primary_bottleneck}`)."
            ),
        })
    return seeds


def run_root(
    payload: schemas.RootInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> schemas.RootOutput:
    """Execute the Root agent for a single user turn.

    Deterministic routing via ``runtime.classify_intent`` first (air-gap
    parity invariant). Then:

    * All intents drive an Analysis run (findings become the narrative).
    * ``optimize`` additionally runs Recommendation (merged into recs).
    * ``verify`` additionally runs Correctness when a gate verdict is
      supplied via ``payload.analysis_options['gate_verdict']``.

    Root's own LLM call wraps the narrative with high-level framing prose;
    it must NEVER be the narrative ALONE. (Bug 1.)
    """
    # Step 1: Deterministic routing — runs in BOTH modes.
    verdict = classify_intent(payload.user_query)
    routed_to = _INTENT_TO_HANDOFF.get(verdict.intent, "analysis")

    # Step 2: Delegate to the relevant sub-agent(s) — see Bug 1 fix.
    analysis_out: Optional[schemas.AnalysisOutput] = None
    rec_out: Optional[schemas.RecommendationOutput] = None
    corr_out: Optional[schemas.CorrectnessOutput] = None
    warnings: List[str] = []

    # Lazy imports — these modules pull the full agent stack; importing at
    # module top would cause a circular import through agents.__init__.
    from perfxpert.agents import analysis as analysis_mod
    from perfxpert.agents import correctness as correctness_mod
    from perfxpert.agents import recommendation as recommendation_mod

    # Analysis is the common driver. Skip only when no DB was supplied and
    # we are in non-analyze intents that need no findings (nothing today).
    if verdict.intent in _INTENTS_WITH_SUBAGENT and payload.database_path:
        _emit(progress_callback, "Consulting Analysis agent…")
        try:
            a_input = schemas.AnalysisInput(
                database_path=payload.database_path,
                top_kernels=10,
            )
            analysis_out = analysis_mod.run_analysis(
                a_input, provider=provider, airgap=airgap,
            )
        except Exception as exc:  # pragma: no cover — defensive
            warnings.append(f"Analysis agent raised {type(exc).__name__}: {exc}")
            analysis_out = None
        finally:
            _emit(progress_callback, "Analysis agent done")

    if verdict.intent == "optimize" and analysis_out is not None:
        _emit(progress_callback, "Consulting Recommendation agent…")
        try:
            r_input = schemas.RecommendationInput(findings=analysis_out)
            rec_out = recommendation_mod.run_recommendation(
                r_input, provider=provider, airgap=airgap,
            )
        except Exception as exc:  # pragma: no cover — defensive
            warnings.append(
                f"Recommendation agent raised {type(exc).__name__}: {exc}"
            )
            rec_out = None
        finally:
            _emit(progress_callback, "Recommendation agent done")

    if verdict.intent == "verify":
        opts = dict(payload.analysis_options or {})
        gate_verdict_raw = opts.get("gate_verdict")
        if gate_verdict_raw is not None:
            _emit(progress_callback, "Consulting Correctness agent…")
            try:
                if isinstance(gate_verdict_raw, schemas.GateVerdictModel):
                    gate_verdict = gate_verdict_raw
                else:
                    gate_verdict = schemas.GateVerdictModel(**gate_verdict_raw)
                c_input = schemas.CorrectnessInput(
                    gate_verdict=gate_verdict,
                    kernel_name=opts.get("kernel_name"),
                    last_technique=opts.get("last_technique"),
                )
                corr_out = correctness_mod.run_correctness(
                    c_input, provider=provider, airgap=airgap,
                )
            except Exception as exc:  # pragma: no cover — defensive
                warnings.append(
                    f"Correctness agent raised {type(exc).__name__}: {exc}"
                )
                corr_out = None
            finally:
                if corr_out is not None and corr_out.verdict == "pass":
                    _emit(progress_callback, "Correctness gate passed")
                else:
                    detail = (
                        corr_out.narrative if corr_out is not None else "no verdict"
                    )
                    _emit(
                        progress_callback,
                        f"Correctness gate failed: {detail}",
                    )
                    if corr_out is not None:
                        warnings.append(
                            f"Correctness gate {corr_out.verdict}: {corr_out.narrative}"
                        )
        else:
            warnings.append(
                "verify intent requires analysis_options['gate_verdict']; "
                "skipping Correctness run."
            )

    # Step 3: Run Root's own LLM prompt for framing prose (optional); it
    # must NEVER supply the narrative alone (Bug 1).
    agent = build_root_agent()
    raw = run_agent(
        agent,
        input_payload=payload.model_dump(),
        provider=provider,
        airgap=airgap,
    )

    # Step 4: Build the final narrative + recs.
    if analysis_out is not None:
        narrative = _build_narrative_from_analysis(
            analysis_out, source_dir=payload.source_dir,
        )
        primary_bottleneck = analysis_out.primary_bottleneck
        hot_kernel_seeds = _analysis_out_to_rec_seeds(analysis_out)
    else:
        # No DB + no sub-agent run — synthesize a minimal narrative and let
        # downstream deterministic / tier-0 sections carry the rest.
        narrative = (
            "No trace database was supplied. "
            + ("Running a Tier-0 source scan — see below for details. "
               if payload.source_dir else "")
            + "Supply -i <database.db> to run the full Analysis pipeline."
        )
        primary_bottleneck = "mixed"
        hot_kernel_seeds = []

    # Optional high-level framing from Root's own LLM call — wrap, do NOT
    # replace the narrative.
    root_framing = ""
    if raw.get("_mode") != "airgap":
        so = raw.get("structured_output") or {}
        root_framing = (so.get("narrative") or "").strip()

    if root_framing:
        # Prepend Root's framing sentence to the findings-derived narrative.
        # Framing adds tone ("Here is what I found:"), the analysis text
        # carries the facts.
        narrative = root_framing.rstrip() + "\n\n" + narrative

    # Recommendations: start with LLM-emitted ones (if any from Root's own
    # call), fold in sub-agent outputs, and seed with analysis-derived
    # hotspot items. Never fall back to stuffing narrative prose into a
    # rec (Bug 2).
    recommendations: List[Dict[str, Any]] = []
    if raw.get("_mode") != "airgap":
        so_recs = raw.get("structured_output") or {}
        for r in so_recs.get("recommendations") or []:
            if isinstance(r, dict):
                recommendations.append(r)

    if rec_out is not None:
        recommendations.extend(list(rec_out.recommendations))

    recommendations.extend(hot_kernel_seeds)

    # Metadata — carry routing + sub-agent evidence.
    metadata: Dict[str, Any] = {
        "routed_to": routed_to,
        "intent": verdict.intent,
    }
    if raw.get("_mode") != "airgap":
        so_meta = (raw.get("structured_output") or {}).get("metadata") or {}
        if isinstance(so_meta, dict):
            metadata = {**so_meta, **metadata}
    if analysis_out is not None:
        metadata["analysis_confidence"] = analysis_out.confidence
        metadata["counter_data_available"] = analysis_out.counter_data_available
    if rec_out is not None:
        metadata["recommendation_specialist"] = rec_out.specialist_used
        metadata["plateau_detected"] = rec_out.plateau_detected
    if corr_out is not None:
        metadata["correctness_verdict"] = corr_out.verdict
        metadata["correctness_action"] = corr_out.action

    if raw.get("_mode") == "airgap":
        warnings.append("airgap mode; deterministic template used")
    else:
        for w in (raw.get("structured_output") or {}).get("warnings") or []:
            if w not in warnings:
                warnings.append(w)

    return schemas.RootOutput(
        narrative=narrative,
        recommendations=recommendations,
        primary_bottleneck=primary_bottleneck,
        warnings=warnings,
        metadata=metadata,
    )


__all__ = ["build_root_agent", "run_root"]
