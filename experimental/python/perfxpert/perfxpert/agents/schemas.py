"""Pydantic I/O schemas for all agent handoffs.

Caps (spec §2, non-negotiable):
- Input schemas: ≤ 10 fields.
- Output schemas: ≤ 5 fields.

All models are frozen to prevent mutation during handoff. Returned objects
flow upward through the hierarchy immutably.

CI enforces caps in test_schema_field_caps.py.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# -- Base ------------------------------------------------------------------


class _FrozenModel(BaseModel):
    """Frozen base — handoff payloads must not mutate."""

    model_config = ConfigDict(frozen=True, extra="forbid")


BottleneckType = Literal["compute", "memory_transfer", "latency", "api_overhead", "mixed", "data_insufficient"]


# -- Gate verdict (consumed by Correctness from runtime/gate_cascade.py) ----


class GateVerdictModel(_FrozenModel):
    """Mirror of runtime.gate_cascade.GateVerdict for use in Pydantic schemas.

    Correctness agent consumes a verdict object; it does not produce one.
    """

    status: Literal["pass", "reject", "regressed"]
    failing_gate: Optional[Literal["compile", "sol", "bitwise", "regression", "anchors"]] = None
    detail: str = ""
    delta_pct: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    rejected_patch_sha: Optional[str] = None
    per_kernel_deltas: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _gate_coherence(self) -> "GateVerdictModel":
        """failing_gate must be set when status != 'pass'."""
        if self.status != "pass" and self.failing_gate is None:
            raise ValueError(f"GateVerdictModel: failing_gate must be set when status={self.status!r}")
        if self.status == "pass" and self.failing_gate is not None:
            raise ValueError(
                f"GateVerdictModel: failing_gate must be None when status='pass', " f"got {self.failing_gate!r}"
            )
        return self


# -- Root (Layer 0) --------------------------------------------------------


class RootInput(_FrozenModel):
    user_query: str
    database_path: Optional[str] = None
    source_dir: Optional[str] = None
    provider: Optional[
        Literal["anthropic", "openai", "ollama", "private", "opencode"]
    ] = None
    airgap: bool = False
    session_id: Optional[str] = None
    intent_hint: Optional[str] = None  # populated by intent.classify upstream
    # CLI-flag side-channel forwarded downstream via Analysis handoff.
    # Kept as a single dict so we respect the ≤10-field schema cap (§2).
    analysis_options: Dict[str, Any] = Field(default_factory=dict)


class RootOutput(_FrozenModel):
    narrative: str  # assembled final answer to user
    recommendations: List[Dict[str, Any]]  # flat list (Root dedups specialist output)
    primary_bottleneck: BottleneckType
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -- Analysis (Layer 1) ----------------------------------------------------


class AnalysisInput(_FrozenModel):
    database_path: str
    top_kernels: int = 10
    att_dir: Optional[str] = None  # auto-detected; not required
    min_duration: float = 0.0


class AnalysisOutput(_FrozenModel):
    primary_bottleneck: BottleneckType
    confidence: float = Field(..., ge=0.0, le=1.0)
    time_breakdown: Dict[str, float]  # kernel_pct / memcpy_pct / api_pct / idle_pct
    hot_kernels: List[Dict[str, Any]]  # [{name, pct, duration_ns, ...}]
    counter_data_available: bool


# -- Recommendation (Layer 1) ----------------------------------------------


class RecommendationInput(_FrozenModel):
    findings: AnalysisOutput
    gfx_id: str
    kernel_filter: Optional[str] = None
    edit_history: List[Dict[str, Any]] = Field(default_factory=list)
    seen_recommendation_hashes: List[str] = Field(default_factory=list)


class RecommendationOutput(_FrozenModel):
    recommendations: List[Dict[str, Any]]  # flat, ranked, deduplicated
    specialist_used: Literal["compute", "memory", "latency", "none"]
    plateau_detected: bool = False


# -- Correctness (Layer 1) -------------------------------------------------


class CorrectnessInput(_FrozenModel):
    gate_verdict: GateVerdictModel
    kernel_name: Optional[str] = None
    last_technique: Optional[str] = None
    source_dir: Optional[str] = None
    edit_history: List[Dict[str, Any]] = Field(default_factory=list)


class CorrectnessOutput(_FrozenModel):
    verdict: Literal["pass", "reject", "regressed"]
    action: Literal["accept", "revert", "reject_and_log"]
    narrative: str
    alternative_technique: Optional[str] = None
    follow_up_task_id: Optional[str] = None

    @model_validator(mode="after")
    def _verdict_action_coherence(self) -> "CorrectnessOutput":
        """action must be coherent with verdict."""
        valid = {
            "pass": {"accept"},
            "reject": {"revert", "reject_and_log"},
            "regressed": {"revert", "reject_and_log"},
        }
        allowed = valid.get(self.verdict, set())
        if self.action not in allowed:
            raise ValueError(
                f"CorrectnessOutput: action={self.action!r} is not valid for "
                f"verdict={self.verdict!r}; allowed: {sorted(allowed)}"
            )
        return self


# -- Specialists (Layer 2) --------------------------------------------------


class ComputeSpecialistInput(_FrozenModel):
    gfx_id: str
    hot_kernels: List[Dict[str, Any]]
    counter_data: Dict[str, Any] = Field(default_factory=dict)
    source_hints: Dict[str, Any] = Field(default_factory=dict)


class ComputeSpecialistOutput(_FrozenModel):
    techniques: List[Dict[str, Any]]  # [{name, rationale, expected_impact, effort, risk}]
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: List[str] = Field(default_factory=list)


class MemorySpecialistInput(_FrozenModel):
    gfx_id: str
    hot_kernels: List[Dict[str, Any]]
    memcpy_data: Dict[str, Any] = Field(default_factory=dict)
    counter_data: Dict[str, Any] = Field(default_factory=dict)


class MemorySpecialistOutput(_FrozenModel):
    techniques: List[Dict[str, Any]]
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: List[str] = Field(default_factory=list)


class LatencySpecialistInput(_FrozenModel):
    gfx_id: str
    hot_kernels: List[Dict[str, Any]]
    api_overhead_pct: float = 0.0
    avg_kernel_duration_us: Optional[float] = None


class LatencySpecialistOutput(_FrozenModel):
    techniques: List[Dict[str, Any]]
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: List[str] = Field(default_factory=list)


# -- Diff specialist (Layer 2) --------------------------------------------
#
# Wraps :func:`perfxpert.tools.trace_diff.diff_runs` + a narrative.
# Output cap is 5 fields, so ``regressions`` and ``improvements`` are
# nested under a single ``kernel_deltas`` dict ({"regressions":[...],
# "improvements":[...]}) — the MCP tool wrapper flattens them back to
# top-level keys for the public return shape documented in
# agent-hierarchy.md.

class DiffSpecialistInput(_FrozenModel):
    baseline_db: str = Field(..., description="Path to baseline rocprofiler-sdk .db")
    new_db: str = Field(..., description="Path to new run's rocprofiler-sdk .db")
    top_kernels: int = Field(default=20, ge=1, le=100)
    user_intent: str = Field(default="summarize the diff")


class DiffSpecialistOutput(_FrozenModel):
    wall_delta_pct: float
    kernel_deltas: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=lambda: {"regressions": [], "improvements": []},
        description="Per-kernel deltas, keyed by 'regressions' / 'improvements'.",
    )
    verdict: Literal["improved", "regressed", "neutral"]
    narrative: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# -- Communication (RCCL payload block) -----------------------------------
#
# Leaf data models, not agent Input/Output — the ≤5-field cap does NOT
# apply. These type the per-collective shape emitted by
# ``perfxpert.tools.rccl_analysis.analyze_collectives`` so downstream
# consumers (formatters, MCP clients, Latency specialist) get a single
# source of truth for the field list. The top-level ``summary`` dict
# stays untyped (less stable shape, best treated as free-form).


class CollectiveEntry(_FrozenModel):
    """One per-collective record inside ``payload["communication"]``.

    Pydantic-validated at the ``analyze_collectives`` boundary so callers
    can rely on field presence + basic types. The set of acceptable
    ``efficiency_label`` values mirrors ``rccl_analysis._classify_efficiency``
    (poor/fair/good) plus the ``unknown`` case emitted by the
    kernel-name-regex fallback when ``capture_incomplete=True``.
    """

    op_type: str
    msg_bytes: int = Field(..., ge=0)
    duration_ns: int = Field(..., ge=0)
    effective_bw_gbps: float
    peak_bw_gbps: Optional[float] = None
    efficiency_pct: float
    efficiency_label: Literal["poor", "fair", "good", "unknown"]
    overlap_ratio: float
    algo_hint: Optional[str] = None
    topology_hint: Optional[str] = None
    regime: Optional[str] = None
    ranks: int = Field(..., ge=1)


class CommunicationBlock(_FrozenModel):
    """Typed wrapper for ``payload["communication"]`` — 3 top-level keys."""

    collectives: List[CollectiveEntry]
    summary: Dict[str, Any]  # top-level aggregate is less stable — keep untyped
    capture_incomplete: bool = False


# -- Exports ---------------------------------------------------------------

__all__ = [
    "BottleneckType",
    "GateVerdictModel",
    "RootInput",
    "RootOutput",
    "AnalysisInput",
    "AnalysisOutput",
    "RecommendationInput",
    "RecommendationOutput",
    "CorrectnessInput",
    "CorrectnessOutput",
    "ComputeSpecialistInput",
    "ComputeSpecialistOutput",
    "MemorySpecialistInput",
    "MemorySpecialistOutput",
    "LatencySpecialistInput",
    "LatencySpecialistOutput",
    "DiffSpecialistInput",
    "DiffSpecialistOutput",
    "CollectiveEntry",
    "CommunicationBlock",
]
