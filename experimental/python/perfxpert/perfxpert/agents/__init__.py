"""Agent hierarchy — 8 agents on a deterministic tool floor (spec §2)."""

from perfxpert.agents.runtime import AnalysisSession, build_session, DEFAULT_PROVIDER
from perfxpert.agents.schemas import (
    RootInput, RootOutput,
    AnalysisInput, AnalysisOutput,
    RecommendationInput, RecommendationOutput,
    CorrectnessInput, CorrectnessOutput,
    GateVerdictModel,
)
from perfxpert.agents.root import run_root, build_root_agent
from perfxpert.agents.analysis import run_analysis, build_analysis_agent
from perfxpert.agents.recommendation import run_recommendation, build_recommendation_agent
from perfxpert.agents.correctness import run_correctness, build_correctness_agent
from perfxpert.agents.compute_specialist import run_compute_specialist, build_compute_specialist
from perfxpert.agents.memory_specialist import run_memory_specialist, build_memory_specialist
from perfxpert.agents.latency_specialist import run_latency_specialist, build_latency_specialist
from perfxpert.agents.diff_specialist import run_diff_specialist, build_diff_specialist


# Canonical agent inventory. The CI guardrail tests (tool allowlist, schema
# field caps, fence size) enumerate this rather than keeping their own lists,
# which previously drifted and left the Diff specialist unchecked.
AGENT_BUILDERS = (
    build_root_agent,
    build_analysis_agent,
    build_recommendation_agent,
    build_correctness_agent,
    build_compute_specialist,
    build_memory_specialist,
    build_latency_specialist,
    build_diff_specialist,
)


__all__ = [
    # Canonical inventory
    "AGENT_BUILDERS",
    # Session
    "AnalysisSession", "build_session", "DEFAULT_PROVIDER",
    # Schemas
    "RootInput", "RootOutput",
    "AnalysisInput", "AnalysisOutput",
    "RecommendationInput", "RecommendationOutput",
    "CorrectnessInput", "CorrectnessOutput",
    "GateVerdictModel",
    # Agent runners + builders
    "run_root", "build_root_agent",
    "run_analysis", "build_analysis_agent",
    "run_recommendation", "build_recommendation_agent",
    "run_correctness", "build_correctness_agent",
    "run_compute_specialist", "build_compute_specialist",
    "run_memory_specialist", "build_memory_specialist",
    "run_latency_specialist", "build_latency_specialist",
    "run_diff_specialist", "build_diff_specialist",
]
