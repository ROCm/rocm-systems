"""Run the current agentic analysis surface on a fixture and compare the
observed result to the fixture's expected contract.

The pre-agentic ``perfxpert.ai_analysis`` module is gone. The parity gate now
validates the current analysis + recommendation pipeline against the canonical
fixture expectations that guard the staged refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from perfxpert.agents import schemas
from perfxpert.agents.runtime import build_session
from perfxpert.analysis.payload import build_analysis_payload

from .fixtures_inventory import ParityFixture


@dataclass(frozen=True)
class ObservedResult:
    primary_bottleneck: Optional[str]
    primary_rec_type: Optional[str]
    primary_rec_technique: Optional[str]
    raw_analysis: object
    raw_recommendation: object
    duration_s: float


@dataclass(frozen=True)
class ParityResult:
    fixture_id: str
    expected_bottleneck: Optional[str]
    expected_rec_type: Optional[str]
    expected_rec_technique: Optional[str]
    observed: ObservedResult

    def agree_bottleneck(self) -> bool:
        return self.observed.primary_bottleneck == self.expected_bottleneck

    def agree_rec_type(self) -> bool:
        return self.observed.primary_rec_type == self.expected_rec_type

    def agree_rec_technique(self) -> bool:
        return self.observed.primary_rec_technique == self.expected_rec_technique

    def agreements(self) -> dict[str, bool]:
        return {
            "bottleneck": self.agree_bottleneck(),
            "rec_type": self.agree_rec_type(),
            "rec_technique": self.agree_rec_technique(),
        }


_SOURCE_CATEGORY_TO_BOTTLENECK = {
    "Memory Transfer": "memory_transfer",
    "Synchronization": "latency",
    "No Streams": "latency",
}
_SOURCE_CATEGORY_TO_REC_TYPE = {
    "Memory Transfer": "memory",
    "Synchronization": "latency",
    "No Streams": "latency",
}
_SOURCE_CATEGORY_TO_TECHNIQUE = {
    "Memory Transfer": "hip_stream_overlap",
    "Synchronization": "device_sync_removal",
    "No Streams": "hip_stream_overlap",
}


class ParityRunner:
    """Run the current analysis surface on a fixture and compare to expectations."""

    def run_fixture(self, fx: ParityFixture) -> ParityResult:
        import time

        start = time.time()
        if fx.source_dir:
            payload = build_analysis_payload(None, source_dir=str(fx.source_dir))
            observed = ObservedResult(
                primary_bottleneck=_extract_source_bottleneck(payload),
                primary_rec_type=_extract_source_rec_type(payload),
                primary_rec_technique=_extract_source_rec_technique(payload),
                raw_analysis=payload,
                raw_recommendation=None,
                duration_s=time.time() - start,
            )
        else:
            session = build_session(airgap=True)
            analysis_output = session.run_analysis(
                schemas.AnalysisInput(database_path=str(fx.db_path))
            )
            recommendation_output = session.run_recommendation(
                schemas.RecommendationInput(findings=analysis_output, gfx_id=fx.gfx_id)
            )
            observed = ObservedResult(
                primary_bottleneck=analysis_output.primary_bottleneck,
                primary_rec_type=_extract_db_rec_type(recommendation_output),
                primary_rec_technique=_extract_db_rec_technique(recommendation_output),
                raw_analysis=analysis_output,
                raw_recommendation=recommendation_output,
                duration_s=time.time() - start,
            )

        return ParityResult(
            fixture_id=fx.id,
            expected_bottleneck=fx.expected_bottleneck,
            expected_rec_type=fx.expected_rec_type,
            expected_rec_technique=fx.expected_rec_technique,
            observed=observed,
        )


def _primary_source_recommendation(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    tier0 = payload.get("tier0_findings") or {}
    recs = tier0.get("code_patterns") or tier0.get("recommendations") or []
    if not isinstance(recs, list) or not recs:
        return None
    first = recs[0]
    return first if isinstance(first, dict) else None


def _extract_source_bottleneck(payload: dict[str, Any]) -> Optional[str]:
    source_rec = _primary_source_recommendation(payload)
    if source_rec is None:
        return None
    return _SOURCE_CATEGORY_TO_BOTTLENECK.get(str(source_rec.get("category")))


def _extract_source_rec_type(payload: dict[str, Any]) -> Optional[str]:
    source_rec = _primary_source_recommendation(payload)
    if source_rec is None:
        return None
    return _SOURCE_CATEGORY_TO_REC_TYPE.get(str(source_rec.get("category")))


def _extract_source_rec_technique(payload: dict[str, Any]) -> Optional[str]:
    source_rec = _primary_source_recommendation(payload)
    if source_rec is None:
        return None
    return _SOURCE_CATEGORY_TO_TECHNIQUE.get(str(source_rec.get("category")))


def _extract_db_rec_type(result: schemas.RecommendationOutput) -> Optional[str]:
    recs = list(result.recommendations or [])
    if recs:
        category = str(recs[0].get("category", "")).strip().lower()
        if category in {"compute", "memory", "latency"}:
            return category
    if result.specialist_used in {"compute", "memory", "latency"}:
        return result.specialist_used
    return None


def _extract_db_rec_technique(result: schemas.RecommendationOutput) -> Optional[str]:
    recs = list(result.recommendations or [])
    if not recs:
        return None
    return recs[0].get("name") or recs[0].get("id")
