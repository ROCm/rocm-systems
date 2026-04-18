"""LLM-enabled end-to-end smoke — asserts rec_type populates with a real LLM.

Skipped when:
  * OPENAI_API_KEY is absent,
  * the memory_bound fixture DB is missing, or
  * the framework's live SDK path (`_sdk_invoke`) is still a stub
    (Phase 8 wires the real Agents-SDK runtime; this test activates
    automatically once that landing happens).
"""

import os
from pathlib import Path

import pytest

FIX = Path(__file__).parent.parent / "fixtures"
FIXTURE_DB = FIX / "memory_bound.db"


def _live_sdk_path_implemented() -> bool:
    """Heuristic: the framework stub raises NotImplementedError for live calls."""
    import inspect

    try:
        from perfxpert.agents import framework  # type: ignore
    except Exception:
        return False
    try:
        src = inspect.getsource(framework._sdk_invoke)
    except (OSError, TypeError):
        return False
    return "NotImplementedError" not in src


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — LLM end-to-end test requires a real provider",
)
@pytest.mark.skipif(
    not FIXTURE_DB.exists(),
    reason=f"fixture db missing: {FIXTURE_DB}",
)
@pytest.mark.skipif(
    not _live_sdk_path_implemented(),
    reason="framework._sdk_invoke is still a Phase-5 stub; live path lands in Phase 8",
)
def test_llm_enabled_produces_rec_type():
    from perfxpert.agents import runtime, schemas

    session = runtime.build_session(provider="openai", airgap=False)
    root_input = schemas.RootInput(
        user_query="Analyze this GPU performance trace.",
        database_path=str(FIXTURE_DB),
        provider="openai",
        airgap=False,
        session_id=session.session_id,
    )
    out = session.run_root(root_input)
    assert out.primary_bottleneck, "primary_bottleneck should be populated"
    assert out.narrative, "narrative should be non-empty"
    assert out.recommendations, "recommendations list should not be empty"
    assert out.recommendations[0].get("type"), (
        f"recommendations[0].type missing — {out.recommendations[0]}"
    )
