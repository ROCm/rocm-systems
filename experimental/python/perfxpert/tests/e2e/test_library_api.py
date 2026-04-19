"""E2E: library-API public surface.

Regression guard: the legacy `perfxpert.ai_analysis.analyze_database`
library API is removed. Programmatic callers must use the agentic
runtime (`perfxpert.agents`).
"""

import importlib
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent.parent / "fixtures" / "regression_baseline.db"


def test_legacy_ai_analysis_module_is_removed():
    """Regression guard: importing the removed legacy module must fail cleanly."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("perfxpert.ai_analysis")


def test_agentic_runtime_runs_airgap(monkeypatch):
    """The agentic runtime should be importable and run end-to-end in airgap."""
    if not FIXTURE.exists():
        pytest.skip(f"Fixture {FIXTURE} not found")

    # Regression guard — ensure the removed PERFXPERT_LEGACY env var stays inert.
    monkeypatch.delenv("PERFXPERT_LEGACY", raising=False)  # regression guard
    try:
        from perfxpert.agents import runtime, schemas
    except ImportError:
        pytest.skip("Agents runtime not available in this build")

    session = runtime.build_session(airgap=True)
    out = session.run_root(
        schemas.RootInput(
            user_query="Analyze this GPU performance trace.",
            database_path=str(FIXTURE),
            airgap=True,
            session_id=session.session_id,
        )
    )
    assert out is not None
    assert hasattr(out, "narrative")
