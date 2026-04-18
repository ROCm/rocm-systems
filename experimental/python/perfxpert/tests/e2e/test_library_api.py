"""E2E: library-API public surface."""

from pathlib import Path

import pytest

from perfxpert.ai_analysis import analyze_database


FIXTURE = Path(__file__).parent.parent / "fixtures" / "regression_baseline.db"


def test_legacy_path_returns_analysis_result(monkeypatch):
    if not FIXTURE.exists():
        pytest.skip(f"Fixture {FIXTURE} not found")

    monkeypatch.delenv("PERFXPERT_USE_AGENTS", raising=False)
    result = analyze_database(database_path=FIXTURE)
    assert result is not None
    assert hasattr(result, "metadata")
    assert hasattr(result, "execution_breakdown") or hasattr(result, "summary")


def test_agentic_flag_surface_error_is_clean(monkeypatch):
    """With flag on but agents runtime absent, the error must be clear — not a traceback dump."""
    if not FIXTURE.exists():
        pytest.skip(f"Fixture {FIXTURE} not found")

    monkeypatch.setenv("PERFXPERT_USE_AGENTS", "1")
    try:
        result = analyze_database(database_path=FIXTURE)
    except RuntimeError as e:
        assert "agent runtime is not available" in str(e)
    except ImportError:
        # Also acceptable if SDK is just missing
        pass
    else:
        # If it succeeded, that's fine too — means Phase 3 runtime is present
        assert result is not None
