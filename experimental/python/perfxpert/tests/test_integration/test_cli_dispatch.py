"""Tests for analyze.py CLI dispatch (agentic is the only path).

Regression guards: assert the legacy dispatch symbols stay removed and
that legacy env vars cannot revive them.
"""

from unittest import mock

import pytest

from perfxpert import analyze as analyze_mod


@pytest.fixture
def fake_db(tmp_path):
    import sqlite3
    db = tmp_path / "fake.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE rocpd_kernel_dispatch (
            id INTEGER PRIMARY KEY, name TEXT, duration_ns INTEGER
        );
        INSERT INTO rocpd_kernel_dispatch VALUES (1, 'matmul', 1000);
    """)
    conn.commit()
    conn.close()
    return db


def test_cli_always_runs_agentic(fake_db, monkeypatch):
    """CLI always uses the agentic path; no feature-flag branching remains."""
    monkeypatch.delenv("PERFXPERT_LEGACY", raising=False)  # regression guard
    with mock.patch.object(analyze_mod, "_execute_agentic") as agentic:
        agentic.return_value = 0
        # Use the kwarg name the agentic layer actually reads. The legacy
        # `format=` kwarg was silently dropped in cycle-1 tests
        # (nitpick: misleading even though harmless).
        analyze_mod.execute(input=mock.MagicMock(), output_format="text")
        agentic.assert_called_once()


def test_cli_legacy_flag_is_no_op(fake_db, monkeypatch):
    """Regression guard: the removed PERFXPERT_LEGACY env var must still route agentic."""
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")  # regression guard
    with mock.patch.object(analyze_mod, "_execute_agentic") as agentic:
        agentic.return_value = 0
        analyze_mod.execute(input=mock.MagicMock(), output_format="text")
        agentic.assert_called_once()


def _analysis_output():
    from perfxpert.agents import schemas

    return schemas.AnalysisOutput(
        primary_bottleneck="compute",
        confidence=0.91,
        time_breakdown={
            "kernel_pct": 0.82,
            "memcpy_pct": 0.05,
            "api_pct": 0.08,
            "idle_pct": 0.05,
        },
        hot_kernels=[
            {
                "name": "tile_mfma_loop",
                "calls": 4,
                "total_duration_ns": 42_000,
                "avg_duration_ns": 10_500,
                "min_duration_ns": 9_000,
                "max_duration_ns": 12_000,
                "pct": 0.82,
            }
        ],
        counter_data_available=True,
    )


def _recommendation_output():
    from perfxpert.agents import schemas

    return schemas.RecommendationOutput(
        recommendations=[
            {
                "name": "mfma_tile_tuning",
                "title": "Tune MFMA tile sizes",
                "description": "Retile the kernel to improve matrix-core utilization.",
                "category": "compute",
                "priority": "HIGH",
                "expected_impact": 0.27,
                "actions": ["Re-profile after retuning block sizes."],
            }
        ],
        specialist_used="compute",
        plateau_detected=False,
    )


@pytest.mark.parametrize(
    "fmt, expected_fragments",
    [
        ("text", ["ROCPD AI PERFORMANCE ANALYSIS", "tile_mfma_loop", "Tune MFMA tile sizes"]),
        ("markdown", ["# PerfXpert AI Performance Analysis", "tile_mfma_loop", "Tune MFMA tile sizes"]),
        ("webview", ["<!DOCTYPE html>", "tile_mfma_loop", "Tune MFMA tile sizes"]),
    ],
)
def test_execute_agentic_runs_analysis_and_formats_reports(fmt, expected_fragments, capsys):
    """Database-backed CLI analysis must run Analysis -> Recommendation and use canonical formatters."""
    analysis_output = _analysis_output()
    recommendation_output = _recommendation_output()
    session = mock.Mock()
    session.session_id = "session-123"
    session.run_analysis.return_value = analysis_output
    session.run_recommendation.return_value = recommendation_output
    fake_input = mock.Mock()
    fake_input._paths = ["/tmp/fake.db"]

    with mock.patch("perfxpert.agents.runtime.build_session", return_value=session) as build_session:
        with mock.patch("perfxpert.agents.schemas.AnalysisInput", return_value="analysis-input") as analysis_input:
            with mock.patch(
                "perfxpert.agents.schemas.RecommendationInput",
                return_value="recommendation-input",
            ) as recommendation_input:
                analyze_mod._execute_agentic(
                    input=fake_input,
                    output_format=fmt,
                    prompt="why is matmul slow?",
                    llm_provider="openai",
                    enable_llm=True,
                    top_kernels=3,
                    att_dir="/tmp/att",
                    min_duration=5000.0,
                    llm_model="gpt-4.1",
                    llm_thinking=8000,
                    llm_local="ollama",
                    llm_local_model="codellama:13b",
                    verbose=True,
                )

    captured = capsys.readouterr()
    for fragment in expected_fragments:
        assert fragment in captured.out
    build_session.assert_called_once_with(provider="openai", airgap=False)
    analysis_input.assert_called_once_with(
        database_path="/tmp/fake.db",
        top_kernels=3,
        att_dir="/tmp/att",
        min_duration=5000.0,
        analysis_options={
            "top_kernels": 3,
            "att_dir": "/tmp/att",
            "min_duration": 5000.0,
            "llm_model": "gpt-4.1",
            "llm_thinking": 8000,
            "llm_local": "ollama",
            "llm_local_model": "codellama:13b",
            "verbose": True,
        },
    )
    recommendation_input.assert_called_once_with(findings=analysis_output)
    session.run_analysis.assert_called_once_with("analysis-input")
    session.run_recommendation.assert_called_once_with("recommendation-input")


def test_execute_agentic_renders_structured_json_from_analysis_outputs(capsys):
    """JSON output should come from the canonical analysis formatter stack."""
    analysis_output = _analysis_output()
    recommendation_output = _recommendation_output()
    session = mock.Mock()
    session.session_id = "session-123"
    session.run_analysis.return_value = analysis_output
    session.run_recommendation.return_value = recommendation_output
    fake_input = mock.Mock()
    fake_input._paths = ["/tmp/fake.db"]

    with mock.patch("perfxpert.agents.runtime.build_session", return_value=session):
        analyze_mod._execute_agentic(input=fake_input, format="json")

    captured = capsys.readouterr()
    import json

    payload = json.loads(captured.out)
    assert payload["summary"]["primary_bottleneck"] == "compute"
    assert payload["hotspots"][0]["name"] == "tile_mfma_loop"
    assert payload["recommendations"][0]["issue"] == "Tune MFMA tile sizes"
    assert payload["metadata"]["database_file"] == "/tmp/fake.db"
    assert payload["execution_breakdown"]["total_runtime_ns"] > 0
    assert payload["execution_breakdown"]["kernel_time_ns"] > 0


def test_execute_agentic_normalizes_claude_code_provider(capsys):
    """`claude-code` should route through the opencode provider internally."""
    analysis_output = _analysis_output()
    recommendation_output = _recommendation_output()
    session = mock.Mock()
    session.session_id = "session-123"
    session.run_analysis.return_value = analysis_output
    session.run_recommendation.return_value = recommendation_output
    fake_input = mock.Mock()
    fake_input._paths = ["/tmp/fake.db"]

    with mock.patch("perfxpert.agents.runtime.build_session", return_value=session) as build_session:
        with mock.patch("perfxpert.agents.schemas.AnalysisInput", return_value="analysis-input") as analysis_input:
            analyze_mod._execute_agentic(
                input=fake_input,
                output_format="text",
                llm_provider="claude-code",
                enable_llm=True,
            )

    captured = capsys.readouterr()
    assert "ROCPD AI PERFORMANCE ANALYSIS" in captured.out
    build_session.assert_called_once_with(provider="opencode", airgap=False)
    analysis_input.assert_called_once_with(
        database_path="/tmp/fake.db",
        top_kernels=10,
        att_dir=None,
        min_duration=0.0,
        analysis_options={},
    )


def test_execute_agentic_preserves_provider_taxonomy():
    """Auth and rate-limit errors should propagate unchanged to callers."""
    from perfxpert.providers._exceptions import AuthError

    session = mock.Mock()
    session.session_id = "session-123"
    session.run_analysis.side_effect = AuthError("openai", "bad key")
    fake_input = mock.Mock()
    fake_input._paths = ["/tmp/fake.db"]

    with mock.patch("perfxpert.agents.runtime.build_session", return_value=session):
        with pytest.raises(AuthError):
            analyze_mod._execute_agentic(
                input=fake_input,
                output_format="text",
                llm_provider="openai",
                enable_llm=True,
            )


def test_legacy_symbols_are_absent():
    """Regression guard: removed legacy symbols must stay gone."""
    assert not hasattr(analyze_mod, "_execute_legacy"), (  # regression guard
        "_execute_legacy was removed during the agentic refactor and must stay gone"
    )
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("perfxpert.ai_analysis")
