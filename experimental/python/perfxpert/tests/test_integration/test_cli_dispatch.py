"""Tests for analyze.py CLI dispatch (agentic is the only path).

Regression guards: assert the legacy dispatch symbols stay removed and
that legacy env vars cannot revive them.
"""

import json
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
    monkeypatch.delenv("PERFXPERT_LEGACY", raising=False)
    with mock.patch.object(analyze_mod, "_execute_agentic") as agentic:
        agentic.return_value = 0
        analyze_mod.execute(input=mock.MagicMock(), output_format="text")
        agentic.assert_called_once()


def test_cli_legacy_flag_is_no_op(fake_db, monkeypatch):
    """Regression guard: the removed PERFXPERT_LEGACY env var must still route agentic."""
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")
    with mock.patch.object(analyze_mod, "_execute_agentic") as agentic:
        agentic.return_value = 0
        analyze_mod.execute(input=mock.MagicMock(), output_format="text")
        agentic.assert_called_once()


def _root_output():
    return {
        "narrative": "Compute-bound workload dominated by tile_mfma_loop.",
        "recommendations": [
            {
                "issue": "Tune MFMA tile sizes",
                "type": "optimization",
                "category": "compute",
                "priority": "HIGH",
                "description": "Retile the kernel to improve matrix-core utilization.",
            }
        ],
        "primary_bottleneck": "compute",
        "warnings": [],
        "metadata": {"intent": "analyze", "routed_to": "analysis"},
    }


def _analysis_payload():
    return {
        "time_breakdown": {
            "kernel_pct": 0.82,
            "memcpy_pct": 0.05,
            "api_pct": 0.08,
            "idle_pct": 0.05,
        },
        "hotspots": [
            {
                "rank": 1,
                "name": "tile_mfma_loop",
                "calls": 4,
                "total_duration_ns": 42_000,
                "avg_duration_ns": 10_500,
                "min_duration_ns": 9_000,
                "max_duration_ns": 12_000,
                "pct_of_total": 82.0,
            }
        ],
        "memory_analysis": {},
        "hardware_counters": {
            "has_counters": True,
            "metrics": {"gpu_utilization_pct": 91.0, "avg_waves": 40.0},
            "counters": {},
        },
        "kernel_resources": {},
        "api_overhead": {"top_apis": []},
        "thread_trace": None,
        "tier0_findings": None,
        "recommendations_deterministic": [
            {
                "issue": "Tune MFMA tile sizes",
                "type": "optimization",
                "category": "compute",
                "priority": "HIGH",
                "suggestion": "Re-profile after retuning block sizes.",
            }
        ],
        "metadata": {},
    }


@pytest.mark.parametrize(
    "fmt, expected_fragments",
    [
        ("text", ["ROCPD AI PERFORMANCE ANALYSIS", "tile_mfma_loop", "Tune MFMA tile sizes"]),
        ("markdown", ["# PerfXpert AI Performance Analysis", "tile_mfma_loop", "Tune MFMA tile sizes"]),
        ("webview", ["<!DOCTYPE html>", "tile_mfma_loop", "Tune MFMA tile sizes"]),
    ],
)
def test_execute_agentic_runs_analysis_and_formats_reports(fmt, expected_fragments, capsys):
    """Database-backed CLI analysis must route through the public API root."""
    fake_input = mock.Mock()
    fake_input._paths = ["/tmp/fake.db"]

    with mock.patch("perfxpert.api.agent_root", return_value=_root_output()) as agent_root:
        with mock.patch(
            "perfxpert.analysis.payload.build_analysis_payload",
            return_value=_analysis_payload(),
        ) as build_payload:
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
    agent_root.assert_called_once()
    root_kwargs = agent_root.call_args.kwargs
    assert root_kwargs["user_query"] == "why is matmul slow?"
    assert root_kwargs["database_path"] == "/tmp/fake.db"
    assert root_kwargs["provider"] == "openai"
    assert root_kwargs["airgap"] is False
    assert root_kwargs["analysis_options"] == {
        "top_kernels": 3,
        "att_dir": "/tmp/att",
        "min_duration": 5000.0,
        "llm_model": "gpt-4.1",
        "llm_thinking": 8000,
        "llm_local": "ollama",
        "llm_local_model": "codellama:13b",
        "verbose": True,
    }
    assert root_kwargs["progress_callback"] is None
    build_payload.assert_called_once_with(
        fake_input,
        source_dir=None,
        att_dir="/tmp/att",
        top_kernels=3,
        min_duration=5000.0,
        progress_callback=mock.ANY,
    )


def test_execute_agentic_renders_structured_json_from_analysis_outputs(capsys):
    """JSON output should come from the canonical analysis formatter stack."""
    fake_input = mock.Mock()
    fake_input._paths = ["/tmp/fake.db"]

    with mock.patch("perfxpert.api.agent_root", return_value=_root_output()):
        with mock.patch(
            "perfxpert.analysis.payload.build_analysis_payload",
            return_value=_analysis_payload(),
        ):
            analyze_mod._execute_agentic(input=fake_input, format="json")

    captured = capsys.readouterr()
    import json

    payload = json.loads(captured.out)
    assert payload["summary"]["primary_bottleneck"] == "compute"
    assert payload["hotspots"][0]["name"] == "tile_mfma_loop"
    assert payload["recommendations"][0]["issue"] == "Tune MFMA tile sizes"
    assert payload["metadata"]["database_file"] == "/tmp/fake.db"
    assert "execution_breakdown" in payload
    assert "kernel_time_ns" in payload["execution_breakdown"]


def test_execute_agentic_preserves_supported_opencode_provider(capsys):
    """The shipped analyze-provider set still includes ``opencode``."""
    fake_input = mock.Mock()
    fake_input._paths = ["/tmp/fake.db"]

    with mock.patch("perfxpert.api.agent_root", return_value=_root_output()) as agent_root:
        with mock.patch(
            "perfxpert.analysis.payload.build_analysis_payload",
            return_value=_analysis_payload(),
        ):
            analyze_mod._execute_agentic(
                input=fake_input,
                output_format="text",
                llm_provider="opencode",
                enable_llm=True,
            )

    captured = capsys.readouterr()
    assert "ROCPD AI PERFORMANCE ANALYSIS" in captured.out
    assert agent_root.call_args.kwargs["provider"] == "opencode"
    assert agent_root.call_args.kwargs["analysis_options"] == {
        "top_kernels": 10,
        "min_duration": 0.0,
    }


def test_execute_agentic_preserves_provider_taxonomy():
    """Auth and rate-limit errors should propagate unchanged to callers."""
    from perfxpert.providers._exceptions import AuthError

    fake_input = mock.Mock()
    fake_input._paths = ["/tmp/fake.db"]

    with mock.patch(
        "perfxpert.api.agent_root",
        side_effect=AuthError("openai", "bad key"),
    ):
        with mock.patch("perfxpert.analysis.payload.build_analysis_payload") as build_payload:
            with pytest.raises(AuthError):
                analyze_mod._execute_agentic(
                    input=fake_input,
                    output_format="text",
                    llm_provider="openai",
                    enable_llm=True,
                )
    build_payload.assert_not_called()


def test_legacy_symbols_are_absent():
    """Regression guard: removed legacy symbols must stay gone."""
    assert not hasattr(analyze_mod, "_execute_legacy"), (
        "_execute_legacy was removed during the agentic refactor and must stay gone"
    )
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("perfxpert.ai_analysis")


def test_execute_agentic_json_output_parity_across_airgap_and_llm(capsys):
    """Product-surface parity guard for ``perfxpert analyze`` JSON output.

    Airgap and LLM mode may differ in narrative phrasing, but the rendered
    analysis verdict, hotspots, recommendations, and execution breakdown must
    stay identical at the CLI output surface.
    """
    fake_input = mock.Mock()
    fake_input._paths = ["/tmp/fake.db"]

    airgap_root = _root_output()
    airgap_root["narrative"] = "Airgap narrative."
    llm_root = _root_output()
    llm_root["narrative"] = "LLM narrative."

    seen_calls = []

    def _fake_agent_root(**kwargs):
        seen_calls.append(kwargs)
        return airgap_root if kwargs.get("airgap") else llm_root

    with mock.patch("perfxpert.api.agent_root", side_effect=_fake_agent_root):
        with mock.patch(
            "perfxpert.analysis.payload.build_analysis_payload",
            return_value=_analysis_payload(),
        ):
            analyze_mod._execute_agentic(
                input=fake_input,
                output_format="json",
                enable_llm=False,
            )
            airgap_payload = json.loads(capsys.readouterr().out)

            analyze_mod._execute_agentic(
                input=fake_input,
                output_format="json",
                enable_llm=True,
                llm_provider="openai",
            )
            llm_payload = json.loads(capsys.readouterr().out)

    assert seen_calls[0]["airgap"] is True
    assert seen_calls[1]["airgap"] is False
    assert seen_calls[1]["provider"] == "openai"

    assert airgap_payload["summary"]["primary_bottleneck"] == llm_payload["summary"]["primary_bottleneck"]
    assert airgap_payload["hotspots"] == llm_payload["hotspots"]
    assert airgap_payload["recommendations"] == llm_payload["recommendations"]
    assert airgap_payload["execution_breakdown"] == llm_payload["execution_breakdown"]
