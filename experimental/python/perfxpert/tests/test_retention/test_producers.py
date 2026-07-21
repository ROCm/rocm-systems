from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from perfxpert import analyze
from perfxpert.agents import runtime
from perfxpert.output_config import output_config
from perfxpert.providers._exceptions import TransientError
from perfxpert.tools.knowledge_history import query_knowledge


def _analysis_payload():
    return {
        "database_path": "",
        "database_paths": [],
        "time_breakdown": {"kernel_percent": 90.0},
        "hotspots": [
            {
                "name": "kernel_retained",
                "percent_of_total": 40.0,
                "total_duration_ns": 400,
            }
        ],
        "memory_analysis": {},
        "hardware_counters": {
            "has_counters": True,
            "metrics": {},
            "counters": {},
        },
        "kernel_resources": {},
        "api_overhead": {},
        "warmup_issues": {},
        "thread_trace": None,
        "tier0_findings": None,
        "recommendations_deterministic": [
            {
                "category": "Compute-Bound Kernel",
                "title": "Tune the kernel",
            }
        ],
        "metadata": {},
        "communication": None,
        "roofline": None,
    }


def test_execute_agentic_records_analysis_and_prediction_once(tmp_path, capsys):
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    fake_input = SimpleNamespace(_paths=[trace])
    root_output = {
        "narrative": "Deterministic result.",
        "recommendations": [],
        "primary_bottleneck": "compute",
        "warnings": [],
        "metadata": {},
    }

    with mock.patch("perfxpert.api.agent_root", return_value=root_output):
        with mock.patch(
            "perfxpert.analysis.payload.build_analysis_payload",
            return_value=_analysis_payload(),
        ):
            analyze._execute_agentic(
                input=fake_input,
                output_format="json",
                enable_llm=False,
                config=output_config(output_file="-"),
            )
    capsys.readouterr()

    analyses = query_knowledge(kind="trace_analysis")
    predictions = query_knowledge(kind="prediction")
    assert len(analyses) == 1
    assert len(predictions) == 1
    assert analyses[0]["seen_count"] == 1
    assert predictions[0]["payload"]["prediction"]["baseline_db"] == trace.name


def test_splice_uses_precomputed_diff_without_recomputing():
    diff_result = {
        "schema_version": "0.3.1",
        "baseline_db": "before.db",
        "new_db": "after.db",
        "wall_delta_ns": 0,
        "wall_delta_pct": 0.0,
        "verdict": "neutral",
        "per_kernel": [],
        "primary_regressions": [],
        "primary_improvements": [],
        "narrative": "No material change.",
    }
    with mock.patch(
        "perfxpert.tools.trace_diff.diff_runs",
        side_effect=AssertionError("must not recompute"),
    ):
        rendered = analyze._splice_baseline_diff(
            "report",
            baseline_db="before.db",
            new_db="after.db",
            output_format="text",
            top_kernels=20,
            diff_result=diff_result,
        )

    assert "No material change." in rendered


def test_retention_setup_failure_does_not_abort_analysis(tmp_path, capsys):
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    fake_input = SimpleNamespace(_paths=[trace])
    root_output = {
        "narrative": "Result remains available.",
        "recommendations": [],
        "primary_bottleneck": "compute",
        "warnings": [],
        "metadata": {},
    }

    with mock.patch(
        "perfxpert.retention.build_retention_policy",
        side_effect=ValueError("bad retention config"),
    ):
        with mock.patch("perfxpert.api.agent_root", return_value=root_output):
            with mock.patch(
                "perfxpert.analysis.payload.build_analysis_payload",
                return_value=_analysis_payload(),
            ):
                analyze._execute_agentic(
                    input=fake_input,
                    output_format="json",
                    enable_llm=False,
                    config=output_config(output_file="-"),
                )

    captured = capsys.readouterr()
    assert "Result remains available." in captured.out
    assert "knowledge retention setup failed" in captured.err


def test_runtime_reports_actual_fallback_provider_and_model(monkeypatch):
    session = runtime.AnalysisSession(
        session_id="test",
        provider="openai",
        providers=("openai", "anthropic"),
        airgap=False,
    )
    monkeypatch.setattr(
        "perfxpert.agents.framework._resolve_model",
        lambda provider: f"resolved/{provider}",
    )

    def _call(provider):
        if provider == "openai":
            raise TransientError(provider, message="retry")
        return "ok"

    runtime.clear_last_provider_execution()
    assert session._run_live(_call) == "ok"
    assert runtime.last_provider_execution() == (
        "anthropic",
        "resolved/anthropic",
    )

    opencode_session = runtime.AnalysisSession(
        session_id="opencode",
        provider="opencode",
        providers=("opencode",),
        airgap=False,
    )
    runtime.clear_last_provider_execution()
    assert opencode_session._run_live(lambda provider: "ok") == "ok"
    assert runtime.last_provider_execution() == (
        "opencode",
        "opencode-default",
    )
