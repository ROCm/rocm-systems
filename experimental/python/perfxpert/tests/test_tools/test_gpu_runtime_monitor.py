"""Tests for perfxpert.tools.gpu_runtime_monitor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perfxpert.tools import gpu_runtime_monitor
from perfxpert.tools._class import ToolClass


def test_all_fns_are_read_only_class():
    for fn in (
        gpu_runtime_monitor.parse_amd_smi_json,
        gpu_runtime_monitor.parse_rocm_smi_json,
        gpu_runtime_monitor.analyze_thermal,
    ):
        assert fn.__tool_class__ == ToolClass.READ_ONLY


def test_parse_amd_smi_happy_path(tmp_path):
    payload = {
        "gpu": [
            {
                "gpu_id": 0,
                "temp_edge_c": 72.5,
                "power_w": 410.0,
                "sclk_mhz": 1950,
                "mclk_mhz": 1600,
                "gfx_busy_pct": 96,
            },
            {
                "gpu_id": 1,
                "temp_edge_c": 71.0,
                "power_w": 400.0,
                "sclk_mhz": 1950,
                "mclk_mhz": 1600,
                "gfx_busy_pct": 93,
            },
        ]
    }
    p = tmp_path / "amd.json"
    p.write_text(json.dumps(payload))
    res = gpu_runtime_monitor.parse_amd_smi_json(str(p))
    assert res["source"] == "amd-smi"
    assert res["gpu_count"] == 2
    assert len(res["samples"]) == 2
    assert res["samples"][0]["temp_c"] == 72.5


def test_parse_amd_smi_edge_case_missing_file(tmp_path):
    res = gpu_runtime_monitor.parse_amd_smi_json(str(tmp_path / "no_such.json"))
    assert res == {"source": "amd-smi", "samples": [], "gpu_count": 0}


def test_parse_rocm_smi_happy_path(tmp_path):
    payload = {
        "card0": {
            "Temperature (Sensor edge) (C)": "65.0",
            "Average Graphics Package Power (W)": "300",
            "sclk": "1800Mhz",
            "mclk": "1600Mhz",
            "GPU use (%)": 88,
        }
    }
    p = tmp_path / "rocm.json"
    p.write_text(json.dumps(payload))
    res = gpu_runtime_monitor.parse_rocm_smi_json(str(p))
    assert res["source"] == "rocm-smi"
    assert res["gpu_count"] == 1
    assert res["samples"][0]["sclk_mhz"] == 1800.0
    assert res["samples"][0]["gfx_busy_pct"] == 88.0


def test_parse_rocm_smi_edge_case_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    res = gpu_runtime_monitor.parse_rocm_smi_json(str(p))
    assert res["samples"] == []


def test_analyze_thermal_healthy_verdict():
    metrics = {"samples": [
        {"gpu_id": 0, "temp_c": 60.0, "power_w": 200.0},
        {"gpu_id": 0, "temp_c": 62.0, "power_w": 210.0},
    ]}
    out = gpu_runtime_monitor.analyze_thermal(metrics, tjmax_c=105.0)
    assert out["verdict"] == "healthy"
    assert out["throttle_events"] == 0
    assert out["max_temp_c"] == 62.0


def test_analyze_thermal_throttling_verdict():
    # Temp at tjmax_c - throttle_margin triggers throttling verdict.
    metrics = {"samples": [
        {"gpu_id": 0, "temp_c": 100.0, "power_w": 300.0},
    ]}
    out = gpu_runtime_monitor.analyze_thermal(metrics, tjmax_c=105.0)
    assert out["verdict"] == "throttling"
    assert out["throttle_events"] >= 1


def test_analyze_thermal_empty_samples_edge_case():
    out = gpu_runtime_monitor.analyze_thermal({"samples": []})
    assert out["verdict"] == "healthy"
    assert out["max_temp_c"] == 0.0


def test_resolve_monitor_log_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv(
        gpu_runtime_monitor.PERFXPERT_GPU_MONITOR_LOG, str(tmp_path / "log.json")
    )
    assert gpu_runtime_monitor.resolve_monitor_log_path().endswith("log.json")


def test_resolve_monitor_log_path_unset(monkeypatch):
    monkeypatch.delenv(gpu_runtime_monitor.PERFXPERT_GPU_MONITOR_LOG, raising=False)
    assert gpu_runtime_monitor.resolve_monitor_log_path() == ""
