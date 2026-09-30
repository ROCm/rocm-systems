"""Tests for knowledge/metric_thresholds.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "metric_thresholds.schema.json"
)


def test_metric_thresholds_loads():
    data = load_yaml("metric_thresholds")
    assert isinstance(data, dict)
    assert len(data) >= 1


def test_metric_thresholds_validates():
    data = load_yaml("metric_thresholds")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_gpu_utilization_threshold_present():
    data = load_yaml("metric_thresholds")
    assert "gpu_utilization_pct" in data


def test_wave_occupancy_threshold_present():
    data = load_yaml("metric_thresholds")
    assert "wave_occupancy_pct" in data or "avg_waves_per_cu" in data


def test_high_exceeds_medium_exceeds_low():
    """Each metric must have monotonic thresholds: high > medium > low."""
    data = load_yaml("metric_thresholds")
    for name, entry in data.items():
        assert entry["high"] > entry["medium"] > entry["low"], (
            f"metric {name} has non-monotonic thresholds"
        )
