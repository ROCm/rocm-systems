# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Offline Navi48 checks that the generic tests do not cover.

Every counter the gfx1201 panels use must be defined for gfx1201, and the
gfx1201 metric formulas must give the expected values on sample counter data.
Passing these tests does not establish SDK exposure or hardware measurement
units; those require validation on the supported ROCm stack.
"""

from typing import Dict, Set

import pandas as pd
import pytest
import yaml

import config
from rocprof_compute_soc.soc_base import iter_analysis_config_metrics
from utils.metrics.expression import build_eval_string
from utils.metrics.metric_evaluator import MetricEvaluator
from utils.utils_counter_defs import parse_counters_text

ARCH = "gfx1201"
SDK_CONFIG = (
    config.rocprof_compute_home
    / "rocprof_compute_soc"
    / "profile_configs"
    / "sdk_config.yaml"
)
PANEL_DIR = config.analysis_configs_dir / ARCH


@pytest.fixture
def dispatch_rate_evaluator():
    """Three dispatches lasting 1, 2, and 4 microseconds, with distinct traffic."""
    return MetricEvaluator(
        pd.DataFrame({
            "Start_Timestamp": [100000, 200000, 300000],
            "End_Timestamp": [101000, 202000, 304000],
            "SQC_LDS_LOAD_DWORDS_sum": [20, 30, 40],
            "SQC_LDS_STORE_DWORDS_sum": [7, 11, 13],
            "SQC_LDS_ATOMIC_DWORDS_sum": [2, 3, 5],
            "SQ_INSTS_VALU_sum": [1000, 4000, 14000],
            "SQ_INSTS_VALU_DP_sum": [100, 600, 2800],
        }),
        {},
        {},
    )


@pytest.fixture
def wait_domain_evaluator():
    """Same-dispatch counters that expose the unvalidated SQ/SQG ratio."""
    return MetricEvaluator(
        pd.DataFrame({
            "SQG_WAVE_CYCLES_sum": [1123998562, 1158838422, 1056240211],
            "SQ_WAVES_sum": [543208, 543160, 543144],
        }),
        {},
        {},
    )


def panel_metric(filename: str, table_id: int, name: str) -> Dict:
    """Load a named metric from the committed architecture configuration."""
    panel = yaml.safe_load((PANEL_DIR / filename).read_text(encoding="utf-8"))
    for source in panel["Panel Config"]["data source"]:
        for table in source.values():
            if table["id"] == table_id:
                return table["metric"][name]
    raise AssertionError(f"Metric table {table_id} was not found in {filename}")


@pytest.fixture(scope="module")
def gfx1201_definitions() -> Dict[str, Dict]:
    """Return each counter's gfx1201 definition."""
    sdk = yaml.safe_load(SDK_CONFIG.read_text(encoding="utf-8"))
    return {
        counter["name"]: definition
        for counter in sdk["rocprofiler-sdk"]["counters"]
        for definition in counter["definitions"]
        if ARCH in definition["architectures"]
    }


@pytest.fixture(scope="module")
def panel_counters() -> Set[str]:
    """Return the counters used by gfx1201 analysis metric expressions."""
    counters: Set[str] = set()
    for metric in iter_analysis_config_metrics(PANEL_DIR, set()):
        for expression in metric.expressions:
            counters.update(parse_counters_text(expression)[0])
    return counters


def test_gfx1201_panel_counters_are_defined(panel_counters, gfx1201_definitions):
    assert not panel_counters - gfx1201_definitions.keys()


@pytest.mark.parametrize(
    ("cycle_counts", "expected_total"),
    [
        pytest.param([0, 0, 0], 0, id="zero-count"),
        pytest.param([42, 205, 7], 254, id="distinct-dispatch-counts"),
    ],
)
def test_gfx1201_valu_cycles_reports_raw_total_without_peak(
    cycle_counts, expected_total
):
    evaluator = MetricEvaluator(
        pd.DataFrame({
            "SQ_INST_CYCLES_VALU_sum": cycle_counts,
            "Start_Timestamp": [100000, 200000, 300000],
            "End_Timestamp": [101000, 202000, 304000],
            "SQ_WAVES_sum": [3, 5, 11],
        }),
        {},
        {},
    )
    definition = panel_metric(
        "0700_workgroup_processor_wgp.yaml", 708, "SQ_INST_CYCLES_VALU"
    )

    assert definition["unit"] == "Cycles"
    assert definition["peak"] == "None"
    assert definition["pct_of_peak"] is False
    assert (
        evaluator.eval_expression(build_eval_string(definition["value"]))
        == expected_total
    )


@pytest.mark.parametrize(
    "filename, table_id, metric, statistic",
    [
        ("0700_workgroup_processor_wgp.yaml", 711, "Wait for Any", "avg"),
        ("0700_workgroup_processor_wgp.yaml", 711, "Wait for Any", "min"),
        ("0700_workgroup_processor_wgp.yaml", 711, "Wait for Any", "max"),
    ],
)
def test_gfx1201_unvalidated_wait_percentages_display_unavailable(
    wait_domain_evaluator, filename, table_id, metric, statistic
):
    expression = panel_metric(filename, table_id, metric)[statistic]
    assert wait_domain_evaluator.eval_expression(build_eval_string(expression)) == "N/A"


def test_gfx1201_wait_mitigation_preserves_wave_life(wait_domain_evaluator):
    expression = panel_metric("0700_workgroup_processor_wgp.yaml", 704, "Wave Life")[
        "avg"
    ]
    assert wait_domain_evaluator.eval_expression(
        build_eval_string(expression)
    ) == pytest.approx(3339077195 / 1629512)


def test_gfx1201_lds_bandwidth_counts_four_bytes_per_load_store_and_atomic_dword(
    dispatch_rate_evaluator,
):
    definition = panel_metric("0300_memory_chart.yaml", 304, "LDS Bandwidth")

    # 90 load, 31 store, and 10 atomic dwords carry 524 bytes over 7 microseconds.
    assert definition["unit"] == "Bytes/s"
    assert dispatch_rate_evaluator.eval_expression(
        build_eval_string(definition["value"])
    ) == pytest.approx(524 / 7e-6)


@pytest.mark.parametrize(
    ("metric_name", "expected_rate"),
    [("VALU Issue Rate", 19 / 7), ("FP64 VALU Issue Rate", 0.5)],
)
def test_gfx1201_valu_issue_rate_reports_instruction_count_per_nanosecond(
    dispatch_rate_evaluator, metric_name, expected_rate
):
    definition = panel_metric("0200_system_speed_of_light.yaml", 201, metric_name)

    assert definition["unit"] == "GInstr/s"
    assert definition["pct_of_peak"] is False
    assert dispatch_rate_evaluator.eval_expression(
        build_eval_string(definition["value"])
    ) == pytest.approx(expected_rate)
