# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Offline Navi48 counter and panel contracts.

Expected mappings follow the gfx1201 enablement baseline and register-spec
comparisons. Passing these tests does not establish SDK exposure or hardware
measurement units; those require validation on the supported ROCm stack.
"""

import re
from typing import Dict, Set

import pandas as pd
import pytest
import yaml

import config
from rocprof_compute_soc.soc_base import iter_analysis_config_metrics
from utils.metrics.expression import build_eval_string
from utils.metrics.metric_evaluator import MetricEvaluator
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import parse_sets_yaml
from utils.utils_counter_defs import (
    counter_to_block,
    parse_counters_text,
)

ARCH = "gfx1201"
SDK_CONFIG = (
    config.rocprof_compute_home
    / "rocprof_compute_soc"
    / "profile_configs"
    / "sdk_config.yaml"
)
PANEL_DIR = config.analysis_configs_dir / ARCH
EXPRESSION_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]*\b")

# These gfx115x definitions are deliberately not exposed by this port. Some
# names exist in the public gfx12 spec with changed IDs or units, so this list
# is a validation boundary for the port, not proof of architectural absence.
UNPORTED_COUNTERS = [
    "SQ_WAVE_CYCLES",
    "SQC_ICACHE_MISSES",
    "SQC_DCACHE_MISSES",
    "GL1C_REQ_MISS",
    "GL1C_STALL_GL2_GL1",
    "GL2C_COMPRESSED_READ_96_REQ",
    "GL2C_EA_RDREQ_96B",
    "GL2C_MC_WRREQ",
    "GCEA_SARB_BUSY",
    "GCEA_RDRAM_SIZE_REQ",
    "SPI_CSN_WAVE",
    "CPC_GUS_READ_REQUEST_SENT",
]

# (counter, SDK block, event) retained or remapped for the Navi48 port.
EVENT_MAPPINGS = [
    ("SQ_WAIT_INST_ANY", "SQ", 26),
    ("SQ_WAIT_ANY", "SQ", 27),
    ("SQ_INST_CYCLES_VALU", "SQ", 99),
    ("SQ_WMMA_VALU_INSTS", "SQ", 233),
    ("SQ_INST_CYCLES_VALU_WMMA", "SQ", 256),
    ("SQ_LDS_BANK_CONFLICT", "SQ", 288),
    ("SQ_LDS_IDX_ACTIVE", "SQ", 293),
    ("SQC_SMEM_LOAD_DWORDS", "SQ", 370),
    ("SQG_WAVE_CYCLES", "SQG", 16),
    ("GL1C_GL2_RET_32B", "GL1C", 67),
    ("GL1C_GL1_RET_32B", "GL1C", 69),
    ("GL1C_STALL_BUFFER_FULL", "GL1C", 72),
    ("GL2C_WRITE", "GL2C", 6),
    ("GL2C_WRITE_SECTORS", "GL2C", 101),
    ("GL2C_EA_RDREQ_256B", "GL2C", 149),
    ("GL2C_EA_RDREQ_DRAM", "GL2C", 165),
    ("GL2C_EA_WRREQ_DRAM", "GL2C", 166),
    ("GL2C_EA_RDREQ_DRAM_32B", "GL2C", 167),
    ("GL2C_EA_WRREQ_DRAM_32B", "GL2C", 168),
    ("GC_EA_SE_DRAM_RD_CMD_POP", "GCEA_SE", 11),
]


@pytest.fixture
def dispatch_rate_evaluator():
    """Three dispatches lasting 1, 2, and 4 microseconds, with distinct traffic."""
    return MetricEvaluator(
        pd.DataFrame({
            "Start_Timestamp": [100000, 200000, 300000],
            "End_Timestamp": [101000, 202000, 304000],
            "GL2C_READ_32_REQ_sum": [1, 0, 2],
            "GL2C_READ_64_REQ_sum": [2, 1, 0],
            "GL2C_READ_128_REQ_sum": [0, 2, 1],
            "GL2C_READ_256_REQ_sum": [1, 3, 4],
            "GL2C_WRITE_32_REQ_sum": [0, 2, 1],
            "GL2C_WRITE_64_REQ_sum": [1, 0, 2],
            "GL2C_WRITE_128_REQ_sum": [2, 1, 0],
            "GL2C_WRITE_256_REQ_sum": [1, 0, 3],
            "GL2C_EA_RDREQ_32B_sum": [10, 0, 0],
            "GL2C_EA_RDREQ_64B_sum": [0, 10, 0],
            "GL2C_EA_RDREQ_128B_sum": [0, 0, 10],
            "GL2C_EA_RDREQ_256B_sum": [0, 0, 20],
            "GL2C_EA_WRREQ_32B_sum": [0, 5, 0],
            "GL2C_EA_WRREQ_64B_sum": [0, 0, 10],
            "GL2C_EA_WRREQ_128B_sum": [5, 0, 0],
            "GL2C_EA_WRREQ_256B_sum": [10, 20, 30],
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
            "SQ_WAIT_ANY_sum": [1711739396, 1696050988, 1764241951],
            "SQ_WAIT_INST_ANY_sum": [1711739396, 1696050988, 1764241951],
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


def hardware_counters(name: str, definitions: Dict[str, Dict]) -> Set[str]:
    """Resolve a derived counter to its hardware counter dependencies."""
    definition = definitions[name]
    if "block" in definition:
        return {name}
    found: Set[str] = set()
    for token in EXPRESSION_TOKEN_RE.findall(definition["expression"]):
        if token in definitions:
            found |= hardware_counters(token, definitions)
    return found


def test_gfx1201_panel_counter_contract_is_nonempty(
    panel_counters, gfx1201_definitions
):
    assert panel_counters
    assert gfx1201_definitions


def test_gfx1201_panel_counters_are_defined(panel_counters, gfx1201_definitions):
    assert not panel_counters - gfx1201_definitions.keys()


def test_gfx1201_panel_counters_have_perfmon_slots(panel_counters, gfx1201_definitions):
    perfmon = mi_gpu_specs.get_perfmon_config(ARCH)
    requested_hardware = set()
    for counter in panel_counters:
        requested_hardware.update(hardware_counters(counter, gfx1201_definitions))
    missing_slots = {
        counter
        for counter in requested_hardware
        if counter_to_block(counter) not in perfmon
    }
    assert not missing_slots


def test_gfx1201_sets_include_available_metrics():
    active_metrics = set()
    for metric in iter_analysis_config_metrics(PANEL_DIR, set()):
        if any(expression != "None" for expression in metric.expressions):
            active_metrics.add(
                f"{int(metric.stem_id) // 100}.{metric.table_id % 100}."
                f"{metric.metric_idx}"
            )
    for set_name, metric_set in parse_sets_yaml(ARCH).items():
        metric_ids = {
            metric_id for metric in metric_set["metric"] for metric_id in metric
        }
        assert metric_ids & active_metrics, (
            f"{set_name} contains only unavailable metrics"
        )


@pytest.mark.parametrize("counter", UNPORTED_COUNTERS)
def test_gfx1201_does_not_inherit_unvalidated_gfx115x_definitions(
    counter, gfx1201_definitions
):
    assert counter not in gfx1201_definitions


@pytest.mark.parametrize("counter, block, event", EVENT_MAPPINGS)
def test_gfx1201_event_ids_match_enablement_mapping(
    counter, block, event, gfx1201_definitions
):
    assert gfx1201_definitions[counter] == {
        "architectures": gfx1201_definitions[counter]["architectures"],
        "block": block,
        "event": event,
    }


def test_gfx1201_valu_cycles_sum_preserves_raw_sdk_count(gfx1201_definitions):
    definition = gfx1201_definitions["SQ_INST_CYCLES_VALU_sum"]

    assert definition["expression"] == "reduce(SQ_INST_CYCLES_VALU,sum)"
    assert hardware_counters("SQ_INST_CYCLES_VALU_sum", gfx1201_definitions) == {
        "SQ_INST_CYCLES_VALU"
    }


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
        ("0200_system_speed_of_light.yaml", 201, "Wave Dependency Wait", "value"),
        ("0200_system_speed_of_light.yaml", 201, "Wave Issue Wait", "value"),
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


@pytest.mark.parametrize(
    ("request_metric", "fabric_metric", "request_rates", "fabric_rate"),
    [
        pytest.param(
            "GL2 Cache Read BW",
            "GL2-Fabric Read BW",
            {"avg": 2720 / 7e-6, "min": 304e6, "max": 544e6},
            7360 / 7e-6,
            id="read",
        ),
        pytest.param(
            "GL2 Cache Write BW",
            "GL2-Fabric Write BW",
            {"avg": 1696 / 7e-6, "min": 96e6, "max": 576e6},
            16800 / 7e-6,
            id="write",
        ),
    ],
)
def test_gfx1201_gl2_request_bandwidth_uses_size_weighted_output_fifo_counters(
    dispatch_rate_evaluator, request_metric, fabric_metric, request_rates, fabric_rate
):
    # Requests carry 416/1088/1216 read bytes or 576/192/928 write bytes.
    # Avg divides total bytes by total duration; min/max use each dispatch's rate.
    request_definition = panel_metric("1300_gl2_cache.yaml", 1304, request_metric)
    actual_rates = {
        statistic: dispatch_rate_evaluator.eval_expression(
            build_eval_string(request_definition[statistic])
        )
        for statistic in request_rates
    }
    assert request_definition["unit"] == "Bytes/s"
    assert actual_rates == pytest.approx(request_rates)

    fabric_definition = panel_metric(
        "0200_system_speed_of_light.yaml", 201, fabric_metric
    )
    assert dispatch_rate_evaluator.eval_expression(
        build_eval_string(fabric_definition["value"])
    ) == pytest.approx(fabric_rate)


def test_gfx1201_lds_bandwidth_counts_four_bytes_per_load_store_and_atomic_dword(
    dispatch_rate_evaluator,
):
    definition = panel_metric("0300_memory_chart.yaml", 304, "LDS Bandwidth")

    # 90 load, 31 store, and 10 atomic dwords carry 524 bytes over 7 microseconds.
    assert definition["unit"] == "Bytes/s"
    assert dispatch_rate_evaluator.eval_expression(
        build_eval_string(definition["value"])
    ) == pytest.approx(524 / 7e-6)


def test_gfx1201_sol_gl2_bandwidth_reports_fifo_rate_without_unqualified_peak(
    dispatch_rate_evaluator,
):
    definition = panel_metric("0200_system_speed_of_light.yaml", 201, "GL2 Cache BW")

    assert definition["unit"] == "Bytes/s"
    assert definition["peak"] == "None"
    assert definition["pct_of_peak"] is False
    assert dispatch_rate_evaluator.eval_expression(
        build_eval_string(definition["value"])
    ) == pytest.approx(2720 / 7e-6)


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
