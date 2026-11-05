# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Checks the gfx1201 (RDNA 4, Navi 48) counter definitions in sdk_config.yaml.

Event IDs come from the gfx1201 register spec. No GPU hardware required.
"""

import re

import pytest
import yaml

import config
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_counter_defs import (
    counter_to_block,
    metric_expressions,
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

# Counters whose gfx11 event IDs name a different quantity in the gfx1201
# register spec, or that the spec does not define at all.
ABSENT_COUNTERS = [
    "SQ_WAVE_CYCLES",
    "SQ_WAIT_INST_ANY",
    "SQ_INST_CYCLES_VALU",
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

# (counter, block, event) as the gfx1201 register spec defines them
SPEC_EVENTS = [
    ("SQ_WMMA_VALU_INSTS", "SQ", 233),
    ("SQ_INST_CYCLES_VALU_WMMA", "SQ", 256),
    ("SQ_LDS_BANK_CONFLICT", "SQ", 288),
    ("SQ_LDS_IDX_ACTIVE", "SQ", 293),
    ("SQC_SMEM_LOAD_DWORDS", "SQ", 370),
    ("SQG_WAVE_CYCLES", "SQG", 16),
    ("GL1C_GL2_RET_32B", "GL1C", 67),
    ("GL1C_STALL_BUFFER_FULL", "GL1C", 72),
    ("GL2C_WRITE", "GL2C", 6),
    ("GL2C_WRITE_SECTORS", "GL2C", 101),
    ("GL2C_EA_RDREQ_256B", "GL2C", 149),
    ("GL2C_EA_RDREQ_DRAM", "GL2C", 165),
    ("GL2C_EA_WRREQ_DRAM", "GL2C", 166),
    ("GC_EA_SE_DRAM_RD_CMD_POP", "GCEA_SE", 11),
]


def gfx1201_definitions() -> dict[str, dict]:
    """The gfx1201 definition of every counter that has one."""
    sdk = yaml.safe_load(SDK_CONFIG.read_text(encoding="utf-8"))
    return {
        counter["name"]: definition
        for counter in sdk["rocprofiler-sdk"]["counters"]
        for definition in counter["definitions"]
        if ARCH in definition["architectures"]
    }


def panel_counters() -> set[str]:
    """Every counter named by a gfx1201 analysis config metric."""
    counters: set[str] = set()
    for path in sorted(PANEL_DIR.glob("*.yaml")):
        panel = yaml.safe_load(path.read_text(encoding="utf-8"))["Panel Config"]
        for source in panel["data source"]:
            for table in source.values():
                if "header" not in table:
                    continue
                for metric in (table.get("metric") or {}).values():
                    for expression in metric_expressions(table, metric):
                        counters |= parse_counters_text(expression)[0]
    return counters


def hardware_counters(name: str, definitions: dict[str, dict]) -> set[str]:
    """The hardware counters a counter is defined from."""
    definition = definitions[name]
    if "block" in definition:
        return {name}
    found: set[str] = set()
    for token in EXPRESSION_TOKEN_RE.findall(definition["expression"]):
        if token in definitions:
            found |= hardware_counters(token, definitions)
    return found


DEFINITIONS = gfx1201_definitions()
PANEL_COUNTERS = sorted(panel_counters())


@pytest.mark.parametrize("counter", PANEL_COUNTERS)
def test_gfx1201_panel_counters_are_defined(counter):
    assert counter in DEFINITIONS


@pytest.mark.parametrize(
    "counter", [name for name in PANEL_COUNTERS if name in DEFINITIONS]
)
def test_gfx1201_panel_counters_have_perfmon_slots(counter):
    perfmon = mi_gpu_specs.get_perfmon_config(ARCH)
    for hardware in hardware_counters(counter, DEFINITIONS):
        assert counter_to_block(hardware) in perfmon


@pytest.mark.parametrize("counter", ABSENT_COUNTERS)
def test_gfx1201_counters_missing_from_register_spec_are_undefined(counter):
    assert counter not in DEFINITIONS


@pytest.mark.parametrize("counter, block, event", SPEC_EVENTS)
def test_gfx1201_event_ids_match_register_spec(counter, block, event):
    assert DEFINITIONS[counter] == {
        "architectures": DEFINITIONS[counter]["architectures"],
        "block": block,
        "event": event,
    }
