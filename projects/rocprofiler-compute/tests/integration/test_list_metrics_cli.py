# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for --list-metrics and the deprecated --list-blocks option."""

import re
from pathlib import Path
from typing import Dict

import pytest
from common import ROOT

from utils.analysis_views import arch_panel_configs
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import get_arch_panel_id_to_alias

# Profiled with filter_blocks cu_ins and cu_pipe (blocks 10 and 11) on gfx90a
FILTERED_WORKLOAD = Path(ROOT) / "tests" / "workloads" / "ipblocks_SQ" / "MI200"


def first_column_ids(output: str) -> list[str]:
    """Return the ids in the first column of the table rows of output."""
    return re.findall(r"^\| ([0-9.]+) +\|", output, flags=re.MULTILINE)


def arch_panels_from_views(arch: str) -> Dict[str, str]:
    """Return {panel_id_str: title} of arch's views."""
    return {
        str(panel_id // 100): panel["title"]
        for panel_id, panel in arch_panel_configs(arch).items()
    }


# =============================================================================
# TESTS FOR MODELESS COMMAND LINE OPTIONS
# =============================================================================


def test_list_metrics(binary_handler_analyze_rocprof_compute, capsys):
    return_code = binary_handler_analyze_rocprof_compute([
        "--list-metrics",
        "--debug-list-metrics-gfx-id",
        "gfx90a",
    ])
    assert return_code == 0

    # Test output
    output = capsys.readouterr().out
    assert "Platform: gfx90a (MI200)" in output
    assert first_column_ids(output) == list(arch_panels_from_views("gfx90a"))
    assert re.search(r"\| 6 +\| spi +\| Workgroup Manager \(SPI\) +\|", output)


def list_blocks_supported_archs() -> list[str]:
    """Return the GPU archs in mi_gpu_spec.yaml."""
    return list(mi_gpu_specs.get_gpu_series_dict().keys())


@pytest.mark.parametrize("arch", list_blocks_supported_archs())
def test_list_blocks_all_archs(binary_handler_analyze_rocprof_compute, capsys, arch):
    """Verify --list-blocks output matches the views' panels and aliases."""
    args = ["--list-blocks", arch]
    if arch in mi_gpu_specs.get_experimental_archs():
        args.append("--experimental")
    return_code = binary_handler_analyze_rocprof_compute(args)
    assert return_code == 0

    output = capsys.readouterr().out
    assert "--list-blocks is deprecated" in output
    assert "INDEX" in output
    assert "BLOCK ALIAS" in output
    assert "BLOCK NAME" in output

    # Fixed-width parse: empty aliases break whitespace splitting.
    # Derive column offsets from the header so this parser tracks the producer.
    lines = output.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("INDEX"))
    header = lines[header_idx]
    alias_col = header.index("BLOCK ALIAS")
    name_col = header.index("BLOCK NAME")
    block_entries: dict[str, tuple[str, str]] = {}
    for line in lines[header_idx + 1 :]:
        block_id = line[:alias_col].strip()
        if not block_id:
            continue
        alias = line[alias_col:name_col].strip()
        name = line[name_col:].strip()
        block_entries[block_id] = (alias, name)

    expected_panels = arch_panels_from_views(arch)
    assert set(block_entries) == set(expected_panels), (
        f"--list-blocks {arch}: rows {sorted(block_entries)} != "
        f"view panels {sorted(expected_panels)}"
    )

    expected_aliases = get_arch_panel_id_to_alias(arch)
    for panel_id, expected_name in expected_panels.items():
        assert block_entries[panel_id] == (expected_aliases[panel_id], expected_name)


def test_list_metrics_selected_blocks(binary_handler_analyze_rocprof_compute, capsys):
    """Aliases select blocks, each table gets a heading, and per-channel
    tables, which list no metrics, are left out."""
    return_code = binary_handler_analyze_rocprof_compute([
        "--list-metrics",
        "sol",
        "l2_per_channel",
        "--debug-list-metrics-gfx-id",
        "gfx950",
    ])
    assert return_code == 0

    output = capsys.readouterr().out
    assert (
        "\n2. System Speed-of-Light (alias: sol)\n2.1 System Speed-of-Light\n" in output
    )
    assert re.search(
        r"\| 2\.1\.0 +\| VALU FLOPs +\| GFLOP/s +\| The total floating", output
    )
    assert "\n18.1 Aggregate Stats (All channels)\n" in output
    assert "\n18.2 " not in output
    assert {metric_id.split(".")[0] for metric_id in first_column_ids(output)} == {
        "2",
        "18",
    }


def test_list_metrics_defaults_to_current_gpu(
    binary_handler_analyze_rocprof_compute, capsys
):
    return_code = binary_handler_analyze_rocprof_compute(["--list-metrics"])
    assert return_code == 0

    output = capsys.readouterr().out
    assert "Platform: gfx" in output
    assert re.search(r"\| Block_ID +\| Alias +\| Block +\|", output)


def test_list_metrics_from_workload(binary_handler_analyze_rocprof_compute, capsys):
    """analyze -p lists the workload's arch and only the blocks it profiled."""
    return_code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "-p",
        str(FILTERED_WORKLOAD),
        "--list-metrics",
    ])
    assert return_code == 0

    output = capsys.readouterr().out
    assert "Platform: gfx90a (MI200)" in output
    assert first_column_ids(output) == ["0", "1", "10", "11"]
