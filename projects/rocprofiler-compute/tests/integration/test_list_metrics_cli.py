# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for --list-metrics and the deprecated --list-blocks option."""

import re
from pathlib import Path

import pytest
import yaml
from common import ROOT, SRC

import utils.utils_common as utils_common
from utils.mi_gpu_spec import mi_gpu_specs

ANALYSIS_CONFIGS = Path(SRC) / "rocprof_compute_soc" / "analysis_configs"
# Profiled with filter_blocks cu_ins and cu_pipe (blocks 10 and 11) on gfx90a
FILTERED_WORKLOAD = Path(ROOT) / "tests" / "workloads" / "ipblocks_SQ" / "MI200"


def first_column_ids(output: str) -> list[str]:
    """Return the ids in the first column of the grid rows of output."""
    return re.findall(r"^│ ([0-9.]+) ", output, flags=re.MULTILINE)


# =============================================================================
# TESTS FOR MODELESS COMMAND LINE OPTIONS
# =============================================================================


def test_list_metrics(binary_handler_analyze_rocprof_compute, capsys):
    return_code = binary_handler_analyze_rocprof_compute([
        "--list-metrics-gfx-id",
        "gfx90a",
    ])
    assert return_code == 0

    # Test output
    output = capsys.readouterr().out
    assert "Platform: gfx90a (MI200)" in output
    assert re.search(r"│ 6 +│ spi +│ Workgroup Manager \(SPI\) +│", output)
    assert "5.2" not in first_column_ids(output)


def list_blocks_supported_archs() -> list[str]:
    """Return sorted arch names from analysis_configs/gfx* directories."""
    return list(mi_gpu_specs.get_gpu_series_dict().keys())


def arch_panels_from_disk(arch: str) -> dict[str, str]:
    """Return {panel_id_str: title} from per-arch yaml Panel Configs."""
    panels: dict[str, str] = {}
    for yaml_path in sorted((ANALYSIS_CONFIGS / arch).glob("*.yaml")):
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not data or "Panel Config" not in data:
            continue
        panel_config = data["Panel Config"]
        panels[str(panel_config["id"] // 100)] = panel_config["title"]
    return panels


def all_template_aliases_by_panel_id() -> dict[str, set[str]]:
    """Return {panel_id_str: {alias, ...}} from all *_config_template.yaml."""
    aliases: dict[str, set[str]] = {}
    for tpl in sorted(ANALYSIS_CONFIGS.glob("*_config_template.yaml")):
        data = yaml.safe_load(tpl.read_text(encoding="utf-8")) or {}
        for panel in data.get("panels") or []:
            alias = panel.get("panel_alias")
            if alias:
                pid = str(panel.get("panel_id"))
                aliases.setdefault(pid, set()).add(alias)
    return aliases


@pytest.mark.parametrize("arch", list_blocks_supported_archs())
def test_list_blocks_all_archs(binary_handler_analyze_rocprof_compute, capsys, arch):
    """Verify --list-blocks output matches on-disk panels and template aliases."""
    return_code = binary_handler_analyze_rocprof_compute(["--list-blocks", arch])
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

    expected_panels = arch_panels_from_disk(
        utils_common.canonical_config_arch(arch) or arch
    )
    assert set(block_entries) == set(expected_panels), (
        f"--list-blocks {arch}: rows {sorted(block_entries)} != "
        f"on-disk panels {sorted(expected_panels)}"
    )

    valid_aliases = all_template_aliases_by_panel_id()
    for panel_id, expected_name in expected_panels.items():
        actual_alias, actual_name = block_entries[panel_id]
        assert actual_name == expected_name, (
            f"--list-blocks {arch} panel {panel_id}: name "
            f"{actual_name!r} != on-disk title {expected_name!r}"
        )
        if actual_alias:
            allowed = valid_aliases.get(panel_id, set())
            assert actual_alias in allowed, (
                f"--list-blocks {arch} panel {panel_id}: alias "
                f"{actual_alias!r} not declared in any template "
                f"(declared: {sorted(allowed)})"
            )


def test_list_metrics_selected_blocks(binary_handler_analyze_rocprof_compute, capsys):
    """Aliases select blocks, each table is a grid, and per-channel tables
    collapse to one line."""
    return_code = binary_handler_analyze_rocprof_compute([
        "--list-metrics-gfx-id",
        "gfx950",
        "sol",
        "l2_per_channel",
    ])
    assert return_code == 0

    output = capsys.readouterr().out
    assert (
        "\n2. System Speed-of-Light (alias: sol)\n2.1 System Speed-of-Light\n" in output
    )
    assert re.search(
        r"│ 2\.1\.0 +│ VALU FLOPs +│ GFLOP/s +│ The total floating", output
    )
    assert "\nPer-channel metrics are not listed: 18.2 to 18.12\n" in output
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
    assert "│ Block_ID │ Alias" in output


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
