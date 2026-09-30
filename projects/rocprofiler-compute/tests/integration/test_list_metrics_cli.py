# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for the modeless --list-metrics and --list-blocks options."""

from pathlib import Path

import pytest
import yaml
from common import SRC

import utils.utils_common as utils_common
from utils.mi_gpu_spec import mi_gpu_specs

ANALYSIS_CONFIGS = Path(SRC) / "rocprof_compute_soc" / "analysis_configs"


def listed_metric_ids(output: str) -> set[str]:
    """Return the ids of the "<id> -> <name>" rows in --list-metrics output."""
    return {
        line.strip().split(" -> ")[0] for line in output.splitlines() if " -> " in line
    }


# =============================================================================
# TESTS FOR MODELESS COMMAND LINE OPTIONS
# =============================================================================


def test_list_metrics(binary_handler_analyze_rocprof_compute, capsys):
    return_code = binary_handler_analyze_rocprof_compute(["--list-metrics", "gfx90a"])
    assert return_code == 0

    # Test output
    output = capsys.readouterr().out
    assert "6 -> Workgroup Manager (SPI)" in output
    assert "5.2 -> Command processor packet processor (CPC)" in output


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


def test_list_metrics_row_format(binary_handler_analyze_rocprof_compute, capsys):
    """Blocks show an alias, metrics a unit and description, and no channel rows."""
    return_code = binary_handler_analyze_rocprof_compute(["--list-metrics", "gfx950"])
    assert return_code == 0

    output = capsys.readouterr().out
    assert "\n2 -> System Speed-of-Light (alias: sol)\n" in output
    assert "\n\t2.1 -> System Speed-of-Light\n" in output
    assert "\n\t\t2.1.0 -> VALU FLOPs [GFLOP/s]: The total floating-point" in output
    assert "\t\t12.2.0 -> LDS Instructions [Instructions per normUnit]: " in output
    assert "\n\t18.2 -> L2 Cache Hit Rate (Percent)\n" in output
    assert "18.2.0 ->" not in output


def test_list_metrics_defaults_to_current_gpu(
    binary_handler_analyze_rocprof_compute, capsys
):
    return_code = binary_handler_analyze_rocprof_compute(["--list-metrics"])
    assert return_code == 0

    output = capsys.readouterr().out
    assert "0 -> Top Stats" in output
    assert "2 -> System Speed-of-Light" in output


def test_list_metrics_with_block_filter(binary_handler_analyze_rocprof_compute, capsys):
    return_code = binary_handler_analyze_rocprof_compute([
        "--list-metrics",
        "gfx950",
        "-b",
        "lds",
        "2.1.3",
    ])
    assert return_code == 0

    listed_ids = listed_metric_ids(capsys.readouterr().out)
    assert {"2", "2.1", "2.1.3", "12", "12.1.0"} <= listed_ids
    assert not {"2.1.0", "3"} & listed_ids


def test_block_without_list_metrics_is_rejected(
    binary_handler_analyze_rocprof_compute,
):
    return_code = binary_handler_analyze_rocprof_compute([
        "--list-blocks",
        "gfx950",
        "-b",
        "2",
    ])
    assert return_code == 1
