# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import pytest

# =============================================================================
# TESTS FOR MODELESS COMMAND LINE OPTIONS
# =============================================================================


@pytest.mark.list_metrics
def test_list_metrics(binary_handler_analyze_rocprof_compute, capsys):
    return_code = binary_handler_analyze_rocprof_compute(["--list-metrics", "gfx90a"])
    assert return_code == 0

    # Test output
    output = capsys.readouterr().out
    assert "6 -> Workgroup Manager (SPI)" in output
    assert "5.2 -> Command processor packet processor (CPC)" in output


def test_list_blocks(binary_handler_analyze_rocprof_compute, capsys):
    return_code = binary_handler_analyze_rocprof_compute(["--list-blocks", "gfx90a"])
    assert return_code == 0

    # Test output
    output = capsys.readouterr().out
    assert "INDEX" in output
    assert "BLOCK ALIAS" in output
    assert "BLOCK NAME" in output

    # Verify specific block id, alias, and name mappings
    lines = output.strip().splitlines()
    block_entries = {}
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) >= 3:
            block_id = parts[0]
            block_alias = parts[1]
            block_name = " ".join(parts[2:])
            block_entries[block_id] = (block_alias, block_name)

    assert block_entries["0"] == ("topstats", "Top Stats")
    assert block_entries["1"] == ("sysinfo", "System Info")
    assert block_entries["6"] == ("spi", "Workgroup Manager (SPI)")

