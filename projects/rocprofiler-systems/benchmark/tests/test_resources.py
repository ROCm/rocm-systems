# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every shipped scenario, example and suite file must stay valid after a move or an edit."""

import json
from pathlib import Path

import pytest
from bench import scenario, snapshot

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = sorted((ROOT / "resource" / "scenarios").rglob("*.json"))
EXAMPLES = sorted((ROOT / "docs" / "examples").glob("*.json"))
SUITES = sorted((ROOT / "resource" / "snapshots").glob("*.json"))


def test_resource_files_are_found():
    assert SCENARIOS and EXAMPLES and SUITES


@pytest.mark.parametrize(
    "path", SCENARIOS + EXAMPLES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_scenario_is_valid(path):
    scenario.load(str(path))


@pytest.mark.parametrize(
    "path", SCENARIOS + EXAMPLES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_scenario_output_stays_inside_resource_results(path):
    output = json.loads(path.read_text()).get("run", {}).get("output")
    if output:
        resolved = (path.parent / output).resolve()
        assert (ROOT / "resource" / "results") in resolved.parents, resolved


@pytest.mark.parametrize(
    "path", SCENARIOS + EXAMPLES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_workload_points_into_build_dir(path):
    workload = Path(
        scenario.resolve_path(json.loads(path.read_text())["workload"], str(path.parent))
    )
    assert workload.parent == ROOT / "build", workload


@pytest.mark.parametrize("path", SUITES, ids=lambda p: p.name)
def test_suite_lists_existing_scenarios(path):
    for entry in snapshot.load_suite(str(path)):
        assert Path(entry).is_file(), entry
