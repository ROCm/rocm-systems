# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Check the historical gfx1100 SDK override and selector references."""

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

from vendored import yaml

PROJECT = Path(__file__).resolve().parents[3]
CONTRACT = Path(__file__).with_name("data") / "gfx1100_sdk_contract.json"


def load_contract() -> Dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    contract = load_contract()
    if "historical_name" in metafunc.fixturenames:
        cases = sorted(contract["definitions"].items())
        metafunc.parametrize(
            "historical_name,historical_definition",
            cases,
            ids=[name for name, _ in cases],
        )
    if "selector_name" in metafunc.fixturenames:
        cases = sorted(contract["soc21_raw_selectors"].items())
        metafunc.parametrize(
            "selector_name,selector_reference", cases, ids=[name for name, _ in cases]
        )
    if "derived_name" in metafunc.fixturenames:
        cases = sorted(
            name
            for name, definition in contract["definitions"].items()
            if "expression" in definition
        )
        metafunc.parametrize("derived_name", cases, ids=cases)


@pytest.fixture(scope="module")
def sdk_config() -> Dict[str, Any]:
    path = PROJECT / "src/rocprof_compute_soc/profile_configs/sdk_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def definitions_for_arch(
    config: Dict[str, Any], architecture: str
) -> Dict[str, Dict[str, Any]]:
    result = {}
    for counter in config["rocprofiler-sdk"]["counters"]:
        for definition in counter["definitions"]:
            if architecture in definition["architectures"]:
                assert counter["name"] not in result, counter["name"]
                result[counter["name"]] = {
                    key: value
                    for key, value in definition.items()
                    if key != "architectures"
                }
    return result


def normalize_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: "".join(value.split()) if key == "expression" else value
        for key, value in definition.items()
    }


@pytest.mark.misc
def test_historical_gfx1100_definition_matches_sdk(
    sdk_config: Dict[str, Any],
    historical_name: str,
    historical_definition: Dict[str, Any],
) -> None:
    actual = definitions_for_arch(sdk_config, "gfx1100")
    assert historical_name in actual
    assert normalize_definition(actual[historical_name]) == normalize_definition(
        historical_definition
    )


@pytest.mark.misc
def test_raw_gfx1100_selector_matches_soc21_reference(
    sdk_config: Dict[str, Any],
    selector_name: str,
    selector_reference: Dict[str, Any],
) -> None:
    actual = definitions_for_arch(sdk_config, "gfx1100")
    assert actual[selector_name]["event"] == selector_reference["event"]
    assert "PERF_SEL" in selector_reference["symbol"]


@pytest.mark.misc
def test_historical_gfx1100_contract_is_complete(sdk_config: Dict[str, Any]) -> None:
    contract = load_contract()
    assert contract["architecture"] == "gfx1100"
    assert len(contract["definitions"]) == 84
    assert len(contract["soc21_raw_selectors"]) == 41
    actual = definitions_for_arch(sdk_config, "gfx1100")
    assert contract["definitions"].keys() <= actual.keys()


@pytest.mark.misc
def test_gfx1100_derived_dependencies_are_available(
    sdk_config: Dict[str, Any], derived_name: str
) -> None:
    actual = definitions_for_arch(sdk_config, "gfx1100")
    known_names = {
        counter["name"] for counter in sdk_config["rocprofiler-sdk"]["counters"]
    }
    referenced = set(
        re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", actual[derived_name]["expression"])
    )
    assert (referenced & known_names) <= actual.keys()


@pytest.mark.misc
@pytest.mark.parametrize(
    "architecture,name,event",
    [
        ("gfx1151", "GL2C_EA_RDREQ_32B", 99),
        ("gfx1151", "GL2C_EA_RDREQ_128B", 102),
        ("gfx1250", "GL2C_EA_RDREQ_32B", 119),
        ("gfx1250", "GL2C_EA_RDREQ_128B", 121),
    ],
)
def test_existing_architecture_selectors_are_preserved(
    sdk_config: Dict[str, Any], architecture: str, name: str, event: int
) -> None:
    actual = definitions_for_arch(sdk_config, architecture)
    assert actual[name]["event"] == event
