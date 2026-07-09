#!/usr/bin/env python3

import sys
import pytest
import re


# JSON size will become large with several counters.
def test_validate_spm_json(spm_json_data):

    def get_agent(agent_id):
        for agent in data["agents"]:
            if agent["id"]["handle"] == agent_id["handle"]:
                return agent
        return None

    def get_counter(counter_id):
        for counter in data["counters"]:
            if counter["id"]["handle"] == counter_id["handle"]:
                return counter
        return None

    pattern = re.compile("^gfx9[0-9]+$")
    data = spm_json_data["rocprofiler-sdk-tool"]
    spm_data = data["callback_records"]["spm_counter_collection"]
    kernel_symbols = data.get("kernel_symbols", [])

    sq_waves_values = []
    for spm_record in spm_data:

        dispatch_data = spm_record["dispatch_data"]
        dispatch_info = dispatch_data["dispatch_info"]

        assert dispatch_info["agent_id"]["handle"] > 0
        assert dispatch_info["queue_id"]["handle"] > 0
        assert dispatch_info["dispatch_id"] > 0

        kernel_id = dispatch_info.get("kernel_id")
        if isinstance(kernel_id, dict):
            kernel_id = kernel_id.get("handle")
        kernel_name = kernel_symbols[kernel_id]["formatted_kernel_name"]
        if "matrixTranspose" not in kernel_name:
            continue
        for record in spm_record["records"]:
            agent = get_agent(dispatch_info["agent_id"])
            counter = get_counter(record["counter_id"])
            assert counter is not None, f"record:\n\t{record}"
            if (
                counter["name"] == "SQ_WAVES"
                and re.match(pattern, agent["name"]) is not None
            ):
                sq_waves_values.append(record["value"])
    if len(sq_waves_values) > 0:
        assert sum(sq_waves_values) > 0, "SQ_WAVES value is not > 0"


def test_validate_spm(pmc_json_data, spm_json_data):

    TOLERANCE = 0.2
    within_tolerance = lambda x, y: abs(x - y) < TOLERANCE * max(x, y)

    def _collect_counter_totals(json_data, record_kind, kernel_filter):
        data = json_data["rocprofiler-sdk-tool"]

        counters = {itr["id"]["handle"]: itr for itr in data.get("counters", [])}
        kernel_symbols = data.get("kernel_symbols", [])

        values = {}
        for entry in data["callback_records"][record_kind]:
            dispatch_info = entry["dispatch_data"]["dispatch_info"]
            kernel_id = dispatch_info.get("kernel_id")
            if isinstance(kernel_id, dict):
                kernel_id = kernel_id.get("handle")
            kernel_name = kernel_symbols[kernel_id]["formatted_kernel_name"]
            if kernel_filter not in kernel_name:
                continue

            for record in entry["records"]:
                counter_id = record["counter_id"]["handle"]
                counter = counters[counter_id]
                counter_name = counter["name"]
                values[counter_name] = values.get(counter_name, 0) + record["value"]

        return values

    pmc_values = _collect_counter_totals(
        pmc_json_data, "counter_collection", "matrixTranspose"
    )
    spm_values = _collect_counter_totals(
        spm_json_data, "spm_counter_collection", "matrixTranspose"
    )

    assert pmc_values and spm_values

    is_cycle = lambda x: x[:2] == "CP" or x == "SQ_CYCLES"
    is_deterministic = lambda x: x[:3] == "SQ_" and x != "SQ_CYCLES"
    # Deterministic and nearly deterministic counters
    for counter_name, pmc_value in pmc_values.items():
        if counter_name not in spm_values:
            continue
        spm_value = spm_values[counter_name]
        if is_deterministic(counter_name):
            assert (
                pmc_value == spm_value
            ), f"{counter_name}: pmc={pmc_value} != spm={spm_value}"
        elif not is_cycle(counter_name):
            assert within_tolerance(
                pmc_value, spm_value
            ), f"{counter_name}: pmc={pmc_value}, spm={spm_value}, not within {TOLERANCE*100}% tolerance"


def test_validate_spm_multigpu_stream_id(spm_json_data):
    data = spm_json_data["rocprofiler-sdk-tool"]
    spm_data = data["callback_records"]["spm_counter_collection"]
    kernel_symbols = data.get("kernel_symbols", [])

    found_subtract = False
    for spm_record in spm_data:
        assert "stream_id" in spm_record, "stream_id missing from SPM record"
        assert "handle" in spm_record["stream_id"]

        dispatch_info = spm_record["dispatch_data"]["dispatch_info"]
        kernel_id = dispatch_info.get("kernel_id")
        if isinstance(kernel_id, dict):
            kernel_id = kernel_id.get("handle")
        kernel_name = kernel_symbols[kernel_id]["formatted_kernel_name"]
        if "subtract_kernel" in kernel_name:
            found_subtract = True
            assert (
                spm_record["stream_id"]["handle"] > 0
            ), f"stream_id should be non-zero for {kernel_name}"

    assert found_subtract, "No subtract_kernel dispatches found in SPM data"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
