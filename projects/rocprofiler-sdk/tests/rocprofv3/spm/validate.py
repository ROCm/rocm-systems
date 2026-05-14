#!/usr/bin/env python3

import sys
import pytest
import pandas as pd
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
        sq_waves_values = []
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
            assert pmc_value == spm_value
        elif not is_cycle(counter_name):
            assert within_tolerance(pmc_value, spm_value)


def test_validate_spm_rocpd_csv(counter_csv: pd.DataFrame, spm_json_data):
    assert not counter_csv.empty

    TOLERANCE = 0.2
    within_tolerance = lambda x, y: abs(x - y) < TOLERANCE * max(x, y)

    kernel_column = "kernel_name" if "kernel_name" in counter_csv else "Kernel_Name"
    counter_column = "counter_name" if "counter_name" in counter_csv else "Counter_Name"
    value_column = "Counter_Value"

    filtered = counter_csv[counter_csv[kernel_column].str.contains("matrixTranspose")]

    csv_values = (
        filtered.groupby(counter_column)[value_column].sum().to_dict()
        if not filtered.empty
        else {}
    )

    assert csv_values

    def _collect_spm_totals(json_data, kernel_filter):
        data = json_data["rocprofiler-sdk-tool"]
        counters = {itr["id"]["handle"]: itr for itr in data.get("counters", [])}
        kernel_symbols = data.get("kernel_symbols", [])

        values = {}
        for entry in data["callback_records"]["spm_counter_collection"]:
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

    spm_values = _collect_spm_totals(spm_json_data, "matrixTranspose")

    assert spm_values

    is_cycle = lambda x: x[:2] == "CP" or x == "SQ_CYCLES"
    is_deterministic = lambda x: x[:3] == "SQ_" and x != "SQ_CYCLES"

    for counter_name, csv_value in csv_values.items():
        if counter_name not in spm_values:
            continue
        spm_value = spm_values[counter_name]
        if is_deterministic(counter_name):
            assert csv_value == spm_value
        elif not is_cycle(counter_name):
            assert within_tolerance(csv_value, spm_value)


def _find_table_or_view(conn, base_name):
    for typ in ("view", "table"):
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name LIKE ?",
            (typ, f"{base_name}%"),
        ).fetchone()
        if row:
            return row[0]
    return None


def test_validate_spm_rocpd(spm_json_data, rocpd_data):
    data = spm_json_data["rocprofiler-sdk-tool"]
    spm_data = data["callback_records"]["spm_counter_collection"]

    pmc_table = _find_table_or_view(rocpd_data, "rocpd_info_pmc")
    pmc_event_table = _find_table_or_view(rocpd_data, "rocpd_pmc_event")

    assert pmc_table is not None
    assert pmc_event_table is not None

    counters = {itr["id"]["handle"]: itr["name"] for itr in data.get("counters", [])}

    spm_counter_names = set()
    for entry in spm_data:
        for record in entry["records"]:
            spm_counter_names.add(counters[record["counter_id"]["handle"]])

    assert len(spm_counter_names) > 0

    placeholders = ",".join(["?"] * len(spm_counter_names))
    pmc_name_list = sorted(spm_counter_names)

    rocpd_pmc_names = rocpd_data.execute(
        f"SELECT name FROM {pmc_table} WHERE name IN ({placeholders})",
        pmc_name_list,
    ).fetchall()

    assert len(rocpd_pmc_names) > 0

    rocpd_spm_count = rocpd_data.execute(
        f"SELECT COUNT(*) FROM {pmc_event_table} e "
        f"JOIN {pmc_table} p ON e.pmc_id = p.id "
        f"WHERE p.name IN ({placeholders})",
        pmc_name_list,
    ).fetchone()[0]

    assert rocpd_spm_count > 0


def test_validate_spm_external_correlation_rocpd(rocpd_data):
    spm_view = _find_table_or_view(rocpd_data, "spm_counters")
    assert spm_view is not None

    rows = rocpd_data.execute(
        f"SELECT DISTINCT correlation_id FROM {spm_view}"
    ).fetchall()

    assert len(rows) > 0, "No SPM records in rocpd"
    for row in rows:
        assert row[0] is not None, "correlation_id should not be NULL"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
