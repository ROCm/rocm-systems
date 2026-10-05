#!/usr/bin/env python3

import sys
import pytest
import pandas as pd


def _find_table_or_view(conn, base_name):
    for typ in ("view", "table"):
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name LIKE ?",
            (typ, f"{base_name}%"),
        ).fetchone()
        if row:
            return row[0]
    return None


def test_validate_spm_rocpd_csv(counter_csv: pd.DataFrame, spm_json_data):
    assert not counter_csv.empty

    kernel_column = "kernel_name" if "kernel_name" in counter_csv else "Kernel_Name"
    counter_column = "counter_name" if "counter_name" in counter_csv else "Counter_Name"
    value_column = "Counter_Value"

    filtered = counter_csv[counter_csv[kernel_column].str.contains("matrixTranspose")]

    assert not filtered.empty, "No matrixTranspose entries in counter CSV"

    filtered = filtered.copy()
    filtered["base_counter"] = filtered[counter_column].str.replace(
        r"\[.*\]$", "", regex=True
    )

    csv_values = filtered.groupby("base_counter")[value_column].sum().to_dict()

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

    for counter_name, csv_value in csv_values.items():
        assert counter_name in spm_values, (
            f"{counter_name} in CSV but not in JSON. "
            f"JSON counters: {sorted(spm_values.keys())}"
        )
        spm_value = spm_values[counter_name]
        assert (
            csv_value == spm_value
        ), f"{counter_name}: csv={csv_value} != json={spm_value}"


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


def test_validate_spm_sample_timestamps(rocpd_data):
    sample_table = _find_table_or_view(rocpd_data, "rocpd_sample")
    assert sample_table is not None

    rows = rocpd_data.execute(f"SELECT id, timestamp FROM {sample_table}").fetchall()

    assert len(rows) > 0, "No samples found in rocpd"
    for row in rows:
        assert row[1] > 0, f"sample id={row[0]} has timestamp={row[1]}, expected > 0"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
