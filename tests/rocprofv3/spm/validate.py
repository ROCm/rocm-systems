#!/usr/bin/env python3

import sys
import pytest
import json
import numpy as np
import pandas as pd
import re


def test_validate_spm_json(json_data):

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

    skip_gfx = ("gfx1101", "gfx1102")
    data = json_data["rocprofiler-sdk-tool"]
    spm_data = data["callback_records"]["streaming_performance_monitor"]
    for spm_record in spm_data:
        assert spm_record["dispatch_id"] > 0
        for record in spm_record["records"]:
            sq_waves_values = []
            agent = get_agent(record["agent_id"])
            counter = get_counter(record["counter_id"])
            assert counter is not None, f"record:\n\t{record}"
            if counter["name"] == "SQ_WAVES" and agent["name"] not in skip_gfx:
                sq_waves_values.append(record["value"])
        if len(sq_waves_values) > 0:
            assert sum(sq_waves_values) > 0, "SQ_WAVES value is not > 0"


def test_validate_spm(pmc_csv: pd.DataFrame, spm_csv: pd.DataFrame, gfx_data):

    assert not pmc_csv.empty and not spm_csv.empty
    SE_PER_XCC = gfx_data
    TOLERANCE = 0.2
    within_tolerance = lambda x, y: abs(x - y) < TOLERANCE * max(x, y)

    # Filter both CSVs with matrixTranspose kernel and dispatch ID
    pmc_csv = pmc_csv[pmc_csv["Kernel_Name"].str.contains("matrixTranspose")]
    spm_csv = spm_csv[spm_csv["Dispatch_Id"] == pmc_csv["Dispatch_Id"].values[0]]

    to_dict = lambda x: {n: v for n, v in zip(x["Counter_Name"], x["Counter_Value"])}

    pmc_value = to_dict(pmc_csv)
    spm_value = to_dict(spm_csv)

    is_cycle = lambda x: x[:2] == "CP" or x == "SQ_CYCLES"
    is_deterministic = lambda x: x[:3] == "SQ_" and x != "SQ_CYCLES"

    # Deterministic and nearly deterministic counters
    for counter_name in pmc_value:
        if is_deterministic(counter_name):
            assert pmc_value[counter_name] == spm_value[counter_name]
        elif not is_cycle(counter_name):
            assert within_tolerance(pmc_value[counter_name], spm_value[counter_name])

    # Approximate GRBM_COUNT
    elapsed_xcc_cycle = spm_value["SQ_CYCLES"] / SE_PER_XCC
    # Short dispatch SPM leaves CPC mostly busy. Note: In device SPM, this is reversed
    assert within_tolerance(spm_value["CPC_CPC_STAT_BUSY"], elapsed_xcc_cycle)
    assert spm_value["CPC_CPC_STAT_IDLE"] < TOLERANCE * elapsed_xcc_cycle


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
