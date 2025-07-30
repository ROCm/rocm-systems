#!/usr/bin/env python3

import sys
import pytest
import numpy as np
import pandas as pd
import re


def test_validate_spm(pmc_csv: pd.DataFrame, spm_csv: pd.DataFrame):

    assert not pmc_csv.empty and not spm_csv.empty
    SE_PER_XCC = 4
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
