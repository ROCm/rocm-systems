#!/usr/bin/env python3

import json
import os
import pytest
import pandas as pd

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):
    parser.addoption(
        "--input-samples-csv",
        action="store",
        help="Path to CSV file containing PC samples.",
    )

    parser.addoption(
        "--input-kernel-trace-csv",
        action="store",
        help="Path to CSV file containing kernel trace.",
    )

    parser.addoption(
        "--input-agent-info-csv",
        action="store",
        help="Path to CSV file containing agents information.",
    )

    parser.addoption(
        "--input-samples-json",
        action="store",
        help="Path to JSON file containing PC samples.",
    )


@pytest.fixture
def input_samples_csv(request):
    filename = request.config.getoption("--input-samples-csv")
    if not os.path.isfile(filename):
        # The CSV file is not generated, because the dependency test
        # responsible to generate this file was skipped or failed.
        # Thus emit the message to skip this test as well.
        print("PC sampling unavailable")
    else:
        with open(filename, "r") as inp:
            return pd.read_csv(
                inp,
                na_filter=False,  # parse empty fields as ""
                keep_default_na=False,  # parse empty fields as ""
                dtype={
                    "Exec_Mask": "uint64",
                    "Instruction": str,
                    "Instruction_Comment": str,
                    "Wave_Issued_Instruction": bool,
                    "Instruction_Type": str,
                    "Stall_Reason": str,
                },
            )


@pytest.fixture
def input_kernel_trace_csv(request):
    filename = request.config.getoption("--input-kernel-trace-csv")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_agent_info_csv(request):
    filename = request.config.getoption("--input-agent-info-csv")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_samples_json(request):
    filename = request.config.getoption("--input-samples-json")
    with open(filename, "r") as inp:
        # Significant overhead of 5-6secs observed when feeding
        # data into the dotdict.
        # Using plain python dict instead
        return collapse_dict_list(json.load(inp))
