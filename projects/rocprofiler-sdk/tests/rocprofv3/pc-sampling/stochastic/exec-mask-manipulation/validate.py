#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys
import pytest
import numpy as np
import pandas as pd


# =========================== Validating fields common for both host-trap and stochastic CSV output


def test_validate_pc_sampling_exec_mask_manipulation_csv(
    input_csv: pd.DataFrame, all_sampled: bool
):
    from rocprofiler_sdk.pc_sampling.exec_mask_manipulation.csv import (
        exec_mask_manipulation_validate_csv,
    )

    exec_mask_manipulation_validate_csv(input_csv, all_sampled=all_sampled)


# # ========================= Validating fields common for both host-trap and stochastic JSON output


def test_validate_pc_sampling_exec_mask_manipulation_json(
    input_json, input_csv: pd.DataFrame, all_sampled: bool
):
    data = input_json["rocprofiler-sdk-tool"]
    # The same amount of samples should be in both CSV and JSON files.
    assert len(input_csv) == len(data["buffer_records"]["pc_sample_stochastic"])
    # # validating JSON output
    from rocprofiler_sdk.pc_sampling.exec_mask_manipulation.json import (
        validate_json_exec_mask_manipulation,
    )

    validate_json_exec_mask_manipulation(
        data, pc_sampling_method="stochastic", all_sampled=all_sampled
    )


# ======================== Validating fields specific for stochastic sampling


def test_validate_pc_sampling_stochastic_specific_csv(input_csv: pd.DataFrame):
    from rocprofiler_sdk.pc_sampling.stochastic.csv.gfx9 import (
        validate_stochastic_samples_csv,
    )

    validate_stochastic_samples_csv(input_csv)


def test_validate_pc_sampling_stochastic_specific_json(input_json):
    from rocprofiler_sdk.pc_sampling.stochastic.json.gfx9 import (
        validate_stochastic_samples_json,
    )

    validate_stochastic_samples_json(input_json["rocprofiler-sdk-tool"])


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
