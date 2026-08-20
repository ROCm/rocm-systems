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


import itertools
import sys
import pytest
import numpy as np
import pandas as pd


# ===================== validation common for both host-trap and stochastic sampling
def test_multi_agent_support(
    input_samples_csv: pd.DataFrame,
    input_kernel_trace_csv: pd.DataFrame,
    input_agent_info_csv: pd.DataFrame,
):
    from rocprofiler_sdk.pc_sampling.transpose_multiple_agents.csv import (
        validate_all_agents_are_sampled,
    )

    validate_all_agents_are_sampled(
        input_samples_csv, input_kernel_trace_csv, input_agent_info_csv
    )


# =================== validation specific to stochastic sampling


def test_validate_pc_sampling_stochastic_specific_csv(
    input_samples_csv: pd.DataFrame, input_agent_info_csv: pd.DataFrame
):
    if is_gfx12(input_agent_info_csv):
        from rocprofiler_sdk.pc_sampling.stochastic.csv.gfx12 import (
            validate_stochastic_samples_csv,
        )
    elif is_gfx9(input_agent_info_csv):
        from rocprofiler_sdk.pc_sampling.stochastic.csv.gfx9 import (
            validate_stochastic_samples_csv,
        )
    else:
        pytest.skip(
            "Stochastic sampling specific CSV checks are not implemented for this architecture"
        )

    validate_stochastic_samples_csv(input_samples_csv)


def test_validate_pc_sampling_stochastic_specific_json(
    input_samples_json, input_agent_info_csv: pd.DataFrame
):
    if is_gfx12(input_agent_info_csv):
        from rocprofiler_sdk.pc_sampling.stochastic.json.gfx12 import (
            validate_stochastic_samples_json,
        )
    elif is_gfx9(input_agent_info_csv):
        from rocprofiler_sdk.pc_sampling.stochastic.json.gfx9 import (
            validate_stochastic_samples_json,
        )
    else:
        pytest.skip(
            "Stochastic sampling specific JSON checks are not implemented for this architecture"
        )

    validate_stochastic_samples_json(input_samples_json["rocprofiler-sdk-tool"])


def is_gfx12(input_agent_info_csv: pd.DataFrame) -> bool:
    return input_agent_info_csv["Name"].str.contains("gfx12").any()


def is_gfx9(input_agent_info_csv: pd.DataFrame) -> bool:
    return input_agent_info_csv["Name"].str.contains(r"gfx9\d", regex=True).any()


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
