#!/usr/bin/env python3

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


def test_validate_pc_sampling_stochastic_specific_csv(input_samples_csv: pd.DataFrame):
    from rocprofiler_sdk.pc_sampling.stochastic.csv.gfx9 import (
        validate_stochastic_samples_csv,
    )

    validate_stochastic_samples_csv(input_samples_csv)


def test_validate_pc_sampling_stochastic_specific_json(input_samples_json):
    from rocprofiler_sdk.pc_sampling.stochastic.json.gfx9 import (
        validate_stochastic_samples_json,
    )

    validate_stochastic_samples_json(input_samples_json["rocprofiler-sdk-tool"])


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
