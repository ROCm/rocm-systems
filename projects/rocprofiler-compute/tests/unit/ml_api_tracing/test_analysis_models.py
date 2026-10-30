# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML API analysis records."""

from pathlib import Path
from typing import Optional, get_type_hints

import pandas as pd

from ml_api_tracing.analysis_models import MlApiTracePair


def test_trace_pair_keeps_input_references_and_mutable_joined_state():
    marker_df = pd.DataFrame({"Operator_Name": ["aten::mm"]})
    counter_path = Path("counters.csv")
    pair = MlApiTracePair(marker_df, counter_path)
    assert pair.marker_df is marker_df
    assert pair.counter_path is counter_path
    assert pair.joined_df is None
    joined_df = pd.DataFrame({"Kernel_Name": ["gemm"]})
    pair.joined_df = joined_df
    assert pair.joined_df is joined_df


def test_analysis_model_annotations_resolve_canonical_records():
    assert get_type_hints(MlApiTracePair) == {
        "marker_df": pd.DataFrame,
        "counter_path": Path,
        "joined_df": Optional[pd.DataFrame],
    }
