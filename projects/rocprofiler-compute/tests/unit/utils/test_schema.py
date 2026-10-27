# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for Workload's ML tracing state and shared model annotations."""

from pathlib import Path
from typing import List, get_type_hints

import pandas as pd

from ml_api_tracing.analysis_models import MlApiTracePair
from ml_api_tracing.models import CallTreeForest, CallTreeNode
from utils.schema import Workload


def test_workload_ml_state_defaults_are_independent():
    first = Workload()
    second = Workload()
    first.ml_api_trace_pairs.append(
        MlApiTracePair(pd.DataFrame(), Path("counters.csv"))
    )
    first.unmatched_kernel_frames.append(pd.DataFrame({"Kernel_Name": ["gemm"]}))
    first.ml_api_trace_df.loc[0, "Operator_Name"] = "aten::mm"
    first.ml_api_call_trees["1"] = [CallTreeNode(name="aten::mm")]
    assert second.ml_api_trace_pairs == []
    assert second.unmatched_kernel_frames == []
    assert second.ml_api_trace_df.empty
    assert second.ml_api_call_trees == {}


def test_workload_annotations_resolve_canonical_records():
    workload_hints = get_type_hints(Workload)
    assert workload_hints["ml_api_trace_pairs"] == List[MlApiTracePair]
    assert workload_hints["ml_api_call_trees"] == CallTreeForest
