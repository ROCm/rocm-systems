#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

from __future__ import absolute_import

import pandas as pd

def validate_all_agents_are_sampled(
    input_samples_csv: pd.DataFrame,
    input_kernel_trace_csv: pd.DataFrame,
    input_agent_info_csv: pd.DataFrame,
):
    all_gpu_agents_df = input_agent_info_csv[
        input_agent_info_csv["Agent_Type"] == "GPU"
    ].copy()
    all_gpu_agents_df["Type_Relative_Id"] = range(len(all_gpu_agents_df))
    gfx9_gfx12_agents_df = all_gpu_agents_df[
        all_gpu_agents_df["Name"].apply(
            lambda name: name == "gfx90a"
            or name.startswith("gfx94")
            or name.startswith("gfx95")
            or name.startswith("gfx12")
        )
    ]

    # Extract samples that originates from know code object it
    samples_df = input_samples_csv[input_samples_csv["Dispatch_Id"] != 0].copy()

    dispatch_agents = input_kernel_trace_csv.drop_duplicates("Dispatch_Id").set_index(
        "Dispatch_Id"
    )["Agent_Id"]
    agent_ids = (
        samples_df["Dispatch_Id"]
        .map(dispatch_agents)
        .str.extract(r"(\d+)$", expand=False)
    )
    assert agent_ids.notna().all(), (
        "at least one sampled dispatch is missing from the kernel trace"
    )
    samples_df["Agent_Id"] = agent_ids.astype("uint64")

    sampled_agents = set(samples_df["Agent_Id"].unique())
    expected_agent_sets = [
        set(gfx9_gfx12_agents_df[column].astype("uint64"))
        for column in ("Node_Id", "Logical_Node_Id", "Type_Relative_Id")
    ]
    assert sampled_agents in expected_agent_sets, (
        f"sampled agent IDs {sorted(sampled_agents)} do not match any "
        f"expected set {[sorted(ids) for ids in expected_agent_sets]}"
    )

    # Extract the embedded source line without resolving or reading the path.
    transpose_lines = samples_df["Instruction_Comment"].str.extract(
        r"(?:^|[/\\])transpose\.cpp:(\d+)", expand=False
    )
    transpose_samples_df = samples_df[transpose_lines.notna()].copy()
    transpose_samples_df["Source_Line_Num"] = transpose_lines.dropna().astype(int)
    assert set(transpose_samples_df["Agent_Id"].unique()) == sampled_agents, (
        "not every sampled agent has a decoded transpose.cpp instruction"
    )
    assert transpose_samples_df["Source_Line_Num"].between(192, 202).all(), (
        "transpose.cpp samples map outside the transpose kernel"
    )
