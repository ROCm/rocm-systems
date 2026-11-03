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

import numpy as np
import pandas as pd


def validate_all_agents_are_sampled(
    input_samples_csv: pd.DataFrame,
    input_kernel_trace_csv: pd.DataFrame,
    input_agent_info_csv: pd.DataFrame,
):
    transpose_kernel_source_line_start = 193
    transpose_kernel_source_line_end = 202

    gfx9_gfx12_agents_df = input_agent_info_csv[
        input_agent_info_csv["Name"].apply(
            lambda name: name == "gfx90a"
            or name.startswith("gfx94")
            or name.startswith("gfx95")
            or name.startswith("gfx12")
        )
    ]

    # Extract samples that originates from know code object it
    samples_df = input_samples_csv[input_samples_csv["Dispatch_Id"] != 0].copy()

    # Determine the agent on which sample was generated
    # Note: Agent_Id is in the following format e.g., "Agent 3",
    # that's why we need a log for extracting integer value of the id.
    # Determine the agent on which sample was generated
    dispatch_agents = (
        input_kernel_trace_csv.set_index("Dispatch_Id")["Agent_Id"]
        .str.extract(r"^Agent ([0-9]+)$", expand=False)
        .astype(np.uint64)
    )
    samples_df["Agent_Id"] = (
        samples_df["Dispatch_Id"].map(dispatch_agents).astype(np.uint64)
    )
    sampled_agents = samples_df["Agent_Id"].unique()
    expected_agents = set(gfx9_gfx12_agents_df["Logical_Node_Id"]) & set(dispatch_agents)
    assert expected_agents and expected_agents.issubset(sampled_agents)

    # separate samples per agents
    grouped_samples_per_agent = samples_df.groupby("Agent_Id")
    for agent_id, agent_samples_df in grouped_samples_per_agent:
        sampled_dispatches = agent_samples_df["Dispatch_Id"].unique()
        # at least 1 sampled dispatch per agent
        assert len(sampled_dispatches) >= 1

    # extract decoded samples that are mapped to the transpose.cpp file
    transpose_samples_df = samples_df[
        samples_df["Instruction_Comment"].apply(
            lambda comment: "transpose.cpp" in comment
        )
    ].copy()
    assert not transpose_samples_df.empty, "no PC samples mapped to transpose.cpp"
    # determine the line number for each sample
    transpose_samples_df["Source_Line_Num"] = transpose_samples_df[
        "Instruction_Comment"
    ].apply(lambda source_line: int(source_line.split(":")[-1]))
    # assert that line belongs to a kernel range
    assert (
        (transpose_samples_df["Source_Line_Num"] >= transpose_kernel_source_line_start)
        & (transpose_samples_df["Source_Line_Num"] <= transpose_kernel_source_line_end)
    ).all()
