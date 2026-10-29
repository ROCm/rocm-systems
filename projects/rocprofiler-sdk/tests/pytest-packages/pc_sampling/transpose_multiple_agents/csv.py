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
    transpose_kernel_source_line_start = 181
    transpose_kernel_source_line_end = 189

    # PC sampling is supported on these architectures. agent_info lists every
    # KFD node, including GPUs hidden by HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES,
    # so this count is not the set of agents the workload can dispatch to.
    gfx9_gfx12_agents_df = input_agent_info_csv[
        input_agent_info_csv["Name"].apply(
            lambda name: name == "gfx90a"
            or name.startswith("gfx94")
            or name.startswith("gfx95")
            or name.startswith("gfx12")
        )
    ]
    assert len(gfx9_gfx12_agents_df) >= 1

    # Agent_Id is "Agent 3" or "GPU 3" depending on --agent-index.
    def agent_ordinals(agent_ids: pd.Series) -> set:
        return set(agent_ids.astype(str).str.split(" ").str[1].astype(np.uint64))

    # Agents the process actually launched kernels on. This is the visible set:
    # a device mask leaves the other GPUs in agent_info with no dispatches.
    kernel_agents = agent_ordinals(input_kernel_trace_csv["Agent_Id"])

    # Extract samples that originate from a known dispatch.
    samples_df = input_samples_csv[input_samples_csv["Dispatch_Id"] != 0].copy()

    # Determine the agent on which sample was generated.
    samples_df["Agent_Id"] = (
        samples_df["Dispatch_Id"]
        .map(
            input_kernel_trace_csv.set_index("Dispatch_Id")["Agent_Id"]
            .str.split(" ")
            .str[1]
        )
        .astype(np.uint64)
    )
    sampled_agents = set(samples_df["Agent_Id"].unique())
    # Every agent that ran a kernel must have delivered at least one sample.
    missing = kernel_agents - sampled_agents
    assert not missing, (
        f"agents with kernel dispatches but no PC samples: {sorted(missing)}; "
        f"sampled {sorted(sampled_agents)}"
    )

    # separate samples per agents
    grouped_samples_per_agent = samples_df.groupby("Agent_Id")
    for agent_id, agent_samples_df in grouped_samples_per_agent:
        sampled_dispatches = agent_samples_df["Dispatch_Id"].unique()
        # at least 1 sampled dispatch per agent
        assert len(sampled_dispatches) >= 1

    # Samples with no decoded source line have an empty Instruction_Comment.
    # Only check the ones the decoder mapped back to transpose.cpp.
    comments = samples_df["Instruction_Comment"]
    transpose_samples_df = samples_df[
        comments.notna() & comments.astype(str).str.contains("transpose.cpp")
    ].copy()
    if not transpose_samples_df.empty:
        # determine the line number for each sample
        transpose_samples_df["Source_Line_Num"] = transpose_samples_df[
            "Instruction_Comment"
        ].apply(lambda source_line: int(source_line.split(":")[-1]))
        # assert that line belongs to a kernel range
        assert (
            (
                transpose_samples_df["Source_Line_Num"]
                >= transpose_kernel_source_line_start
            )
            & (
                transpose_samples_df["Source_Line_Num"]
                <= transpose_kernel_source_line_end
            )
        ).all()
