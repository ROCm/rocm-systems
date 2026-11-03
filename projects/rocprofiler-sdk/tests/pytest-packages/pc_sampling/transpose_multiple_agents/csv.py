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
    agent_index: str = "relative",
):
    if agent_index not in {"absolute", "relative", "type-relative"}:
        raise ValueError(f"unsupported agent index mode: {agent_index}")

    gpu_agents_df = input_agent_info_csv[input_agent_info_csv["Agent_Type"] == "GPU"].copy()
    # PC sampling is supported on gfx90a, gfx94x/gfx95x, and gfx12 GPUs.
    supported_gpus_df = gpu_agents_df[
        gpu_agents_df["Name"].apply(
            lambda name: name == "gfx90a"
            or name.startswith("gfx94")
            or name.startswith("gfx95")
            or name.startswith("gfx12")
        )
    ]
    assert len(supported_gpus_df) >= 1

    if agent_index == "absolute":
        supported_agent_ids = set(supported_gpus_df["Node_Id"].astype(np.uint64))
    elif agent_index == "relative":
        supported_agent_ids = set(supported_gpus_df["Logical_Node_Id"].astype(np.uint64))
    else:
        # agent_info.csv omits Logical_Node_Type_Id. GPU ordinals follow KFD node order.
        gpu_agents_df = gpu_agents_df.sort_values("Node_Id")
        gpu_agents_df["Agent_Index"] = np.arange(len(gpu_agents_df), dtype=np.uint64)
        supported_agent_ids = set(
            gpu_agents_df.loc[supported_gpus_df.index, "Agent_Index"].astype(np.uint64)
        )

    # Agent_Id is "Agent N" for absolute/relative and "GPU N" for type-relative.
    def agent_ordinals(agent_ids: pd.Series) -> set:
        parsed = agent_ids.astype(str).str.extract(r"^(Agent|GPU) ([0-9]+)$")
        assert parsed.notna().all().all(), f"invalid Agent_Id values: {agent_ids.unique()}"
        expected_prefix = "GPU" if agent_index == "type-relative" else "Agent"
        assert parsed[0].eq(expected_prefix).all(), (
            f"Agent_Id format does not match --agent-index {agent_index}: "
            f"{agent_ids.unique()}"
        )
        return set(parsed[1].astype(np.uint64))

    # Only agents with kernel dispatches can be expected to produce samples.
    kernel_agents = agent_ordinals(input_kernel_trace_csv["Agent_Id"])
    expected_agents = kernel_agents & supported_agent_ids
    assert expected_agents, "no supported GPU ran a kernel dispatch"

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
    # Every supported agent that ran a kernel must have delivered at least one sample.
    missing = expected_agents - sampled_agents
    assert not missing, (
        f"supported agents with kernel dispatches but no PC samples: {sorted(missing)}; "
        f"sampled {sorted(sampled_agents)}"
    )

    # separate samples per agents
    grouped_samples_per_agent = samples_df.groupby("Agent_Id")
    for agent_id, agent_samples_df in grouped_samples_per_agent:
        sampled_dispatches = agent_samples_df["Dispatch_Id"].unique()
        # at least 1 sampled dispatch per agent
        assert len(sampled_dispatches) >= 1
