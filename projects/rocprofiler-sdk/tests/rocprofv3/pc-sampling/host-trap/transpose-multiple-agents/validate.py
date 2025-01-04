#!/usr/bin/env python3

import itertools
import sys
import pytest
import numpy as np
import pandas as pd


def test_multi_agent_support(
    input_samples_csv: pd.DataFrame,
    input_kernel_trace_csv: pd.DataFrame,
    input_agent_info_csv: pd.DataFrame,
):
    transpose_kernel_source_line_start = 137
    transpose_kernel_source_line_end = 145

    mi2xx_mi3xx_agents_df = input_agent_info_csv[
        input_agent_info_csv["Name"].apply(
            lambda name: name == "gfx90a" or name.startswith("gfx94")
        )
    ]

    # Extract samples that originates from know code object it
    samples_df = input_samples_csv[input_samples_csv["Dispatch_Id"] != 0].copy()

    # Determine the agent on which sample was generated
    samples_df["Agent_Id"] = (
        samples_df["Dispatch_Id"]
        .map(input_kernel_trace_csv.set_index("Dispatch_Id")["Agent_Id"])
        .astype(np.uint64)
    )
    sampled_agents = samples_df["Agent_Id"].unique()
    sampled_agents_num = len(sampled_agents)
    # all agents must be sampled
    assert sampled_agents_num == len(mi2xx_mi3xx_agents_df)

    # separate samples per agents
    grouped_samples_per_agent = samples_df.groupby("Agent_Id")
    for agent_id, agent_samples_df in grouped_samples_per_agent:
        sampled_dispatches = agent_samples_df["Dispatch_Id"].unique()
        # at least 1 sampled dispatch per agent
        assert len(sampled_dispatches) >= 1

    # extract decoded samples that are mapped to the transpose.cpp file
    transpose_samples_df = samples_df[
        samples_df["Instruction_Comment"].apply(
            lambda comment: "transpose-all-agents.cpp" in comment
        )
    ].copy()
    # determine the line number for each sample
    transpose_samples_df["Source_Line_Num"] = transpose_samples_df[
        "Instruction_Comment"
    ].apply(lambda source_line: int(source_line.split(":")[-1]))
    # assert that line belongs to a kernel range
    assert (
        (transpose_samples_df["Source_Line_Num"] >= transpose_kernel_source_line_start)
        & (transpose_samples_df["Source_Line_Num"] <= transpose_kernel_source_line_end)
    ).all()


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
