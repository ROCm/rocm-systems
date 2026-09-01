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

import itertools
import os
import pathlib
import sys
import warnings
import pytest
import numpy as np
import pandas as pd


def _transpose_kernel_line_range(source_paths):
    """Returns the inclusive line range of the transpose kernel body, or None if the
    source cannot be read.

    The range is read back from transpose.cpp rather than hardcoded: the kernel has
    already moved within the file once, which left the previous fixed window pointing
    at unrelated lines and failed the suite on every machine.
    """
    for path in source_paths:
        try:
            lines = pathlib.Path(path).read_text().splitlines()
        except OSError:
            continue

        for idx, line in enumerate(lines):
            # The definition spans two lines: "__global__ void" then "transpose(...)".
            if not line.lstrip().startswith("transpose("):
                continue
            if "__global__" not in "".join(lines[max(0, idx - 2) : idx]):
                continue

            # Walk the signature to its body, skipping the forward declaration that
            # carries the same name and ends in a semicolon instead of a brace.
            body_start = None
            for i in range(idx, min(idx + 10, len(lines))):
                stripped = lines[i].strip()
                if stripped.endswith(";"):
                    break
                if stripped.endswith("{"):
                    body_start = i
                    break
            if body_start is None:
                continue

            depth = 0
            for body_end in range(body_start, len(lines)):
                depth += lines[body_end].count("{") - lines[body_end].count("}")
                if depth == 0:
                    # Convert the 0-based indices of '{' and '}' to 1-based line numbers.
                    return body_start + 1, body_end + 1

    return None


def _expected_sampled_agents_num(available_agents_num):
    """Returns how many agents the application could actually dispatch to.

    A runner may expose every GPU to the process via KFD while restricting the
    application to a subset, in which case fewer agents are sampled than agent_info.csv
    reports and the all-agents check has to be relaxed accordingly.
    """
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        value = os.environ.get(var)
        if not value or not value.strip():
            continue
        visible_num = len([tok for tok in value.split(",") if tok.strip()])
        return min(visible_num, available_agents_num)

    return available_agents_num


def validate_all_agents_are_sampled(
    input_samples_csv: pd.DataFrame,
    input_kernel_trace_csv: pd.DataFrame,
    input_agent_info_csv: pd.DataFrame,
):
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
    samples_df["Agent_Id"] = (
        samples_df["Dispatch_Id"]
        .map(
            input_kernel_trace_csv.set_index("Dispatch_Id")["Agent_Id"]
            .str.split(" ")
            .str[1]
        )
        .astype(np.uint64)
    )
    sampled_agents = samples_df["Agent_Id"].unique()
    sampled_agents_num = len(sampled_agents)
    # every agent the application could dispatch to must be sampled
    assert sampled_agents_num == _expected_sampled_agents_num(len(gfx9_gfx12_agents_df))

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
    # determine the line number for each sample
    transpose_samples_df["Source_Line_Num"] = transpose_samples_df[
        "Instruction_Comment"
    ].apply(lambda source_line: int(source_line.split(":")[-1]))

    # the comment is "<source path>:<line>", so the samples carry the path of the
    # transpose.cpp they were compiled from
    source_paths = {
        comment.rsplit(":", 1)[0]
        for comment in transpose_samples_df["Instruction_Comment"]
    }
    kernel_line_range = _transpose_kernel_line_range(source_paths)

    if kernel_line_range is None:
        warnings.warn(
            "Could not read the transpose kernel line range from any of "
            f"{sorted(source_paths)}; skipping the source line range check."
        )
    else:
        # assert that line belongs to a kernel range
        kernel_line_start, kernel_line_end = kernel_line_range
        assert (
            (transpose_samples_df["Source_Line_Num"] >= kernel_line_start)
            & (transpose_samples_df["Source_Line_Num"] <= kernel_line_end)
        ).all()
