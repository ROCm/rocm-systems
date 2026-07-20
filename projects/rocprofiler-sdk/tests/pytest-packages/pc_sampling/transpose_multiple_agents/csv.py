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
import re
import sys
import pytest
import numpy as np
import pandas as pd


# Cache of parsed source files so that we only read/parse each file once.
_kernel_source_range_cache = {}


def _extract_source_file_and_line(instruction_comment: str, source_basename: str):
    """Extract the (absolute_source_path, line_number) pair that refers to
    ``source_basename`` from an instruction comment.

    The instruction comment produced by the PC sampling decoder has the form
    ``/path/to/file.cpp:LINE`` and, when instructions are inlined, a chain such
    as ``/hdr.h:12 -> /hdr.h:44 -> /path/to/transpose.cpp:198``.  We therefore
    look at every ``file:line`` segment and return the one that belongs to the
    requested source file instead of blindly taking the last segment.
    """

    for segment in re.split(r"\s*->\s*", instruction_comment.strip()):
        path, sep, line = segment.rpartition(":")
        if not sep or not path:
            continue
        if os.path.basename(path) == source_basename and line.isdigit():
            return path, int(line)
    return None, None


def _kernel_source_line_range(source_path: str, kernel_name: str):
    """Return the inclusive ``(start_line, end_line)`` range (1-based) of the
    ``__global__`` kernel definition ``kernel_name`` inside ``source_path``.

    The range is derived directly from the source file so that the validation
    does not rely on hard-coded line numbers that silently drift whenever the
    test kernel source is edited.
    """

    cache_key = (source_path, kernel_name)
    if cache_key in _kernel_source_range_cache:
        return _kernel_source_range_cache[cache_key]

    assert os.path.isfile(
        source_path
    ), f"kernel source file '{source_path}' referenced by the samples is not readable"

    with open(source_path, "r") as src:
        lines = src.readlines()

    num_lines = len(lines)
    signature_re = re.compile(r"\b%s\s*\(" % re.escape(kernel_name))
    kernel_range = None

    for idx, line in enumerate(lines):
        if "__global__" not in line:
            continue

        # The signature may be on the same line as __global__ or a few lines
        # below it (return type and name split across lines).
        signature_idx = next(
            (
                j
                for j in range(idx, min(idx + 5, num_lines))
                if signature_re.search(lines[j])
            ),
            None,
        )
        if signature_idx is None:
            continue

        # Find the opening brace of the definition. If we hit a ';' first this
        # is only a forward declaration, so keep searching.
        open_brace_idx = None
        for j in range(signature_idx, min(signature_idx + 8, num_lines)):
            if "{" in lines[j]:
                open_brace_idx = j
                break
            if ";" in lines[j]:
                break
        if open_brace_idx is None:
            continue

        # Brace-match to find the end of the kernel body.
        depth = 0
        for j in range(open_brace_idx, num_lines):
            depth += lines[j].count("{") - lines[j].count("}")
            if depth == 0:
                kernel_range = (idx + 1, j + 1)
                break
        if kernel_range is not None:
            break

    assert (
        kernel_range is not None
    ), f"could not locate __global__ kernel '{kernel_name}' in '{source_path}'"

    _kernel_source_range_cache[cache_key] = kernel_range
    return kernel_range


def validate_all_agents_are_sampled(
    input_samples_csv: pd.DataFrame,
    input_kernel_trace_csv: pd.DataFrame,
    input_agent_info_csv: pd.DataFrame,
):
    transpose_source_basename = "transpose.cpp"
    transpose_kernel_name = "transpose"

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
    # all agents must be sampled
    assert sampled_agents_num == len(gfx9_gfx12_agents_df)

    # separate samples per agents
    grouped_samples_per_agent = samples_df.groupby("Agent_Id")
    for agent_id, agent_samples_df in grouped_samples_per_agent:
        sampled_dispatches = agent_samples_df["Dispatch_Id"].unique()
        # at least 1 sampled dispatch per agent
        assert len(sampled_dispatches) >= 1

    # extract decoded samples that are mapped to the transpose.cpp file
    transpose_samples_df = samples_df[
        samples_df["Instruction_Comment"].apply(
            lambda comment: transpose_source_basename in comment
        )
    ].copy()

    # determine the source file and line number for each sample, parsing the
    # transpose.cpp segment of the (possibly inlined) instruction comment
    source_refs = transpose_samples_df["Instruction_Comment"].apply(
        lambda comment: _extract_source_file_and_line(
            comment, transpose_source_basename
        )
    )
    transpose_samples_df["Source_File"] = source_refs.apply(lambda ref: ref[0])
    transpose_samples_df["Source_Line_Num"] = source_refs.apply(lambda ref: ref[1])

    # every extracted reference must resolve to a concrete file:line pair
    assert (
        transpose_samples_df["Source_Line_Num"].notna().all()
    ), "failed to parse transpose.cpp source line from an instruction comment"

    # derive the kernel's line range directly from the source file instead of
    # relying on hard-coded line numbers that drift when the kernel is edited
    for source_file in transpose_samples_df["Source_File"].unique():
        start_line, end_line = _kernel_source_line_range(
            source_file, transpose_kernel_name
        )
        file_samples = transpose_samples_df[
            transpose_samples_df["Source_File"] == source_file
        ]
        # assert that every sampled line belongs to the transpose kernel range
        assert (
            (file_samples["Source_Line_Num"] >= start_line)
            & (file_samples["Source_Line_Num"] <= end_line)
        ).all(), (
            f"transpose.cpp samples map outside the kernel '{transpose_kernel_name}' "
            f"line range [{start_line}, {end_line}] in {source_file}"
        )
