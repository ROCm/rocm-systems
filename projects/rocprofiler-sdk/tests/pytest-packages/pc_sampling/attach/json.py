# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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


def _tool(json_data):
    tool = json_data["rocprofiler-sdk-tool"]
    return tool[0] if isinstance(tool, list) else tool


def validate_attach_json(json_data, method, csv_row_count):
    tool = _tool(json_data)
    records = tool["buffer_records"]["pc_sample_" + method]
    instructions = tool["strings"]["pc_sample_instructions"]

    # CSV and JSON must describe the same set of samples
    assert (
        len(records) == csv_row_count
    ), f"CSV rows ({csv_row_count}) != JSON records ({len(records)})"

    for sample in records:
        rec = sample.get("record", sample)
        inst_index = sample["inst_index"]
        # inst_index is -1 (undecodable) or a valid index into the string table
        assert inst_index == -1 or 0 <= inst_index < len(
            instructions
        ), f"inst_index out of range: {inst_index}"
        dispatch_id = rec.get("dispatch_id")
        internal_cid = rec.get("corr_id", {}).get("internal")
        # a sample is correlated to a dispatch iff it has a correlation id
        assert (internal_cid == 0) == (dispatch_id == 0), (
            "corr_id.internal and dispatch_id disagree on whether the sample "
            "is correlated"
        )
        # correlated samples must carry a non-zero execution mask
        if dispatch_id and dispatch_id > 0:
            assert rec.get("exec_mask", 0) > 0, "correlated sample has exec_mask == 0"
