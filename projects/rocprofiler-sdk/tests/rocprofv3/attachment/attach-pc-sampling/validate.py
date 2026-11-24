#!/usr/bin/env python3

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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def test_pc_sampling_attach_output(json_data, method):
    tool = json_data["rocprofiler-sdk-tool"]
    records = tool["buffer_records"]

    kernel_dispatches = records.get("kernel_dispatch", [])
    assert kernel_dispatches, "late attachment did not capture any kernel dispatches"

    sample_kind = {
        "host_trap": "pc_sample_host_trap",
        "stochastic": "pc_sample_stochastic",
    }[method]
    samples = records.get(sample_kind, [])
    assert samples, f"late attachment did not capture any {method} PC samples"

    dispatch_correlations = {
        dispatch["dispatch_info"]["dispatch_id"]: dispatch["correlation_id"]["internal"]
        for dispatch in kernel_dispatches
    }
    correlated_samples = 0
    mapped_samples = 0
    for sample in samples:
        record = sample["record"]
        correlation_id = record["corr_id"]["internal"]
        dispatch_id = record["dispatch_id"]

        if correlation_id == 0:
            assert dispatch_id == 0, "uncorrelated samples must use dispatch ID zero"
            continue

        assert dispatch_id > 0
        assert sample["inst_index"] >= 0, "correlated samples must decode an instruction"
        correlated_samples += 1

        # A high-throughput workload can overflow the kernel-trace buffer independently of PC
        # sampling. Validate every dispatch that survived in both streams and require overlap.
        if dispatch_id in dispatch_correlations:
            assert correlation_id == dispatch_correlations[dispatch_id]
            mapped_samples += 1

    assert correlated_samples > 0, "no correlated PC samples were decoded"
    assert mapped_samples > 0, "no PC samples overlap the retained kernel trace"
