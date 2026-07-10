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

MIN_V_MOV_B32_SAMPLES = 100
MIN_V_MOV_B32_RATIO = 0.30

RESUME_WINDOW_KERNELS = ("pc_sampling_kernel", "target_kernel")
PAUSED_KERNELS = ("kernel_add", "kernel_mult")
REF_COUNT_KERNEL = "nested_kernel"


def _tool(json_data):
    tool = json_data["rocprofiler-sdk-tool"]
    return tool[0] if isinstance(tool, list) else tool


def _records_key(method):
    return "pc_sample_host_trap" if method == "host_trap" else "pc_sample_stochastic"


def validate_csv_json_parity(df, json_data, method):
    # CSV and JSON report the same set of samples
    records = _tool(json_data)["buffer_records"][_records_key(method)]
    assert len(records) > 0, f"no {method} PC sampling records in JSON"
    assert len(records) == len(df), (
        f"CSV rows ({len(df)}) != JSON records ({len(records)})"
    )


def validate_data_integrity(json_data, method):
    # samples contains v_mov_b32 instruction
    tool = _tool(json_data)
    records = tool["buffer_records"][_records_key(method)]
    instructions = tool["strings"]["pc_sample_instructions"]

    v_mov_b32_count = 0
    for sample in records:
        inst_index = sample["inst_index"]
        if inst_index >= 0 and instructions[inst_index].startswith("v_mov_b32"):
            v_mov_b32_count += 1

    assert v_mov_b32_count >= MIN_V_MOV_B32_SAMPLES, (
        f"expected >= {MIN_V_MOV_B32_SAMPLES} v_mov_b32 samples, got {v_mov_b32_count}"
    )
    ratio = v_mov_b32_count / len(records)
    assert ratio >= MIN_V_MOV_B32_RATIO, (
        f"expected v_mov_b32 samples >= {MIN_V_MOV_B32_RATIO:.0%}, got {ratio:.2%}"
    )


def _dispatch_id_to_kernel_name(tool):
    # map each dispatch_id to its kernel name using the kernel trace
    ks = tool["kernel_symbols"]
    if isinstance(ks, list):
        names = {k["kernel_id"]: k["formatted_kernel_name"] for k in ks}
    else:
        names = {int(k): v["formatted_kernel_name"] for k, v in ks.items()}
    d2k = {}
    for kd in tool["buffer_records"].get("kernel_dispatch", []):
        di = kd.get("dispatch_info") or kd.get("dispatch_data", {}).get("dispatch_info", {})
        did, kid = di.get("dispatch_id"), di.get("kernel_id")
        if did is not None and kid in names:
            d2k[did] = names[kid]
    return d2k


def _kernel_dispatch_names(tool):
    # list every kernel name present in the kernel trace
    ks = tool["kernel_symbols"]
    if isinstance(ks, list):
        names = {k["kernel_id"]: k["formatted_kernel_name"] for k in ks}
    else:
        names = {int(k): v["formatted_kernel_name"] for k, v in ks.items()}
    dispatched = []
    for kd in tool["buffer_records"].get("kernel_dispatch", []):
        di = kd.get("dispatch_info") or {}
        name = names.get(di.get("kernel_id"))
        if name:
            dispatched.append(name)
    return dispatched


def _kernel_sample_counts(json_data, method):
    # count PC samples per kernel via dispatch_id
    tool = _tool(json_data)
    d2k = _dispatch_id_to_kernel_name(tool)
    counts = {}
    for sample in tool["buffer_records"][_records_key(method)]:
        rec = sample.get("record", sample)
        name = d2k.get(rec.get("dispatch_id"), "<unmapped>")
        counts[name] = counts.get(name, 0) + 1
    assert counts, "no PC samples mapped to any kernel"
    assert "<unmapped>" not in counts, (
        f"some PC samples did not map to a dispatched kernel: {counts}"
    )
    return counts


def _count_for(names_or_counts, kernel):
    # count of a kernel in samples or in the trace
    if isinstance(names_or_counts, dict):
        return sum(n for name, n in names_or_counts.items() if kernel in name)
    return sum(1 for name in names_or_counts if kernel in name)


def _assert_paused_kernels_silent(counts):
    # kernels that only run while paused must produce no samples
    for paused in PAUSED_KERNELS:
        leaked = _count_for(counts, paused)
        assert leaked == 0, (
            f"'{paused}' runs only while paused but produced {leaked} PC samples "
            f"— selected-regions gating failed"
        )


def _assert_only(counts, allowed):
    # only allowed kernels are sampled
    for name in counts:
        assert any(a in name for a in allowed), (
            f"PC samples came from unexpected kernel '{name}'; expected only {allowed}"
        )


def validate_selected_regions_gating(json_data, method):
    tool = _tool(json_data)
    counts = _kernel_sample_counts(json_data, method)
    _assert_paused_kernels_silent(counts)
    # only the resume-pause window kernels are sampled
    _assert_only(counts, RESUME_WINDOW_KERNELS)
    # nested_kernel should not be sampled
    assert _count_for(counts, REF_COUNT_KERNEL) == 0, (
        f"'{REF_COUNT_KERNEL}' produced PC samples without --selected-regions-ref-count"
    )
    # no nested_kernel in kernel_trace
    assert _count_for(_kernel_dispatch_names(tool), REF_COUNT_KERNEL) == 0, (
        f"'{REF_COUNT_KERNEL}' should not be traced without --selected-regions-ref-count"
    )


def validate_selected_regions_ref_count_gating(json_data, method):
    tool = _tool(json_data)
    counts = _kernel_sample_counts(json_data, method)
    _assert_paused_kernels_silent(counts)
    # nested_kernel is traced
    _assert_only(counts, RESUME_WINDOW_KERNELS + (REF_COUNT_KERNEL,))
    # only one nested_kernel traced
    traced = _count_for(_kernel_dispatch_names(tool), REF_COUNT_KERNEL)
    assert traced == 1, (
        f"expected exactly one '{REF_COUNT_KERNEL}' in the kernel dispatch trace "
        f"with --selected-regions-ref-count, got {traced}"
    )
