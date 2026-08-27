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

# Canonical method names used to select the PC-sample record stream.
HOST_TRAP_METHOD_STR = "host_trap"
STOCHASTIC_METHOD_STR = "stochastic"

_RECORD_KEY_BY_METHOD = {
    HOST_TRAP_METHOD_STR: "pc_sample_host_trap",
    STOCHASTIC_METHOD_STR: "pc_sample_stochastic",
}

MIN_V_MOV_B32_SAMPLES = 100
MIN_V_MOV_B32_RATIO = 0.30

RESUME_WINDOW_KERNELS = ("pc_sampling_kernel", "target_kernel")
PAUSED_KERNELS = ("kernel_add", "kernel_mult")
REF_COUNT_KERNEL = "nested_kernel"


def get_tool(json_data):
    # Normalize the tool payload to a single dictionary.
    tool = json_data["rocprofiler-sdk-tool"]
    if isinstance(tool, list):
        assert len(tool) == 1, f"expected one rocprofiler-sdk-tool entry, got {len(tool)}"
        tool = tool[0]
    return tool


def get_records(json_data, method):
    assert method in _RECORD_KEY_BY_METHOD, f"unsupported PC sampling method: {method}"
    record_key = _RECORD_KEY_BY_METHOD[method]
    records = get_tool(json_data)["buffer_records"][record_key]
    assert isinstance(records, list), f"expected '{record_key}' to contain a list"
    return records


def validate_csv_json_parity_num_samples(df, json_data, method):
    # CSV and JSON must report the same sample count.
    records = get_records(json_data, method)
    assert len(records) > 0, f"no {method} PC sampling records in JSON"
    assert len(records) == len(
        df
    ), f"CSV rows ({len(df)}) != JSON records ({len(records)})"


def validate_data_integrity(json_data, method):
    # Require a representative volume of the workload's hot instruction.
    tool = get_tool(json_data)
    records = get_records(json_data, method)
    instructions = tool["strings"]["pc_sample_instructions"]

    v_mov_b32_count = 0
    for sample in records:
        inst_index = sample["inst_index"]
        if inst_index >= 0 and instructions[inst_index].startswith("v_mov_b32"):
            v_mov_b32_count += 1

    assert (
        v_mov_b32_count >= MIN_V_MOV_B32_SAMPLES
    ), f"expected >= {MIN_V_MOV_B32_SAMPLES} v_mov_b32 samples, got {v_mov_b32_count}"
    ratio = v_mov_b32_count / len(records)
    assert (
        ratio >= MIN_V_MOV_B32_RATIO
    ), f"expected v_mov_b32 samples >= {MIN_V_MOV_B32_RATIO:.0%}, got {ratio:.2%}"


def _dispatch_id_to_kernel_name(tool):
    # Map each dispatch ID to its kernel name using the kernel trace.
    ks = tool["kernel_symbols"]
    if isinstance(ks, list):
        names = {k["kernel_id"]: k["formatted_kernel_name"] for k in ks}
    else:
        names = {int(k): v["formatted_kernel_name"] for k, v in ks.items()}
    d2k = {}
    kernel_dispatches = tool["buffer_records"]["kernel_dispatch"]
    assert kernel_dispatches, "expected kernel-dispatch trace records"
    for kd in kernel_dispatches:
        # --kernel-trace records contain dispatch_info at the top level.
        di = kd["dispatch_info"]
        did, kid = di["dispatch_id"], di["kernel_id"]
        assert kid in names, f"kernel_id {kid} is missing from kernel_symbols"
        d2k[did] = names[kid]
    return d2k


def _kernel_dispatch_names(tool):
    # List every kernel name present in the kernel trace.
    ks = tool["kernel_symbols"]
    if isinstance(ks, list):
        names = {k["kernel_id"]: k["formatted_kernel_name"] for k in ks}
    else:
        names = {int(k): v["formatted_kernel_name"] for k, v in ks.items()}
    dispatched = []
    kernel_dispatches = tool["buffer_records"]["kernel_dispatch"]
    assert kernel_dispatches, "expected kernel-dispatch trace records"
    for kd in kernel_dispatches:
        # --kernel-trace records contain dispatch_info at the top level.
        di = kd["dispatch_info"]
        kid = di["kernel_id"]
        assert kid in names, f"kernel_id {kid} is missing from kernel_symbols"
        dispatched.append(names[kid])
    return dispatched


def _kernel_sample_counts(json_data, method):
    # Count decoded PC samples per kernel using dispatch IDs.
    tool = get_tool(json_data)
    d2k = _dispatch_id_to_kernel_name(tool)
    counts = {}
    for sample in get_records(json_data, method):
        rec = sample["record"]
        correlation_id = rec["corr_id"]["internal"]
        dispatch_id = rec["dispatch_id"]
        inst_index = sample["inst_index"]
        code_object_id = rec["pc"]["code_object_id"]

        assert correlation_id >= 0

        if correlation_id == 0:
            # Validate unmappable blit/self-modifying-code sample sentinels.
            assert dispatch_id == 0, "uncorrelated samples must have Dispatch_Id == 0"
            assert (
                inst_index == -1
            ), "uncorrelated samples must not have a decoded instruction"
            assert code_object_id == 0, "uncorrelated samples must not have a code object"
            continue

        # Correlation and dispatch IDs are independent for decoded samples.
        assert dispatch_id > 0
        assert inst_index >= 0
        assert code_object_id != 0

        name = d2k.get(dispatch_id, "<unmapped>")
        counts[name] = counts.get(name, 0) + 1
    assert counts, "no decoded PC samples mapped to any kernel"
    assert (
        "<unmapped>" not in counts
    ), f"some PC samples did not map to a dispatched kernel: {counts}"
    return counts


def _count_for(names_or_counts, kernel):
    # Count a kernel in either sampled counts or traced names.
    if isinstance(names_or_counts, dict):
        return sum(n for name, n in names_or_counts.items() if kernel in name)
    return sum(1 for name in names_or_counts if kernel in name)


def _assert_paused_kernels_silent(counts):
    # Kernels launched while paused must produce no samples.
    for paused in PAUSED_KERNELS:
        leaked = _count_for(counts, paused)
        assert leaked == 0, (
            f"'{paused}' runs only while paused but produced {leaked} PC samples "
            f"— selected-regions gating failed"
        )


def _assert_only(counts, allowed):
    # Reject samples from kernels outside the selected regions.
    for name in counts:
        assert any(
            a in name for a in allowed
        ), f"PC samples came from unexpected kernel '{name}'; expected only {allowed}"


def validate_selected_regions_gating(json_data, method):
    tool = get_tool(json_data)
    counts = _kernel_sample_counts(json_data, method)
    _assert_paused_kernels_silent(counts)
    # Only resume-window kernels may be sampled.
    _assert_only(counts, RESUME_WINDOW_KERNELS)
    # Nested profiling remains disabled without reference counting.
    assert (
        _count_for(counts, REF_COUNT_KERNEL) == 0
    ), f"'{REF_COUNT_KERNEL}' produced PC samples without --selected-regions-ref-count"
    # The kernel trace must follow the same gating behavior.
    assert (
        _count_for(_kernel_dispatch_names(tool), REF_COUNT_KERNEL) == 0
    ), f"'{REF_COUNT_KERNEL}' should not be traced without --selected-regions-ref-count"


def validate_selected_regions_ref_count_gating(json_data, method):
    tool = get_tool(json_data)
    counts = _kernel_sample_counts(json_data, method)
    _assert_paused_kernels_silent(counts)
    # Reference counting allows the nested kernel inside the outer region.
    _assert_only(counts, RESUME_WINDOW_KERNELS + (REF_COUNT_KERNEL,))
    # Exactly one nested kernel is inside the active nested region.
    traced = _count_for(_kernel_dispatch_names(tool), REF_COUNT_KERNEL)
    assert traced == 1, (
        f"expected exactly one '{REF_COUNT_KERNEL}' in the kernel dispatch trace "
        f"with --selected-regions-ref-count, got {traced}"
    )
