# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Environment variable checks: leftovers from other profilers, visibility
masks, and a dump of every ROCPROF*/ROCPROFILER* variable in scope."""

from __future__ import absolute_import

from rocprofv3.doctor_registry import register
from rocprofv3.doctor_result import (
    make_fail,
    make_pass,
    make_warn,
    SEV_INFO,
    SEV_WARNING,
)

# v1/v2 profiler libraries that cannot coexist with rocprofiler-sdk in one
# process: they install their own HSA intercepts and fight over the device
CONFLICTING_PRELOADS = (
    "libroctracer64.so",
    "librocprofiler64.so",
    "librocprofiler64v2.so",
    "librocprof.so",
)

BETA_GATES = (
    ("ROCPROFILER_PC_SAMPLING_BETA_ENABLED", "PC sampling"),
    ("ROCPROFILER_SPM_BETA_ENABLED", "Streaming Performance Monitor"),
)


@register(
    id="environ.ld-preload-conflict",
    group="environ",
    title="LD_PRELOAD free of conflicting profilers",
    severity=SEV_WARNING,
    order=50,
)
def check_ld_preload_conflict(accessor):
    value = accessor.getenv("LD_PRELOAD", "") or ""
    data = {"LD_PRELOAD": value}

    if not value.strip():
        return make_pass("LD_PRELOAD is not set", "", data)

    entries = []
    for entry in value.replace(" ", ":").split(":"):
        if entry:
            entries.append(entry)
    data["entries"] = entries

    conflicts = []
    for entry in entries:
        base = accessor.basename(entry)
        for name in CONFLICTING_PRELOADS:
            if base.startswith(name):
                conflicts.append(entry)
                break
    data["conflicts"] = conflicts

    if conflicts:
        return make_fail(
            "LD_PRELOAD loads a conflicting profiler: {}. Only one profiler may "
            "intercept the ROCm runtime at a time.".format(", ".join(conflicts)),
            "unset LD_PRELOAD",
            data,
        )
    return make_pass(
        "LD_PRELOAD is set ({} entr(y|ies)) but none conflicts with "
        "rocprofiler-sdk".format(len(entries)),
        "",
        data,
    )


@register(
    id="environ.hsa-tools-lib",
    group="environ",
    title="HSA_TOOLS_LIB not pointing at a foreign profiler",
    severity=SEV_WARNING,
    order=51,
)
def check_hsa_tools_lib(accessor):
    value = accessor.getenv("HSA_TOOLS_LIB", "") or ""
    data = {"HSA_TOOLS_LIB": value}

    if not value.strip():
        return make_pass("HSA_TOOLS_LIB is not set", "", data)

    foreign = []
    for entry in value.split(":"):
        if not entry:
            continue
        base = accessor.basename(entry)
        if "rocprofiler-sdk" in base:
            continue
        foreign.append(entry)
    data["foreign_entries"] = foreign

    if foreign:
        return make_warn(
            "HSA_TOOLS_LIB loads non-SDK tool librar(y|ies): {}. These may "
            "conflict with rocprofv3.".format(", ".join(foreign)),
            "unset HSA_TOOLS_LIB",
            data,
        )
    return make_pass("HSA_TOOLS_LIB refers only to rocprofiler-sdk", "", data)


@register(
    id="environ.rocp-tool-libraries",
    group="environ",
    title="ROCP_TOOL_LIBRARIES contents",
    severity=SEV_INFO,
    order=52,
)
def check_rocp_tool_libraries(accessor):
    value = accessor.getenv("ROCP_TOOL_LIBRARIES", "") or ""
    data = {"ROCP_TOOL_LIBRARIES": value}

    if not value.strip():
        return make_pass(
            "ROCP_TOOL_LIBRARIES is not set (rocprofv3 sets it itself)", "", data
        )

    missing = []
    for entry in value.split(":"):
        if entry and not accessor.path_exists(entry):
            missing.append(entry)
    data["missing"] = missing

    if missing:
        return make_warn(
            "ROCP_TOOL_LIBRARIES references non-existent path(s): {}".format(
                ", ".join(missing)
            ),
            "unset ROCP_TOOL_LIBRARIES",
            data,
        )
    return make_pass("ROCP_TOOL_LIBRARIES entries all exist", "", data)


@register(
    id="environ.gpu-visibility",
    group="environ",
    title="GPU visibility variables do not hide every GPU",
    severity=SEV_WARNING,
    order=53,
)
def check_gpu_visibility(accessor):
    data = {}
    set_vars = []
    hiding = []

    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        value = accessor.getenv(name)
        if value is None:
            continue
        data[name] = value
        set_vars.append("{}={!r}".format(name, value))
        if value.strip() == "":
            hiding.append(name)

    if not set_vars:
        return make_pass("no GPU visibility variables are set", "", data)

    if hiding:
        return make_warn(
            "{} set to an empty string, which hides every GPU".format(
                " and ".join(hiding)
            ),
            "unset {}".format(" ".join(hiding)),
            data,
        )

    return make_warn(
        "GPU visibility is restricted: {}. This is often intentional, but it "
        "limits which agents rocprofv3 can profile.".format(", ".join(set_vars)),
        "unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES",
        data,
    )


@register(
    id="environ.beta-gates",
    group="environ",
    title="Beta feature gates",
    severity=SEV_INFO,
    order=54,
)
def check_beta_gates(accessor):
    data = {}
    enabled = []
    for name, description in BETA_GATES:
        value = accessor.getenv(name)
        data[name] = value
        if value and value.lower() not in ("0", "off", "false", "no"):
            enabled.append(description)

    if enabled:
        return make_pass("beta features enabled: {}".format(", ".join(enabled)), "", data)
    return make_pass(
        "no beta feature gates set; PC sampling and SPM require "
        "ROCPROFILER_PC_SAMPLING_BETA_ENABLED / ROCPROFILER_SPM_BETA_ENABLED",
        "",
        data,
    )


@register(
    id="environ.rocprof-vars-dump",
    group="environ",
    title="rocprofiler-related environment variables",
    severity=SEV_INFO,
    order=55,
)
def check_rocprof_vars_dump(accessor):
    # This check always passes; it exists so that an unexpected leftover
    # variable shows up in --verbose and JSON output where a human or a bug
    # report reader can spot it.
    # Deliberately narrow: HSA_/HIP_/AMD_ are set on most ROCm dev machines for
    # reasons unrelated to profiling, and including them buries the leftover
    # ROCPROF_ variable this check exists to surface.
    prefixes = ("ROCPROF_", "ROCPROFILER_", "ROCP_")
    found = {}
    for key, value in accessor.environ_items():
        for prefix in prefixes:
            if key.startswith(prefix):
                found[key] = value
                break

    if not found:
        return make_pass("no rocprofiler-related environment variables set", "", found)
    return make_pass(
        "{} rocprofiler-related variable(s) set: {}".format(
            len(found), ", ".join(sorted(found.keys()))
        ),
        "",
        found,
    )
