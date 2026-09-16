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

"""Opt-in smoke checks that run the profiler's own tools.

Registered with ``default_enabled=False``: they run only with
``--run-smoke-test`` or when named with ``--only`` -- either is an explicit
opt-in, and a check that is selected always runs.

What they establish is deliberately narrow. ``smoke.avail-info`` shows the HSA
runtime initializes and enumerates GPU agents through rocprofiler-sdk.
``smoke.rocprofv3-launcher`` shows rocprofv3 can start, inject its tool
library, and run a program to completion; the program launches no GPU work, so
it says nothing about kernel execution, callback delivery, or trace output.
"""

from __future__ import absolute_import

from rocprofv3.doctor_checks_driver import gpu_nodes
from rocprofv3.doctor_checks_tools import resolve_tool, tool_command
from rocprofv3.doctor_diagnose import explain_failure
from rocprofv3.doctor_env import RUN_TIMEOUT
from rocprofv3.doctor_registry import register
from rocprofv3.doctor_result import (
    make_fail,
    make_pass,
    make_skip,
    make_warn,
    PROBE_GPU,
    PROBE_PROCESS,
    SEV_INFO,
)


@register(
    id="smoke.avail-info",
    group="smoke",
    title="rocprofv3-avail info reports at least one GPU",
    severity=SEV_INFO,
    depends=["tools.rocprofv3-avail"],
    order=100,
    default_enabled=False,
    probe=PROBE_GPU,
)
def check_avail_info(accessor):
    path, _ = resolve_tool(accessor, "rocprofv3-avail")
    if path is None:
        return make_skip("rocprofv3-avail was not found")
    returncode, stdout, stderr = accessor.run(
        tool_command(accessor, path, ["info"]), timeout=30
    )
    data = {"tool": path, "returncode": returncode, "stderr_tail": stderr.strip()[-400:]}

    if returncode == RUN_TIMEOUT:
        return make_fail(
            "rocprofv3-avail info timed out after 30 seconds",
            "Check dmesg for amdgpu errors:\n  sudo dmesg | grep -i amdgpu",
            data,
        )

    gpu_lines = []
    for line in stdout.splitlines():
        if line.strip().startswith("GPU"):
            gpu_lines.append(line.strip())
    data["gpu_lines"] = gpu_lines[:16]

    if returncode != 0:
        detail, remediation, data["diagnoses"] = explain_failure(
            accessor,
            "rocprofv3-avail info",
            returncode,
            stdout,
            stderr,
            "Run it directly to see the error:\n  rocprofv3-avail info",
        )
        return make_warn(detail, remediation, data)
    if not gpu_lines:
        return make_warn(
            "rocprofv3-avail info succeeded but reported no GPU agents",
            "Confirm a GPU is visible:\n  rocm_agent_enumerator",
            data,
        )
    return make_pass(
        "rocprofv3-avail info reported {} GPU line(s)".format(len(gpu_lines)), "", data
    )


@register(
    id="smoke.rocprofv3-launcher",
    group="smoke",
    title="rocprofv3 starts, injects its tool, and runs a program",
    severity=SEV_INFO,
    depends=["tools.rocprofv3"],
    order=101,
    default_enabled=False,
    probe=PROBE_PROCESS,
)
def check_rocprofv3_launcher(accessor):
    if not accessor.path_exists("/bin/true"):
        return make_skip("/bin/true is not available on this system")

    rocprofv3, _ = resolve_tool(accessor, "rocprofv3")
    if rocprofv3 is None:
        return make_skip("rocprofv3 was not found")

    # output goes to a directory this run owns: removed on success, kept on
    # failure so whatever rocprofv3 managed to write can be inspected
    output_dir = accessor.make_temp_dir()
    if output_dir is None:
        return make_skip("could not create a temporary output directory")
    command = ["--kernel-trace", "-d", output_dir, "--", "/bin/true"]
    returncode, stdout, stderr = accessor.run(
        tool_command(accessor, rocprofv3, command), timeout=30
    )
    data = {
        "tool": rocprofv3,
        "returncode": returncode,
        "stderr_tail": stderr.strip()[-600:],
        "stdout_tail": stdout.strip()[-600:],
    }
    if returncode == 0:
        accessor.remove_tree(output_dir)
        return make_pass(
            "rocprofv3 started and ran /bin/true to completion; no GPU work was "
            "exercised, so kernel tracing itself is not verified",
            "",
            data,
        )
    data["output_dir"] = output_dir

    if returncode == RUN_TIMEOUT:
        return make_fail(
            "rocprofv3 hung for 30 seconds running /bin/true",
            "Check dmesg for amdgpu errors:\n  sudo dmesg | grep -i amdgpu\n"
            "Partial output was kept in {}".format(output_dir),
            data,
        )

    # Without a usable GPU a non-zero exit is expected rather than a defect,
    # so degrade to warn rather than reporting a failure the user cannot fix.
    has_gpu = bool(gpu_nodes(accessor))
    data["gpu_detected"] = has_gpu
    if not has_gpu:
        accessor.remove_tree(output_dir)
        del data["output_dir"]
        return make_warn(
            "rocprofv3 exited {} on /bin/true, which is expected with no GPU "
            "present".format(returncode),
            "",
            data,
        )
    detail, remediation, data["diagnoses"] = explain_failure(
        accessor,
        "rocprofv3 --kernel-trace on /bin/true",
        returncode,
        stdout,
        stderr,
        "Run it directly to see the error:\n"
        "  rocprofv3 --kernel-trace -d /tmp/rocprofv3-out -- /bin/true",
    )
    remediation += "\nPartial output was kept in {}".format(output_dir)
    return make_fail(detail + " (a GPU is present)", remediation, data)
