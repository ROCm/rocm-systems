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

"""Companion tool checks: are the sibling CLI tools installed and runnable?"""

from __future__ import absolute_import

from rocprofv3.doctor_diagnose import explain_failure
from rocprofv3.doctor_env import RUN_TIMEOUT
from rocprofv3.doctor_layout import reinstall_hint
from rocprofv3.doctor_registry import register
from rocprofv3.doctor_result import (
    make_fail,
    make_pass,
    make_warn,
    PROBE_PROCESS,
    SEV_ERROR,
    SEV_WARNING,
)


def resolve_tool(accessor, name):
    """Locate companion tool ``name``; return ``(path_or_None, searched)``.

    The ROCm root's bin/ is preferred so the report describes the installation
    under inspection, but PATH is consulted as well: distro packages symlink
    these tools into /usr/bin, and a non-default prefix may have no bin/ under
    the ROCm root at all. Failing on either of those would report a broken
    installation on a perfectly healthy machine.
    """
    searched = []
    candidate = accessor.join(accessor.rocm_root, "bin", name)
    searched.append(candidate)
    if accessor.path_exists(candidate):
        return (candidate, searched)

    found, path_searched = accessor.which(name)
    searched.extend(path_searched)
    return (found, searched)


def tool_command(accessor, path, args):
    """argv that runs companion tool ``path`` with ``args`` as a user would.

    The tool is executed directly, so its shebang chooses the interpreter --
    exactly as when the user types its name. Running it under this doctor's
    own interpreter instead would hide the most common breakage of a Python
    tool: a shebang interpreter that is missing or cannot import its modules.
    Native launchers (TheRock venv shims) run the same way.
    """
    return [path] + list(args)


def _tool_present(accessor, name, purpose, remediation, failure_factory):
    path, searched = resolve_tool(accessor, name)
    data = {"tool": name, "searched": searched, "purpose": purpose}

    if path is None:
        return failure_factory(
            "{} was not found under {}/bin or on PATH".format(name, accessor.rocm_root),
            remediation,
            data,
        )

    data["path"] = path
    if not accessor.path_executable(path):
        return failure_factory(
            "{} exists but is not executable".format(path),
            "sudo chmod +x {}".format(path),
            data,
        )
    return make_pass("{} present and executable".format(path), "", data)


@register(
    id="tools.rocprofv3",
    group="tools",
    title="rocprofv3 present and executable",
    severity=SEV_ERROR,
    depends=["install.rocm-root"],
    order=90,
)
def check_rocprofv3(accessor):
    return _tool_present(
        accessor,
        "rocprofv3",
        "the main profiler CLI",
        reinstall_hint(accessor),
        make_fail,
    )


@register(
    id="tools.rocprofv3-avail",
    group="tools",
    title="rocprofv3-avail present and executable",
    severity=SEV_ERROR,
    depends=["install.rocm-root"],
    order=91,
)
def check_rocprofv3_avail(accessor):
    return _tool_present(
        accessor,
        "rocprofv3-avail",
        "counter and agent enumeration",
        reinstall_hint(accessor),
        make_fail,
    )


@register(
    id="tools.rocprof-attach",
    group="tools",
    title="rocprof-attach present and executable",
    severity=SEV_WARNING,
    depends=["install.rocm-root"],
    order=92,
)
def check_rocprof_attach(accessor):
    return _tool_present(
        accessor,
        "rocprof-attach",
        "attaching to an already-running process",
        "Only needed for `rocprofv3 --pid`; safe to ignore otherwise.",
        make_warn,
    )


@register(
    id="tools.rocpd",
    group="tools",
    title="rocpd present and executable",
    severity=SEV_WARNING,
    depends=["install.rocm-root"],
    order=93,
)
def check_rocpd(accessor):
    return _tool_present(
        accessor,
        "rocpd",
        "converting the rocpd output database to other formats",
        "Install the rocpd component of rocprofiler-sdk.",
        make_warn,
    )


@register(
    id="tools.rocprofv3-avail-runnable",
    group="tools",
    title="rocprofv3-avail --help runs successfully",
    severity=SEV_WARNING,
    depends=["tools.rocprofv3-avail"],
    order=94,
    probe=PROBE_PROCESS,
)
def check_rocprofv3_avail_runnable(accessor):
    path, _ = resolve_tool(accessor, "rocprofv3-avail")
    if path is None:
        return make_warn(
            "rocprofv3-avail was not found; cannot verify that it runs",
            reinstall_hint(accessor),
            {"tool": "rocprofv3-avail"},
        )

    returncode, stdout, stderr = accessor.run(
        tool_command(accessor, path, ["--help"]), timeout=10
    )
    data = {"tool": path, "returncode": returncode, "stderr_tail": stderr.strip()[-400:]}

    if returncode == RUN_TIMEOUT:
        return make_warn(
            "rocprofv3-avail --help timed out after 10 seconds",
            "Run it directly to investigate:\n  rocprofv3-avail --help",
            data,
        )
    if returncode != 0:
        detail, remediation, data["diagnoses"] = explain_failure(
            accessor,
            "rocprofv3-avail --help",
            returncode,
            stdout,
            stderr,
            "Run it directly to see the error:\n  rocprofv3-avail --help",
        )
        return make_warn(detail, remediation, data)
    data["stdout_bytes"] = len(stdout)
    return make_pass("rocprofv3-avail --help exited 0", "", data)
