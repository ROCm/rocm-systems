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

"""Counter-collection permission checks.

# SYNC: source/lib/rocprofiler-sdk/counters/ioctl.cpp (EBUSY/EPERM/EINVAL
# handling) -- the remediation strings below describe the conditions that path
# reports. If those error conditions change, revisit these messages.
"""

from __future__ import absolute_import

import re

from rocprofv3.doctor_checks_tools import resolve_tool, tool_command
from rocprofv3.doctor_diagnose import explain_failure
from rocprofv3.doctor_env import CAP_PERFMON_BIT, CAP_SYS_ADMIN_BIT, RUN_TIMEOUT
from rocprofv3.doctor_registry import register
from rocprofv3.doctor_result import (
    make_fail,
    make_pass,
    make_warn,
    DOCTOR_COMMAND,
    DOCTOR_FLAG,
    PROBE_GPU,
    SEV_INFO,
    SEV_WARNING,
)

# `rocprofv3-avail info` prints one "Counter_Name        :<TAB><name>" line per
# counter; the key is left-padded to 20 columns by build_counter_string().
COUNTER_NAME_RE = re.compile(r"^\s*Counter_Name\s*:")

# process names that indicate another profiler may already hold the device
_PROFILER_MARKERS = (
    "rocprofv3",
    "rocprofv2",
    "rocprof",
    "roctracer",
    "rocprofiler",
    "omniperf",
    "rocprofiler-compute",
)


# NOTE: perf_event_paranoid is deliberately not checked. rocprofiler-sdk does
# not use perf events; GPU counters go through KFD, whose profiler lock is
# gated on CAP_PERFMON alone (counters/ioctl.cpp), so the sysctl has no bearing
# on rocprofv3 and recommending a change to it would be wrong.
#
# Severity is SEV_INFO: without the capability, dispatch counter collection
# still works -- only exclusive device locking and device-wide collection are
# affected -- so its absence must not put a warning on every normal account.
@register(
    id="counters.cap-perfmon",
    group="counters",
    title="CAP_PERFMON or CAP_SYS_ADMIN available",
    severity=SEV_INFO,
    depends=["driver.kfd-readable"],
    order=41,
)
def check_cap_perfmon(accessor):
    cap_eff = accessor.get_cap_eff()
    data = {}
    if cap_eff is None:
        return make_warn(
            "could not read the effective capability mask from /proc/self/status",
            "",
            data,
        )

    has_perfmon = bool(cap_eff & (1 << CAP_PERFMON_BIT))
    has_sys_admin = bool(cap_eff & (1 << CAP_SYS_ADMIN_BIT))
    data["cap_eff"] = "0x{:016x}".format(cap_eff)
    data["cap_perfmon"] = has_perfmon
    data["cap_sys_admin"] = has_sys_admin

    if has_perfmon or has_sys_admin:
        held = "CAP_PERFMON" if has_perfmon else "CAP_SYS_ADMIN"
        return make_pass("process holds {}".format(held), "", data)

    # SYNC: counters/ioctl.cpp -- the KFD profiler lock fails with EPERM
    # without the capability, and the SDK then warns that "PMC Counters may be
    # inaccurate and System Counter Collection will be degraded".
    return make_pass(
        "neither CAP_PERFMON nor CAP_SYS_ADMIN is held, which is normal for an "
        "unprivileged account: dispatch counter collection works, but the GPU "
        "cannot be locked for exclusive profiling (values may be inaccurate if "
        "another process profiles the same GPU) and device-wide counter "
        "collection is degraded",
        "Only needed for exact counter values under contention or for\n"
        "device-wide collection. Run with CAP_PERFMON, for example as root:\n"
        "  sudo -E rocprofv3 ...\n"
        "Setting file capabilities on rocprofv3 has no effect: it is a Python\n"
        "script, and Linux ignores file capabilities on scripts.",
        data,
    )


def is_doctor_run(argv, accessor):
    """True when ``argv`` is a doctor run rather than a profiling session.

    Recognizes the doctor script itself, and ``rocprofv3 --doctor`` with the
    flag among the options that follow that rocprofv3 -- up to its ``--``,
    after which the arguments belong to the profiled application, so
    ``rocprofv3 ... -- app --doctor`` is a real profiling session. Scoping the
    flag to the rocprofv3 word, not to the whole line, also handles shells
    whose command text has unrelated ``--`` tokens before it. Matching is
    exact on purpose: a profiler whose output directory is called
    "doctor-results" must still be reported.
    """
    seen_separator = False
    for index, argument in enumerate(argv):
        name = accessor.basename(argument)
        if name in ("rocprofv3-doctor", "rocprofv3-doctor.py") and not seen_separator:
            return True
        if name in ("rocprofv3", "rocprofv3.py"):
            for option in argv[index + 1 :]:
                if option == "--":
                    break
                if option == DOCTOR_FLAG:
                    return True
        if argument == "--":
            seen_separator = True
    return False


@register(
    id="counters.no-profiler-lock",
    group="counters",
    title="No other profiler appears to be running",
    severity=SEV_WARNING,
    depends=["driver.kfd-readable"],
    order=42,
)
def check_no_profiler_lock(accessor):
    # Heuristic only: scanning /proc for profiler-looking processes. The KFD
    # only permits one profiling session per device, but probing that directly
    # would mean taking the lock ourselves, which is too intrusive for a
    # diagnostic. Report warn, never fail.
    own_pid = "{}".format(accessor.getpid())
    others = []
    for cmdline_path in accessor.glob("/proc/[0-9]*/cmdline"):
        pid = cmdline_path.split("/")[2]
        if pid == own_pid:
            # identify ourselves by pid, not by matching "doctor" in the text:
            # a name-based filter would also hide a real profiler whose command
            # line happens to mention a file or directory called "doctor"
            continue

        content = accessor.read_file(cmdline_path)
        if not content:
            continue
        cmdline = content.replace("\x00", " ").strip()
        if not cmdline:
            continue

        # Scan every argument, not just argv[0]. rocprofv3 and rocprofv3-avail
        # are #!/usr/bin/env python3 scripts, so argv[0] is the interpreter and
        # the profiler name only ever appears in a later argument -- checking
        # argv[0] alone would miss the exact case this check exists to catch.
        arguments = cmdline.split(" ")
        # Checked on both the real argv and the whitespace-split command line:
        # a shell such as `bash -c "rocprofv3 --doctor ..."` (our own parent,
        # in CI) carries its whole command as one argument. Skipping a shell
        # that wraps a real profiler loses nothing -- the profiler is its own
        # process and is scanned on its own.
        argv = [arg for arg in content.split("\x00") if arg]
        if is_doctor_run(argv, accessor) or is_doctor_run(arguments, accessor):
            # another doctor run, or the shell that launched this one: it may
            # inspect the GPU, but it holds no profiling lock
            continue
        matched = None
        for argument in arguments:
            basename = accessor.basename(argument)
            for marker in _PROFILER_MARKERS:
                if marker in basename:
                    matched = marker
                    break
            if matched:
                break

        if matched:
            others.append("{} ({})".format(pid, cmdline[:80]))

    data = {"candidate_processes": others, "own_pid": own_pid}
    if others:
        return make_warn(
            "other profiler-like process(es) are running: {}. Only one profiling "
            "session may hold a GPU at a time.".format("; ".join(others)),
            "Stop the other profiling session before collecting counters.",
            data,
        )
    return make_pass("no other profiler process detected", "", data)


@register(
    id="counters.avail-enumeration",
    group="counters",
    title="rocprofv3-avail enumerates counters",
    severity=SEV_INFO,
    depends=["install.avail-library", "driver.gpu-topology"],
    order=43,
    # rocprofv3-avail initializes the HSA runtime and opens every GPU agent
    probe=PROBE_GPU,
)
def check_avail_enumeration(accessor):
    # `info --pmc`, not `list --pmc`: the list path prints a bare column grid of
    # names, whereas info prints one "Counter_Name        :<TAB><name>" line per
    # counter, which is what COUNTER_NAME_RE below matches.
    # SYNC: source/lib/python/rocprofv3/avail.py build_counter_string() and
    # source/bin/rocprofv3-avail.py info_counters().
    avail, searched = resolve_tool(accessor, "rocprofv3-avail")
    data = {"tool": avail, "searched": searched}

    if avail is None:
        return make_warn(
            "rocprofv3-avail was not found; cannot verify counter enumeration",
            "Reinstall the rocprofiler-sdk package.",
            data,
        )

    returncode, stdout, stderr = accessor.run(
        tool_command(accessor, avail, ["info", "--pmc"]), timeout=15
    )
    data["returncode"] = returncode
    data["stderr_tail"] = stderr.strip()[-400:]

    if returncode == RUN_TIMEOUT:
        return make_fail(
            "rocprofv3-avail timed out after 15 seconds -- the driver may be hung",
            "Check dmesg for amdgpu errors:\n  sudo dmesg | grep -i amdgpu\n"
            "Skip this check with: " + DOCTOR_COMMAND + " --skip "
            "counters.avail-enumeration",
            data,
        )
    if returncode != 0:
        detail, remediation, data["diagnoses"] = explain_failure(
            accessor,
            "rocprofv3-avail info --pmc",
            returncode,
            stdout,
            stderr,
            "Run it directly to see the error:\n  rocprofv3-avail info --pmc",
        )
        return make_warn(detail, remediation, data)

    counter_lines = 0
    for line in stdout.splitlines():
        if COUNTER_NAME_RE.match(line):
            counter_lines += 1
    data["counter_entries"] = counter_lines

    if counter_lines == 0:
        return make_warn(
            "rocprofv3-avail succeeded but listed no counters",
            "Run it directly to inspect the output:\n  rocprofv3-avail info --pmc",
            data,
        )
    return make_pass(
        "rocprofv3-avail listed {} counter(s)".format(counter_lines), "", data
    )
