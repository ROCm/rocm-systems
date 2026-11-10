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

"""rocprofv3-collector writes rocprofv3's output files from a traced process.

    check_outputs.py <rocprofv3-collector> <collector-test-app> <test schema dir> <work dir>

The test program's "hip" mode makes 4 hipMemcpyAsync calls and 1 hipModuleGetFunction call
through a stand-in rocm_hip provider; rocprofv3-collector runs it, as rocprofv3 --collector
does, and must write the CSV, rocpd and JSON outputs for that process, and return its exit
status. Exits 77 (skipped) when no lttng-sessiond can be found.
"""

import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys

helper, app, schema_dir, work = sys.argv[1:5]


def have_session_daemon():
    prefix = os.environ.get("ROCPROFILER_COLLECTOR_LTTNG_PREFIX")
    if prefix:
        return os.access(os.path.join(prefix, "bin", "lttng-sessiond"), os.X_OK)
    return shutil.which("lttng-sessiond") is not None


if not have_session_daemon():
    print("lttng-sessiond not found (set ROCPROFILER_COLLECTOR_LTTNG_PREFIX): skipped")
    sys.exit(77)

failures = []


def expect(condition, what):
    if not condition:
        failures.append(what)
        print(f"FAILED: {what}")


def run(name, extra_env, args):
    out = os.path.join(work, name)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    env = dict(os.environ)
    for key in [k for k in env if k.startswith("ROCPROF_")]:
        del env[key]
    env.update(
        {
            "ROCPROF_OUTPUT_PATH": out,
            "ROCPROF_OUTPUT_FILE_NAME": "run",
            "ROCPROF_OUTPUT_FORMAT": "csv,json,rocpd",
            "ROCPROFILER_COLLECTOR_SCHEMA_DIR": schema_dir,
        }
    )
    env.update(extra_env)
    result = subprocess.run([helper, "--", app] + args, env=env)
    return out, result.returncode


# The HIP runtime API: every format, the application's exit status, no capture left behind.
out, status = run("runtime", {"ROCPROF_HIP_RUNTIME_API_TRACE": "1"}, ["4", "3", "hip"])
expect(status == 3, f"exit status of the application (3), got {status}")
expect(
    not [f for f in os.listdir(out) if f.startswith(".rocprofv3-collector")],
    "the capture directory is removed",
)

with open(os.path.join(out, "run_hip_api_trace.csv")) as f:
    rows = list(csv.DictReader(f))
expect(len(rows) == 5, f"5 rows in run_hip_api_trace.csv, got {len(rows)}")
expect(
    [r["Function"] for r in rows] == ["hipMemcpyAsync"] * 4 + ["hipModuleGetFunction"],
    "the calls in order",
)
expect({r["Domain"] for r in rows} == {"HIP_RUNTIME_API_EXT"}, "the HIP runtime domain")
pids = {r["Process_Id"] for r in rows}
expect(len(pids) == 1 and "0" not in pids, f"one traced process, got {pids}")
expect(
    sorted(int(r["Correlation_Id"]) for r in rows) == [1, 2, 3, 4, 5],
    "correlation ids 1..5",
)
expect(
    all(int(r["Start_Timestamp"]) <= int(r["End_Timestamp"]) for r in rows),
    "start <= end",
)
expect(os.path.isfile(os.path.join(out, "run_agent_info.csv")), "run_agent_info.csv")

db = sqlite3.connect(os.path.join(out, "run_results.db"))
args = db.execute(
    "select s.string, a.position, a.type, a.name, a.value from rocpd_region r "
    "join rocpd_string s on s.id = r.name_id "
    "join rocpd_arg a on a.event_id = r.event_id order by r.start, a.position"
).fetchall()
kinds = [a[4] for a in args if a[0] == "hipMemcpyAsync" and a[3] == "kind"]
expect(
    kinds == ["HostToHost", "HostToDevice", "DeviceToHost", "DeviceToDevice"],
    f"hipMemcpyAsync kind arguments, got {kinds}",
)
expect(
    ("hipModuleGetFunction", 2, "char const*", "kname", "kernel_name") in args,
    "the hipModuleGetFunction kname argument",
)
processes = db.execute("select pid, command from rocpd_info_process").fetchall()
expect(
    len(processes) == 1 and str(processes[0][0]) in pids and "hip" in processes[0][1],
    f"the traced process and its command line, got {processes}",
)

with open(os.path.join(out, "run_results.json")) as f:
    doc = json.load(f)
records = doc["rocprofiler-sdk-tool"][0]["buffer_records"]["hip_api"]
expect(len(records) == 5, f"5 hip_api records in the JSON output, got {len(records)}")

# The HIP compiler API only: the runtime calls are not written.
out, status = run("compiler", {"ROCPROF_HIP_COMPILER_API_TRACE": "1"}, ["4", "0", "hip"])
expect(status == 0, f"exit status 0, got {status}")
expect(
    not os.path.exists(os.path.join(out, "run_hip_api_trace.csv")),
    "no runtime calls without ROCPROF_HIP_RUNTIME_API_TRACE",
)

# A program that cannot be started: the shell's status for it, and no output.
missing = os.path.join(work, "missing")
shutil.rmtree(missing, ignore_errors=True)
result = subprocess.run(
    [helper, "--", os.path.join(work, "no-such-program")],
    env=dict(os.environ, ROCPROF_OUTPUT_PATH=missing, ROCPROF_HIP_RUNTIME_API_TRACE="1"),
)
expect(
    result.returncode == 127,
    f"exit status 127 for a missing program, got {result.returncode}",
)

if failures:
    sys.exit(1)
print("rocprofv3-collector outputs: all checks passed")
