#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import csv
import glob
import os
import re
import sqlite3

import pytest

# The output files of a process: <pid>_<suffix>, as rocprofv3 -o %pid% writes them.
_POINTER = re.compile(r"0x[0-9a-fA-F]+")


def pytest_addoption(parser):
    parser.addoption(
        "--in-process-dir",
        action="store",
        help="Output directory of rocprofv3 --hip-trace",
    )
    parser.addoption(
        "--collector-dir",
        action="store",
        help="Output directory of rocprofv3 --collector --hip-trace",
    )


class TracedProcess:
    """The output files of one traced process."""

    def __init__(self, directory, pid):
        self.directory = directory
        self.pid = pid
        self.files = sorted(
            os.path.basename(f)[len(pid) + 1 :]
            for f in glob.glob(os.path.join(directory, f"{pid}_*"))
        )

    def path(self, suffix):
        return os.path.join(self.directory, f"{self.pid}_{suffix}")

    def csv_rows(self, suffix):
        with open(self.path(suffix)) as f:
            reader = csv.reader(f)
            header = next(reader)
            return header, [dict(zip(header, row)) for row in reader]

    def calls(self):
        """{thread: [(function, [(position, type, name, value)])]} in start order, from rocpd.
        Pointer values are masked: the runs being compared are separate."""
        db = sqlite3.connect(self.path("results.db"))
        args = {}
        for event_id, pos, typ, name, value in db.execute(
            "select event_id, position, type, name, value from rocpd_arg order by event_id, position"
        ):
            masked = "<pointer>" if value and _POINTER.fullmatch(value) else value
            args.setdefault(event_id, []).append((pos, typ, name, masked))
        threads = {}
        for tid, name, event_id in db.execute(
            "select r.tid, s.string, r.event_id from rocpd_region r "
            "join rocpd_string s on s.id = r.name_id order by r.start, r.id"
        ):
            threads.setdefault(tid, []).append((name, args.get(event_id, [])))
        return threads


def _processes(directory):
    pids = sorted(
        {
            os.path.basename(f).split("_", 1)[0]
            for f in glob.glob(os.path.join(directory, "*_hip_api_trace.csv"))
        }
    )
    return [TracedProcess(directory, pid) for pid in pids]


@pytest.fixture
def in_process(request):
    directory = request.config.getoption("--in-process-dir")
    processes = _processes(directory)
    if not processes:
        pytest.fail(f"no output from rocprofv3 --hip-trace in {directory}")
    return processes


@pytest.fixture
def collector(request):
    directory = request.config.getoption("--collector-dir")
    processes = _processes(directory)
    if not processes:
        pytest.skip(
            f"no output from rocprofv3 --collector in {directory}: its run was skipped "
            "(collector prerequisites missing)"
        )
    return processes
