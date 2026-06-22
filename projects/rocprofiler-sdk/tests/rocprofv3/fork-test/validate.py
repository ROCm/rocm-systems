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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import sys
import glob
import pytest


def get_pid_databases(output_dir):
    """Find rocpd database files, split into the clean parent file and PID-suffixed children."""
    parent_pattern = os.path.join(output_dir, "fork_out_results.db")
    child_pattern = os.path.join(output_dir, "fork_out_*_results.db")
    parent_files = glob.glob(parent_pattern)
    child_files = [f for f in glob.glob(child_pattern) if f not in parent_files]
    return parent_files, child_files


def test_multiple_pid_databases(output_dir):
    """
    Test that the parent keeps the clean output name and each forked/spawned
    child writes its own PID-suffixed database, so no process overwrites another.
    """
    assert (
        output_dir is not None
    ), "output_dir must be provided via --output-dir when running this test."

    parent_files, child_files = get_pid_databases(output_dir)

    # Parent keeps the clean name; children are PID-suffixed.
    assert len(parent_files) == 1, (
        f"Expected exactly 1 clean parent database (fork_out_results.db), "
        f"but found {len(parent_files)}: {parent_files}"
    )
    assert len(child_files) >= 1, (
        f"Expected at least 1 PID-suffixed child database, "
        f"but found {len(child_files)}: {child_files}"
    )
    db_files = parent_files + child_files

    # Verify all database files exist and are non-empty
    for db_file in db_files:
        assert os.path.exists(db_file), f"Database file {db_file} does not exist"
        assert os.path.getsize(db_file) > 0, f"Database file {db_file} is empty"


def _is_parent_csv(path):
    """The parent keeps the clean name (fork_out_kernel_trace.csv); children are
    PID-suffixed (fork_out_<pid>_kernel_trace.csv)."""
    return os.path.basename(path) == "fork_out_kernel_trace.csv"


def test_kernel_traces_in_csv(kernel_trace_files, kernel_trace_data):
    """
    Test that kernel traces are captured for both processes, and that each process'
    CSV contains its OWN kernel -- proving the parent and the fork+exec'd child wrote
    to separate files rather than overwriting each other.

    The parent runs the vectorAdd kernel directly; the child execs simple-transpose,
    whose kernel is matrixTranspose. So the parent CSV must contain vectorAdd and the
    child CSV must contain matrixTranspose.
    """
    # We expect at least 2 kernel trace CSV files (parent + child)
    assert len(kernel_trace_files) >= 2, (
        f"Expected at least 2 kernel trace CSV files (parent + child), "
        f"but found {len(kernel_trace_files)}: {kernel_trace_files}"
    )

    parent_csvs = [f for f in kernel_trace_files if _is_parent_csv(f)]
    child_csvs = [f for f in kernel_trace_files if not _is_parent_csv(f)]

    assert len(parent_csvs) == 1, (
        f"Expected exactly 1 parent kernel trace CSV (fork_out_kernel_trace.csv), "
        f"but found {len(parent_csvs)}: {parent_csvs}"
    )
    assert len(child_csvs) >= 1, (
        f"Expected at least 1 PID-suffixed child kernel trace CSV, "
        f"but found {len(child_csvs)}: {child_csvs}"
    )

    def _kernel_names(csv_file):
        data = kernel_trace_data[csv_file]
        assert len(data) > 0, f"Kernel trace CSV file {csv_file} is empty"
        return [row.get("Kernel_Name", "") for row in data]

    # Parent ran vectorAdd directly.
    parent_csv = parent_csvs[0]
    parent_kernels = _kernel_names(parent_csv)
    assert any("vectorAdd" in name for name in parent_kernels), (
        f"Expected to find 'vectorAdd' kernel in parent CSV {parent_csv}, "
        f"but found kernels: {parent_kernels}"
    )

    # Child exec'd simple-transpose, which runs matrixTranspose.
    for child_csv in child_csvs:
        child_kernels = _kernel_names(child_csv)
        assert any("matrixTranspose" in name for name in child_kernels), (
            f"Expected to find 'matrixTranspose' kernel in child CSV {child_csv}, "
            f"but found kernels: {child_kernels}"
        )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
