#!/usr/bin/env python3

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


def test_kernel_traces_in_csv(kernel_trace_files, kernel_trace_data):
    """
    Test that kernel traces exist in CSV files for both parent and child processes.

    This validates that CSV output is generated correctly for each process
    and contains the expected vectorAdd kernel.
    """
    # We expect at least 2 kernel trace CSV files (parent + child)
    assert len(kernel_trace_files) >= 2, (
        f"Expected at least 2 kernel trace CSV files (parent + child), "
        f"but found {len(kernel_trace_files)}: {kernel_trace_files}"
    )

    # Check each CSV file for kernel traces
    for csv_file, data in kernel_trace_data.items():
        print(f"\nValidating kernel traces in: {csv_file}")

        # Verify the CSV file is not empty
        assert len(data) > 0, f"Kernel trace CSV file {csv_file} is empty"

        # Look for vectorAdd kernel in the traces
        kernel_names = [row.get("Kernel_Name", "") for row in data]
        vectorAdd_found = any("vectorAdd" in name for name in kernel_names)

        assert vectorAdd_found, (
            f"Expected to find 'vectorAdd' kernel in {csv_file}, "
            f"but found kernels: {kernel_names}"
        )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
