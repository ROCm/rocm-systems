#!/usr/bin/env python3

import os
import sys
import glob
import pytest


def get_pid_databases(output_dir):
    """Find all .rocpd database files with PIDs in the output directory."""
    pattern = os.path.join(output_dir, "fork_out_*_results.db")
    db_files = glob.glob(pattern)
    return db_files


def test_multiple_pid_databases(output_dir):
    """
    Test that multiple database files are created for different PIDs.

    This validates the fix where using -o flag with %pid% substitution
    creates separate database files for parent and child processes.
    """
    db_files = get_pid_databases(output_dir)

    # We expect at least 2 databases (parent + child)
    assert len(db_files) >= 2, (
        f"Expected at least 2 database files (parent + child), "
        f"but found {len(db_files)}: {db_files}"
    )

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
