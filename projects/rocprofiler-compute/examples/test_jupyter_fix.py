#!/usr/bin/env python3
"""Test script to verify Jupyter module fixes."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test import
try:
    import rocprof_compute_jupyter as rc

    print("✓ Successfully imported rocprof_compute_jupyter")

    # Check if the module has the expected functions
    assert hasattr(rc, "open"), "Missing 'open' function"
    assert hasattr(rc, "analysis"), "Missing 'analysis' function"
    assert hasattr(rc, "get_dataframe"), "Missing 'get_dataframe' function"
    assert hasattr(rc, "list_tables"), "Missing 'list_tables' function"
    print("✓ All expected functions are present")

    # Test with a sample workload directory (if provided)
    if len(sys.argv) > 1:
        perf_data_dir = sys.argv[1]
        print(f"\nTesting with workload: {perf_data_dir}")
        rc.open(perf_data_dir)
        print("✓ Successfully loaded performance data")
        print("\nAvailable tables:")
        rc.list_tables()
    else:
        print("\nTo test with actual data, run:")
        print("  python examples/test_jupyter_fix.py /path/to/workload")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
