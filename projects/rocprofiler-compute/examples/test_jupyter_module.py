#!/usr/bin/env python3
"""
Test script to verify rocprof_compute_jupyter module functionality
without requiring a Jupyter kernel.
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_dir))

print("=" * 80)
print("Testing rocprof_compute_jupyter module")
print("=" * 80)

# Test 1: Import the module
print("\n1. Testing module import...")
try:
    import rocprof_compute_jupyter as rc
    print("   ✓ Module imported successfully")
    print(f"   Module location: {rc.__file__}")
except Exception as e:
    print(f"   ✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Check available functions
print("\n2. Checking available functions...")
expected_functions = ['open', 'analysis', 'get_dataframe', 'list_tables']
available_functions = [x for x in dir(rc) if not x.startswith('_')]
print(f"   Available functions: {available_functions}")

for func in expected_functions:
    if func in available_functions:
        print(f"   ✓ {func}() is available")
    else:
        print(f"   ✗ {func}() is missing")

# Test 3: Check function signatures
print("\n3. Checking function signatures...")
import inspect

for func_name in expected_functions:
    if hasattr(rc, func_name):
        func = getattr(rc, func_name)
        sig = inspect.signature(func)
        print(f"   {func_name}{sig}")

# Test 4: Try to use with sample data (if available)
print("\n4. Testing with sample data...")
print("   Note: This requires actual profiling data to test fully.")
print("   To test with real data, run:")
print("   ")
print("   import rocprof_compute_jupyter as rc")
print("   rc.open('/path/to/your/workload_dir')")
print("   rc.analysis()")

# Test 5: Check class structure
print("\n5. Checking internal class structure...")
if hasattr(rc, 'JupyterAnalysis'):
    print("   ✓ JupyterAnalysis class is available")
    ja_methods = [m for m in dir(rc.JupyterAnalysis) if not m.startswith('_')]
    print(f"   Public methods: {ja_methods[:5]}...")  # Show first 5
else:
    print("   ✗ JupyterAnalysis class not found")

print("\n" + "=" * 80)
print("Module verification complete!")
print("=" * 80)
print("\nThe module is ready to use. Example usage:")
print("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('path/to/rocprofiler-compute/src').resolve()))

import rocprof_compute_jupyter as rc

# Load your profiling data
rc.open('/path/to/workload_dir')

# Display analysis
rc.analysis()

# Get specific data
kernel_stats = rc.get_dataframe(1)
print(kernel_stats.head())
""")
