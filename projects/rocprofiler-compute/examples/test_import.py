#!/usr/bin/env python3
"""
Simple test script to verify the import path for rocprof_compute_jupyter
"""

import sys
from pathlib import Path

# Get the absolute path to the src directory
current_dir = Path(__file__).parent.resolve()
src_dir = current_dir.parent / 'src'

print(f"Current directory: {current_dir}")
print(f"Source directory: {src_dir}")
print(f"Source directory exists: {src_dir.exists()}")

# Add to path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

print(f"\nPython path:")
for p in sys.path[:5]:
    print(f"  {p}")

# Try to import
try:
    import rocprof_compute_jupyter as rc
    print(f"\n✓ Successfully imported rocprof_compute_jupyter")
    print(f"  Module location: {rc.__file__}")
    print(f"  Available functions: {[x for x in dir(rc) if not x.startswith('_')]}")
except ImportError as e:
    print(f"\n✗ Failed to import: {e}")
    
    # Check if file exists
    module_file = src_dir / 'rocprof_compute_jupyter.py'
    print(f"\nModule file exists: {module_file.exists()}")
    if module_file.exists():
        print(f"Module file path: {module_file}")
