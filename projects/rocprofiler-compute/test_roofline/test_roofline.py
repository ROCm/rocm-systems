#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
##############################################################################

"""
Test module for calc_roofline_data function from analysis_db.py.

This module demonstrates how to call the calc_roofline_data method from
db_analysis class with properly structured dummy data.

The calc_roofline_data function calculates roofline metrics including:
- total_flops: Total floating point operations (in GFLOPs)
- l1_cache_data: Arithmetic intensity at L1 cache level (FLOPS/byte)
- l2_cache_data: Arithmetic intensity at L2 cache level (FLOPS/byte)
- hbm_cache_data: Arithmetic intensity at HBM/main memory level (FLOPS/byte)

Input Requirements:
-------------------
The function requires a db_analysis instance with:
1. self._pmc_df_per_workload: Dictionary mapping workload paths to PMC DataFrames
2. self._runs: Dictionary mapping workload paths to Workload objects
3. self._arch_configs: Dictionary mapping GPU architectures to ArchConfig objects
4. self.get_args(): Method returning args with max_stat_num

Output Format:
--------------
Returns a dictionary mapping workload paths to DataFrames:
{
  '/path/to/workload': pd.DataFrame({
    'kernel_name': ['kernel_1', 'kernel_2'],
    'total_flops': [1000.0, 2000.0],
    'l1_cache_data': [33.3, 50.0],
    'l2_cache_data': [50.0, 75.0],
    'hbm_cache_data': [100.0, 150.0]
  })
}
"""

import sys
from pathlib import Path

# Add src directory to Python path (parent directory contains src/)
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pandas as pd

from rocprof_compute_analyze.analysis_db import db_analysis
from utils.schema import ArchConfig, Workload


class TestDbAnalysis:
    """Minimal test class that mimics db_analysis structure."""

    def __init__(self):
        self._pmc_df_per_workload = {}
        self._runs = {}
        self._arch_configs = {}
        self._args = None

    def get_args(self):
        """Return args object."""
        return self._args

    def calc_roofline_data(self):
        """Call the actual calc_roofline_data method from db_analysis."""
        return db_analysis.calc_roofline_data(self)


class Args:
    """Simple args class to hold configuration."""

    def __init__(self, max_stat_num=10):
        self.max_stat_num = max_stat_num


def create_test_instance():
    """Create a test instance with minimal test data."""

    # Create test instance
    test_instance = TestDbAnalysis()

    # Create dummy workload path
    workload_path = "/tmp/dummy_workload"

    # 1. Create dummy PMC DataFrame
    pmc_data = pd.DataFrame({
        "Kernel_Name": ["test_kernel_1", "test_kernel_2", "test_kernel_1"],
        "Start_Timestamp": [0, 1000, 2000],
        "End_Timestamp": [1000, 3000, 3000],
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
    })

    # 2. Create dummy Workload with sys_info
    workload = Workload()
    sys_info_data = {
        "gpu_arch": ["gfx90a"],
        "gpu_model": ["MI210"],
        "version": ["1"],
        "timestamp": ["2024-01-01"],
        "ip_blocks": ["SQ"],
        "num_xcd": [1],
        "max_waves_per_cu": [32],
        "max_sclk": [1700],
        "max_mclk": [1600],
        "num_hbm_channels": [32],
        "cu_per_gpu": [104],
        "se_per_gpu": [8],
        "pipes_per_gpu": [4],
        "simd_per_cu": [4],
        "sqc_per_gpu": [32],
        "lds_banks_per_cu": [32],
        "cur_sclk": [1700],
        "cur_mclk": [1600],
        "wave_size": [64],
        "total_l2_chan": [32],
        "l2_banks": [32],
    }
    workload.sys_info = pd.DataFrame(sys_info_data)

    # 3. Create dummy ArchConfig with roofline table (ID 402)
    arch_config = ArchConfig()
    roofline_table_data = {
        "Metric": [
            "Performance (GFLOPs)",
            "AI L1",
            "AI L2",
            "AI HBM"
        ],
        "Value": [
            "1000.0",  # Dummy expression for total_flops
            "33.3",    # Dummy expression for AI L1
            "50.0",    # Dummy expression for AI L2
            "100.0",   # Dummy expression for AI HBM
        ],
    }
    arch_config.dfs[402] = pd.DataFrame(roofline_table_data)

    # 4. Create args
    args = Args(max_stat_num=10)

    # 5. Set up the instance attributes
    test_instance._pmc_df_per_workload = {workload_path: pmc_data}
    test_instance._runs = {workload_path: workload}
    test_instance._arch_configs = {"gfx90a": arch_config}
    test_instance._args = args

    return test_instance, workload_path


def main():
    """Main function to test calc_roofline_data."""
    print("=" * 80)
    print("Testing calc_roofline_data function")
    print("=" * 80)

    # Create test instance
    print("\n1. Creating test db_analysis instance...")
    test_instance, workload_path = create_test_instance()

    # Call the function
    print("\n2. Calling calc_roofline_data...")

    try:
        result = test_instance.calc_roofline_data()

        print("\n" + "=" * 80)
        print("SUCCESS: calc_roofline_data returned successfully")
        print("=" * 80)

        print("\nResult structure:")
        print(f"  Type: {type(result)}")
        print(f"  Keys (workload paths): {list(result.keys())}")

        if result:
            print("\nResult contents:")
            for path, df in result.items():
                print(f"\nWorkload: {path}")
                print(f"  DataFrame shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()}")
                print("\nData:")
                print(df.to_string(index=False))

        return result

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: calc_roofline_data failed with exception:")
        print("=" * 80)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result is not None else 1)
