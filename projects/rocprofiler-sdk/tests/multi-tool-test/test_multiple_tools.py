#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""
test_multiple_tools.py -- Integration test demonstrating multiple concurrent
rocprofiler-sdk tool libraries (kinetoesque and protonesque) controlled
independently via ctypes from Python.

Test flow:
    1. Import torchesque (which imports kinetoesque submodule)
    2. Initialize kinetoesque profiler, start profiling
    3. Launch trivial HIP kernels on 8 streams via torchesque
    4. Stop kinetoesque profiler
    5. Import tritonesque (which imports protonesque submodule)
    6. Initialize protonesque profiler, start profiling
    7. Launch trivial HIP kernels on 4 streams via tritonesque
    8. Stop protonesque profiler
    9. Finalize both tools and verify JSON output files
"""

import json
import os
import sys
import tempfile


def main():
    print("=== Multiple Anytime Tool Configuration Test ===\n")

    # Set up output files in a temp directory
    output_dir = os.environ.get(
        "TEST_OUTPUT_DIR", tempfile.mkdtemp(prefix="multi-tool-test-")
    )
    kinetoesque_output = os.path.join(output_dir, "kinetoesque-trace.json")
    protonesque_output = os.path.join(output_dir, "protonesque-trace.json")

    os.environ["KINETOESQUE_OUTPUT_FILE"] = kinetoesque_output
    os.environ["PROTONESQUE_OUTPUT_FILE"] = protonesque_output

    print(f"Output directory: {output_dir}")
    print(f"Kinetoesque output: {kinetoesque_output}")
    print(f"Protonesque output: {protonesque_output}")

    # ======================================================================
    # Phase 1: Import torchesque (imports kinetoesque submodule automatically)
    # ======================================================================
    print("\n--- Phase 1: Import torchesque ---")
    import torchesque

    print("  torchesque imported (kinetoesque submodule loaded)")

    # Initialize and start the kinetoesque profiler
    torchesque.kinetoesque.init()
    torchesque.kinetoesque.start_profiler()
    print("  kinetoesque profiler started")

    # ======================================================================
    # Phase 2: Launch HIP kernels on 8 streams via torchesque
    # ======================================================================
    print("\n--- Phase 2: Launch HIP kernels via torchesque (8 streams) ---")
    with torchesque.HipKernelRunner(num_streams=8) as runner:
        # Launch kernels multiple times
        for iteration in range(3):
            runner.launch_kernels()
            print(f"  Iteration {iteration + 1}: launched kernels on 8 streams")
        runner.synchronize()
        print("  All torchesque kernels synchronized")

    kinetoesque_count_after_torch = torchesque.kinetoesque.get_trace_count()
    print(
        f"  Kinetoesque traces after torchesque kernels: {kinetoesque_count_after_torch}"
    )

    # Stop kinetoesque briefly (demonstrates anytime start/stop)
    torchesque.kinetoesque.stop_profiler()
    print("  kinetoesque profiler stopped temporarily")

    # ======================================================================
    # Phase 3: Import tritonesque (imports protonesque submodule automatically)
    # ======================================================================
    print("\n--- Phase 3: Import tritonesque ---")
    import tritonesque

    print("  tritonesque imported (protonesque submodule loaded)")

    # Initialize and start the protonesque profiler
    tritonesque.protonesque.init()
    tritonesque.protonesque.start_profiler()
    print("  protonesque profiler started")

    # Also restart kinetoesque to show both can run concurrently
    torchesque.kinetoesque.start_profiler()
    print("  kinetoesque profiler restarted (both tools now active)")

    # ======================================================================
    # Phase 4: Launch HIP kernels on 4 streams via tritonesque
    # ======================================================================
    print("\n--- Phase 4: Launch HIP kernels via tritonesque (4 streams) ---")
    with tritonesque.HipKernelRunner(num_streams=4) as runner:
        for iteration in range(3):
            runner.launch_kernels()
            print(f"  Iteration {iteration + 1}: launched kernels on 4 streams")
        runner.synchronize()
        print("  All tritonesque kernels synchronized")

    # Both tools should have captured the tritonesque kernel traces
    kinetoesque_count_final = torchesque.kinetoesque.get_trace_count()
    protonesque_count_final = tritonesque.protonesque.get_trace_count()
    print(f"  Kinetoesque total traces: {kinetoesque_count_final}")
    print(f"  Protonesque total traces: {protonesque_count_final}")

    # ======================================================================
    # Phase 5: Stop both profilers and finalize
    # ======================================================================
    print("\n--- Phase 5: Stop and finalize both tools ---")
    torchesque.kinetoesque.stop_profiler()
    tritonesque.protonesque.stop_profiler()
    print("  Both profilers stopped")

    torchesque.kinetoesque.finalize()
    tritonesque.protonesque.finalize()
    print("  Both tools finalized")

    # ======================================================================
    # Phase 6: Validate output files
    # ======================================================================
    print("\n--- Phase 6: Validate output ---")
    errors = []

    # Validate kinetoesque output
    if not os.path.isfile(kinetoesque_output):
        errors.append(f"Kinetoesque output file not found: {kinetoesque_output}")
    else:
        with open(kinetoesque_output, "r") as f:
            data = json.load(f)
        traces = data.get("kinetoesque-traces", [])
        print(f"  Kinetoesque JSON: {len(traces)} trace records")
        if len(traces) == 0:
            errors.append("Kinetoesque output has no traces")

        # Check that we have HIP API traces
        hip_traces = [t for t in traces if "hip" in t.get("name", "").lower()]
        print(f"  Kinetoesque HIP traces: {len(hip_traces)}")
        if len(hip_traces) == 0:
            errors.append("Kinetoesque output has no HIP API traces")

    # Validate protonesque output
    if not os.path.isfile(protonesque_output):
        errors.append(f"Protonesque output file not found: {protonesque_output}")
    else:
        with open(protonesque_output, "r") as f:
            data = json.load(f)
        traces = data.get("protonesque-traces", [])
        print(f"  Protonesque JSON: {len(traces)} trace records")
        if len(traces) == 0:
            errors.append("Protonesque output has no traces")

        hip_traces = [t for t in traces if "hip" in t.get("name", "").lower()]
        print(f"  Protonesque HIP traces: {len(hip_traces)}")
        if len(hip_traces) == 0:
            errors.append("Protonesque output has no HIP API traces")

    # Report results
    print("\n=== Test Results ===")
    if errors:
        for err in errors:
            print(f"  FAIL: {err}", file=sys.stderr)
        print(f"\nFAILED ({len(errors)} error(s))")
        return 1
    else:
        print("  All checks passed")
        print("\nPASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
