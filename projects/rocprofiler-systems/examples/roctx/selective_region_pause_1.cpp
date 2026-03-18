// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates interleaving of Pause/Resume with Selective Region Tracing.
// Scenario: Pause and Resume both occur INSIDE the target region.
//
// Code flow:
//   CodeBlock_Z                                 (outside region — not profiled)
//   roctxRangeStartA("Region 1")               (enter target region — profiling starts)
//     CodeBlock_A                               (profiled)
//     roctxProfilerPause                        (valid pause inside region — profiling
//     stops) CodeBlock_B                               (paused — not profiled)
//     roctxProfilerResume                       (valid resume inside region — profiling
//     resumes) CodeBlock_C                               (profiled)
//   roctxRangeStop("Region 1")                  (region ends — profiling stops)
//   CodeBlock_D                                 (outside region — not profiled)
//
// Run with filter:
//   ROCPROFSYS_TRACE_REGION="Region 1" rocprof-sys -- ./selective_region_pause_1
//
// Expected: profiling data recorded for {CodeBlock_A, CodeBlock_C}

#include "roctx_example_kernels.hpp"

DEFINE_KERNEL(CodeBlock_Z, 10)
DEFINE_KERNEL(CodeBlock_A, 20)
DEFINE_KERNEL(CodeBlock_B, 30)
DEFINE_KERNEL(CodeBlock_C, 40)
DEFINE_KERNEL(CodeBlock_D, 50)

int
main()
{
    gpu_buffer buf;
    float*     d = buf.get();

    roctx_thread_id_t tid{};
    roctxGetThreadId(&tid);

    // Outside region
    LAUNCH_KERNEL(CodeBlock_Z, d);

    // Region 1
    roctx_range_id_t region1_id = roctxRangeStartA("Region 1");

    LAUNCH_KERNEL(CodeBlock_A, d);

    // Pause inside region
    roctxProfilerPause(tid);

    LAUNCH_KERNEL(CodeBlock_B, d);

    // Resume inside region
    roctxProfilerResume(tid);

    LAUNCH_KERNEL(CodeBlock_C, d);

    roctxRangeStop(region1_id);

    // Outside region
    LAUNCH_KERNEL(CodeBlock_D, d);

    return 0;
}
