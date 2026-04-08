// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates interleaving of Pause/Resume with Selective Region Tracing.
// Scenario: Pause and Resume both occur INSIDE the target region.
//
// Code flow:
//   CodeBlock_Z                                 (outside region — not profiled)
//   roctxRangeStartA("Region 1")
//     CodeBlock_A                               (profiled)
//     roctxProfilerPause
//     CodeBlock_B                               (paused — not profiled)
//     roctxProfilerResume
//     CodeBlock_C                               (profiled)
//   roctxRangeStop(Region 1)
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

    roctx_thread_id_t roctx_tid{};
    roctxGetThreadId(&roctx_tid);

    LAUNCH_KERNEL(CodeBlock_Z, d);

    roctx_range_id_t region1_id = roctxRangeStartA("Region1");

    LAUNCH_KERNEL(CodeBlock_A, d);

    roctxProfilerPause(roctx_tid);

    LAUNCH_KERNEL(CodeBlock_B, d);

    roctxProfilerResume(roctx_tid);

    LAUNCH_KERNEL(CodeBlock_C, d);

    roctxRangeStop(region1_id);

    LAUNCH_KERNEL(CodeBlock_D, d);

    return 0;
}
