// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates interleaving of Pause/Resume with Selective Region Tracing.
// Scenario: Pause occurs OUTSIDE the target region (before it starts).
//
// The pause happens outside any target region, so it is not valid in the
// context of region filtering. When the target region starts, profiling
// begins normally. The resume inside the region is a no-op since there
// was no valid pause to undo.
//
// Code flow:
//   roctxProfilerPause
//   CodeBlock_Z                                 (outside region — not profiled)
//   roctxRangeStartA("Region 1")
//     CodeBlock_A                               (profiled)
//     CodeBlock_B                               (profiled)
//     roctxProfilerResume (no-op)
//     CodeBlock_C                               (profiled)
//   roctxRangeStop(Region 1)
//   CodeBlock_D                                 (outside region — not profiled)
//
// Run with filter:
//   ROCPROFSYS_SELECTED_REGIONS="Region 1" rocprof-sys -- ./selective_region_pause_2
//
// Expected: profiling data recorded for {CodeBlock_A, CodeBlock_B, CodeBlock_C}

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

    roctxProfilerPause(roctx_tid);

    LAUNCH_KERNEL(CodeBlock_Z, d);

    roctx_range_id_t region1_id = roctxRangeStartA("Region1");

    LAUNCH_KERNEL(CodeBlock_A, d);

    LAUNCH_KERNEL(CodeBlock_B, d);

    roctxProfilerResume(roctx_tid);

    LAUNCH_KERNEL(CodeBlock_C, d);

    roctxRangeStop(region1_id);

    LAUNCH_KERNEL(CodeBlock_D, d);

    return 0;
}
