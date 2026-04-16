// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates interleaving of Pause/Resume with Selective Region Tracing.
// Scenario: Region ends while profiling is paused, resume occurs outside region.
//
// The pause happens inside the target region and is valid. Then the region
// ends while still paused (a warning is logged). After the region ends,
// the resume occurs outside the region and is ignored.
//
// Code flow:
//   roctxRangeStartA("Region 1")
//     CodeBlock_A                               (profiled)
//     roctxProfilerPause
//     CodeBlock_C                               (paused — not profiled)
//   roctxRangeStop(Region 1)
//   CodeBlock_D                                 (outside region — not profiled)
//   roctxProfilerResume (outside — ignored)
//
// Run with filter:
//   ROCPROFSYS_TRACE_REGION="Region 1" rocprof-sys -- ./selective_region_pause_3
//
// Expected: profiling data recorded for {CodeBlock_A}

#include "roctx_example_kernels.hpp"

DEFINE_KERNEL(CodeBlock_A, 10)
DEFINE_KERNEL(CodeBlock_C, 30)
DEFINE_KERNEL(CodeBlock_D, 40)

int
main()
{
    gpu_buffer buf;
    float*     d = buf.get();

    roctx_thread_id_t roctx_tid{};
    roctxGetThreadId(&roctx_tid);

    roctx_range_id_t region1_id = roctxRangeStartA("Region1");

    LAUNCH_KERNEL(CodeBlock_A, d);

    roctxProfilerPause(roctx_tid);

    LAUNCH_KERNEL(CodeBlock_C, d);

    roctxRangeStop(region1_id);

    LAUNCH_KERNEL(CodeBlock_D, d);

    roctxProfilerResume(roctx_tid);

    return 0;
}
