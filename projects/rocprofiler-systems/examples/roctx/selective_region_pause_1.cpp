// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates interleaving of Pause/Resume with Selective Region Tracing.
// Scenario: Pause and Resume both occur INSIDE the target region.
//
// Supports two region marker styles (selected via --push-pop argument):
//   default (start/stop): roctxRangeStartA / roctxRangeStop
//   --push-pop:           roctxRangePushA  / roctxRangePop
//
// Code flow:
//   CodeBlock_Z                                 (outside region — not profiled)
//   <range start>("Region1")
//     CodeBlock_A                               (profiled)
//     roctxProfilerPause
//     CodeBlock_B                               (paused — not profiled)
//     roctxProfilerResume
//     CodeBlock_C                               (profiled)
//   <range stop>
//   CodeBlock_D                                 (outside region — not profiled)
//
// Run with filter:
//   ROCPROFSYS_SELECTED_REGIONS="Region1" rocprof-sys -- ./selective_region_pause_1
//   ROCPROFSYS_SELECTED_REGIONS="Region1" rocprof-sys -- ./selective_region_pause_1
//   --push-pop
//
// Expected: profiling data recorded for {CodeBlock_A, CodeBlock_C}

#include "roctx_example_kernels.hpp"

DEFINE_KERNEL(CodeBlock_Z, 10)
DEFINE_KERNEL(CodeBlock_A, 20)
DEFINE_KERNEL(CodeBlock_B, 30)
DEFINE_KERNEL(CodeBlock_C, 40)
DEFINE_KERNEL(CodeBlock_D, 50)

int
main(int argc, char** argv)
{
    const bool use_push_pop = (argc > 1 && std::string_view(argv[1]) == "--push-pop");

    gpu_buffer buf;
    float*     d = buf.get();

    roctx_thread_id_t roctx_tid{};
    roctxGetThreadId(&roctx_tid);

    LAUNCH_KERNEL(CodeBlock_Z, d);

    roctx_range_id_t region1_id = range_start("Region1", use_push_pop);

    LAUNCH_KERNEL(CodeBlock_A, d);

    roctxProfilerPause(roctx_tid);

    LAUNCH_KERNEL(CodeBlock_B, d);

    roctxProfilerResume(roctx_tid);

    LAUNCH_KERNEL(CodeBlock_C, d);

    range_stop(region1_id, use_push_pop);

    LAUNCH_KERNEL(CodeBlock_D, d);

    return 0;
}
