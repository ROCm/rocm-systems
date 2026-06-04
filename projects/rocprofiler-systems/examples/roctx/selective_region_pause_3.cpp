// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates interleaving of Pause/Resume with Selective Region Tracing.
// Scenario: Region ends while profiling is paused, resume occurs outside region.
//
// Supports two region marker styles (selected via --push-pop argument):
//   default (start/stop): roctxRangeStartA / roctxRangeStop
//   --push-pop:           roctxRangePushA  / roctxRangePop
//
// The pause happens inside the target region and is valid. Then the region
// ends while still paused (a warning is logged). After the region ends,
// the resume occurs outside the region and is ignored.
//
// Code flow:
//   <range start>("Region1")
//     CodeBlock_A                               (profiled)
//     roctxProfilerPause
//     CodeBlock_C                               (paused — not profiled)
//   <range stop>
//   CodeBlock_D                                 (outside region — not profiled)
//   roctxProfilerResume (outside — ignored)
//
// Run with filter:
//   ROCPROFSYS_SELECTED_REGIONS="Region1" rocprof-sys -- ./selective_region_pause_3
//   ROCPROFSYS_SELECTED_REGIONS="Region1" rocprof-sys -- ./selective_region_pause_3
//   --push-pop
//
// Expected: profiling data recorded for {CodeBlock_A}

#include "roctx_example_kernels.hpp"

DEFINE_KERNEL(CodeBlock_A, 10)
DEFINE_KERNEL(CodeBlock_C, 30)
DEFINE_KERNEL(CodeBlock_D, 40)

int
main(int argc, char** argv)
{
    const bool use_push_pop = (argc > 1 && std::string_view(argv[1]) == "--push-pop");

    gpu_buffer buf;
    float*     d = buf.get();

    roctx_thread_id_t roctx_tid{};
    roctxGetThreadId(&roctx_tid);

    roctx_range_id_t region1_id = range_start("Region1", use_push_pop);

    LAUNCH_KERNEL(CodeBlock_A, d);

    roctxProfilerPause(roctx_tid);

    LAUNCH_KERNEL(CodeBlock_C, d);

    range_stop(region1_id, use_push_pop);

    LAUNCH_KERNEL(CodeBlock_D, d);

    roctxProfilerResume(roctx_tid);

    return 0;
}
