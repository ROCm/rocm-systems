// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates the ROCPROFSYS_TRACE_REGION selective region tracing feature.
// Uses roctxRangeStartA/roctxRangeStop (process-wide markers) for region filtering.
//
// Code flow:
//   CodeBlock_A                                 (outside any region)
//   roctxRangeStartA("Region 1")
//     CodeBlock_B                               (inside Region 1)
//     roctxRangeStartA("Region 2")
//       CodeBlock_C                             (inside Region 1 + Region 2)
//     roctxRangeStop("Region 2")
//     CodeBlock_D                               (inside Region 1)
//   roctxRangeStop("Region 1")
//   roctxRangeStartA("Region 3")
//     CodeBlock_E                               (inside Region 3)
//   roctxRangeStop("Region 3")
//   roctxRangeStartA("Region 1")
//     CodeBlock_F                               (inside Region 1 again)
//   roctxRangeStop("Region 1")
//   CodeBlock_G                                 (outside any region)
//
// Run without filter (traces everything):
//   rocprof-sys -- ./selective_region
//
// Run with filter (only traces inside "Region 1"):
//   ROCPROFSYS_TRACE_REGION="Region 1" rocprof-sys -- ./selective_region
//
// Expected with filter: profiling data recorded for {CodeBlock_B, CodeBlock_C,
//                        CodeBlock_D, CodeBlock_F}

#include "roctx_example_kernels.hpp"

DEFINE_KERNEL(CodeBlock_A, 10)
DEFINE_KERNEL(CodeBlock_B, 20)
DEFINE_KERNEL(CodeBlock_C, 30)
DEFINE_KERNEL(CodeBlock_D, 40)
DEFINE_KERNEL(CodeBlock_E, 50)
DEFINE_KERNEL(CodeBlock_F, 60)
DEFINE_KERNEL(CodeBlock_G, 70)

int
main()
{
    gpu_buffer buf;
    float*     d = buf.get();

    // Outside any region
    LAUNCH_KERNEL(CodeBlock_A, d);

    // Region 1
    roctx_range_id_t region1_id = roctxRangeStartA("Region 1");

    LAUNCH_KERNEL(CodeBlock_B, d);

    // Nested Region 2 (does not match filter)
    roctx_range_id_t region2_id = roctxRangeStartA("Region 2");

    LAUNCH_KERNEL(CodeBlock_C, d);

    roctxRangeStop(region2_id);

    LAUNCH_KERNEL(CodeBlock_D, d);

    roctxRangeStop(region1_id);

    // Region 3 (does not match filter)
    roctx_range_id_t region3_id = roctxRangeStartA("Region 3");

    LAUNCH_KERNEL(CodeBlock_E, d);

    roctxRangeStop(region3_id);

    // Region 1 again
    roctx_range_id_t region1b_id = roctxRangeStartA("Region 1");

    LAUNCH_KERNEL(CodeBlock_F, d);

    roctxRangeStop(region1b_id);

    // Outside any region
    LAUNCH_KERNEL(CodeBlock_G, d);

    return 0;
}
