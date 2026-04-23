// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates the ROCPROFSYS_SELECTED_REGIONS selective region tracing feature.
// Uses roctxRangeStartA/roctxRangeStop (process-wide markers) for region filtering.
// Multiple threads run concurrently, synchronized at region boundaries via barriers
// because roctxRangeStartA/roctxRangeStop are process-wide operations.
//
// Code flow per thread:
//   CodeBlock_A                                 (outside any region)
//   --- barrier --- thread 0: roctxRangeStartA("Region 1") --- barrier ---
//     CodeBlock_B                               (inside Region 1)
//     --- barrier --- thread 0: roctxRangeStartA("Region 2") --- barrier ---
//       CodeBlock_C                             (inside Region 1 + Region 2)
//     --- barrier --- thread 0: roctxRangeStop(Region 2) --- barrier ---
//     CodeBlock_D                               (inside Region 1)
//   --- barrier --- thread 0: roctxRangeStop(Region 1) --- barrier ---
//   --- barrier --- thread 0: roctxRangeStartA("Region 3") --- barrier ---
//     CodeBlock_E                               (inside Region 3)
//   --- barrier --- thread 0: roctxRangeStop(Region 3) --- barrier ---
//   --- barrier --- thread 0: roctxRangeStartA("Region 1") --- barrier ---
//     CodeBlock_F                               (inside Region 1 again)
//   --- barrier --- thread 0: roctxRangeStop(Region 1) --- barrier ---
//   CodeBlock_G                                 (outside any region)
//
// Run without filter (traces everything):
//   rocprof-sys -- ./selective_region
//
// Run with filter (only traces inside "Region 1"):
//   ROCPROFSYS_SELECTED_REGIONS="Region 1" rocprof-sys -- ./selective_region
//
// Expected with filter: profiling data recorded for {CodeBlock_B, CodeBlock_C,
//                        CodeBlock_D, CodeBlock_F} on each thread

#include "roctx_example_kernels.hpp"

DEFINE_KERNEL(CodeBlock_A, 10)
DEFINE_KERNEL(CodeBlock_B, 20)
DEFINE_KERNEL(CodeBlock_C, 30)
DEFINE_KERNEL(CodeBlock_D, 40)
DEFINE_KERNEL(CodeBlock_E, 50)
DEFINE_KERNEL(CodeBlock_F, 60)
DEFINE_KERNEL(CodeBlock_G, 70)

// Number of times each code block repeats its kernel launches.
// Multiplied with KERNEL_LAUNCH_REPEATS to ensure each thread runs >= 3 seconds.
constexpr int CODE_BLOCK_REPEATS = 5;

#define LAUNCH_BLOCK(name, stream, data)                                                 \
    do                                                                                   \
    {                                                                                    \
        for(int _blk = 0; _blk < CODE_BLOCK_REPEATS; ++_blk)                             \
            LAUNCH_KERNEL_STREAM(name, stream, data);                                    \
    } while(0)

static thread_barrier barrier{ DEFAULT_NUM_THREADS };

void
run(int tid, hipStream_t stream, float* d)
{
    {
        std::lock_guard<std::mutex> lk{ print_lock };
        printf("[selective_region][thread %d] starting\n", tid);
    }

    roctx_range_id_t region1_id  = 0;
    roctx_range_id_t region2_id  = 0;
    roctx_range_id_t region3_id  = 0;
    roctx_range_id_t region1b_id = 0;

    LAUNCH_BLOCK(CodeBlock_A, stream, d);

    barrier.wait();
    if(tid == 0) region1_id = roctxRangeStartA("Region1");
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_B, stream, d);

    barrier.wait();
    if(tid == 0) region2_id = roctxRangeStartA("Region2");
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_C, stream, d);

    barrier.wait();
    if(tid == 0) roctxRangeStop(region2_id);
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_D, stream, d);

    barrier.wait();
    if(tid == 0) roctxRangeStop(region1_id);
    barrier.wait();

    barrier.wait();
    if(tid == 0) region3_id = roctxRangeStartA("Region3");
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_E, stream, d);

    barrier.wait();
    if(tid == 0) roctxRangeStop(region3_id);
    barrier.wait();

    barrier.wait();
    if(tid == 0) region1b_id = roctxRangeStartA("Region1");
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_F, stream, d);

    barrier.wait();
    if(tid == 0) roctxRangeStop(region1b_id);
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_G, stream, d);
}

int
main()
{
    gpu_buffer buf;
    float*     d = buf.get();

    run_on_threads(DEFAULT_NUM_THREADS,
                   [d](int tid, hipStream_t stream) { run(tid, stream, d); });

    return 0;
}
