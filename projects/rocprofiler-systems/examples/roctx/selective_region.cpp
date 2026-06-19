// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates the ROCPROFSYS_SELECTED_REGIONS selective region tracing feature.
// Supports two region marker styles (selected via --push-pop argument):
//   default (start/stop): roctxRangeStartA / roctxRangeStop
//   --push-pop:           roctxRangePushA  / roctxRangePop
//
// Multiple threads run concurrently, synchronized at region boundaries via barriers
// so that thread 0 manages region markers on behalf of all threads.
//
// Code flow per thread:
//   CodeBlock_A                                 (outside any region)
//   --- barrier --- thread 0: <start>("Region1") --- barrier ---
//     CodeBlock_B                               (inside Region1)
//     --- barrier --- thread 0: <start>("Region2") --- barrier ---
//       CodeBlock_C                             (inside Region1 + Region2)
//     --- barrier --- thread 0: <stop>(Region2) --- barrier ---
//     CodeBlock_D                               (inside Region1)
//   --- barrier --- thread 0: <stop>(Region1) --- barrier ---
//   --- barrier --- thread 0: <start>("Region3") --- barrier ---
//     CodeBlock_E                               (inside Region3)
//   --- barrier --- thread 0: <stop>(Region3) --- barrier ---
//   --- barrier --- thread 0: <start>("Region1") --- barrier ---
//     CodeBlock_F                               (inside Region1 again)
//   --- barrier --- thread 0: <stop>(Region1) --- barrier ---
//   CodeBlock_G                                 (outside any region)
//
// Run without filter (traces everything):
//   rocprof-sys -- ./selective_region
//   rocprof-sys -- ./selective_region --push-pop
//
// Run with filter (only traces inside "Region1"):
//   ROCPROFSYS_SELECTED_REGIONS="Region1" rocprof-sys -- ./selective_region
//   ROCPROFSYS_SELECTED_REGIONS="Region1" rocprof-sys -- ./selective_region --push-pop
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
run(int tid, hipStream_t stream, float* d, bool use_push_pop)
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
    if(tid == 0)
    {
        region1_id = range_start("Region1", use_push_pop);
    }
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_B, stream, d);

    barrier.wait();
    if(tid == 0)
    {
        region2_id = range_start("Region2", use_push_pop);
    }
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_C, stream, d);

    barrier.wait();
    if(tid == 0)
    {
        range_stop(region2_id, use_push_pop);
    }
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_D, stream, d);

    barrier.wait();
    if(tid == 0)
    {
        range_stop(region1_id, use_push_pop);
    }
    barrier.wait();

    barrier.wait();
    if(tid == 0)
    {
        region3_id = range_start("Region3", use_push_pop);
    }
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_E, stream, d);

    barrier.wait();
    if(tid == 0)
    {
        range_stop(region3_id, use_push_pop);
    }
    barrier.wait();

    barrier.wait();
    if(tid == 0)
    {
        region1b_id = range_start("Region1", use_push_pop);
    }
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_F, stream, d);

    barrier.wait();
    if(tid == 0)
    {
        range_stop(region1b_id, use_push_pop);
    }
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_G, stream, d);
}

int
main(int argc, char** argv)
{
    const bool use_push_pop = (argc > 1 && std::string_view(argv[1]) == "--push-pop");

    gpu_buffer buf;
    float*     d = buf.get();

    run_on_threads(DEFAULT_NUM_THREADS, [d, use_push_pop](int tid, hipStream_t stream) {
        run(tid, stream, d, use_push_pop);
    });

    return 0;
}
