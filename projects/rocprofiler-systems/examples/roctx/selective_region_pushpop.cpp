// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Manual test SR-M09: region filtering applies only to roctxRangeStartA/roctxRangeStop,
// not roctxRangePushA/roctxRangePop. Same kernel layout as selective_region.cpp but
// uses push/pop markers with identical region names (Region1, Region2, Region3).
//
// Run with region filter (filter should NOT match push/pop names for tracing control):
//   ROCPROFSYS_SELECTED_REGIONS=Region1 \
//   ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,kernel_dispatch,marker_core_range_api \
//   rocprof-sys-run -- ./selective_region_pushpop
//
// Expected: no selective-region window opens (filter ignored for push/pop) — typically
// no kernel dispatches in the trace while the filter is active. Compare with
// selective_region + Region1 filter (kernels B,C,D,F present).

#include "roctx_example_kernels.hpp"

DEFINE_KERNEL(CodeBlock_A, 10)
DEFINE_KERNEL(CodeBlock_B, 20)
DEFINE_KERNEL(CodeBlock_C, 30)
DEFINE_KERNEL(CodeBlock_D, 40)
DEFINE_KERNEL(CodeBlock_E, 50)
DEFINE_KERNEL(CodeBlock_F, 60)
DEFINE_KERNEL(CodeBlock_G, 70)

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
        printf("[selective_region_pushpop][thread %d] starting\n", tid);
    }

    LAUNCH_BLOCK(CodeBlock_A, stream, d);

    barrier.wait();
    if(tid == 0) roctxRangePushA("Region1");
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_B, stream, d);

    barrier.wait();
    if(tid == 0) roctxRangePushA("Region2");
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_C, stream, d);

    barrier.wait();
    if(tid == 0) roctxRangePop();
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_D, stream, d);

    barrier.wait();
    if(tid == 0) roctxRangePop();
    barrier.wait();

    barrier.wait();
    if(tid == 0) roctxRangePushA("Region3");
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_E, stream, d);

    barrier.wait();
    if(tid == 0) roctxRangePop();
    barrier.wait();

    barrier.wait();
    if(tid == 0) roctxRangePushA("Region1");
    barrier.wait();

    LAUNCH_BLOCK(CodeBlock_F, stream, d);

    barrier.wait();
    if(tid == 0) roctxRangePop();
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
