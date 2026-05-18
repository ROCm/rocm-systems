// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Manual test SR-M07: OpenMP parallel region with a ROCTx process-wide range (HotPhase).
// HIP kernels inside HotPhase should be traced when ROCPROFSYS_SELECTED_REGIONS=HotPhase.
//
// Build requires OpenMP (host). Example:
//   cmake --build <build_dir> --target omp_app
//
// Run:
//   ROCPROFSYS_USE_OMPT=ON \
//   ROCPROFSYS_SELECTED_REGIONS=HotPhase \
//   ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,kernel_dispatch,marker_core_range_api \
//   rocprof-sys-run -- ./omp_app
//
// Expected: CodeBlock_Warmup absent; CodeBlock_Hot present. OMPT slices optional.

#include "roctx_example_kernels.hpp"

#include <omp.h>

DEFINE_KERNEL(CodeBlock_Warmup, 10)
DEFINE_KERNEL(CodeBlock_Hot, 20)
DEFINE_KERNEL(CodeBlock_Cooldown, 30)

constexpr int CODE_BLOCK_REPEATS = 5;

#define LAUNCH_BLOCK(name, data)                                                           \
    do                                                                                     \
    {                                                                                      \
        for(int _blk = 0; _blk < CODE_BLOCK_REPEATS; ++_blk)                               \
            LAUNCH_KERNEL(name, data);                                                     \
    } while(0)

int
main()
{
    gpu_buffer buf;
    float*     d = buf.get();

    const int nthreads = static_cast<int>(DEFAULT_NUM_THREADS);

    LAUNCH_BLOCK(CodeBlock_Warmup, d);

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();

        if(tid == 0)
        {
            std::lock_guard<std::mutex> lk{ print_lock };
            printf("[omp_app] OpenMP team size %d\n", omp_get_num_threads());
        }

#pragma omp barrier

        roctx_range_id_t hot_id = 0;
        if(tid == 0) hot_id = roctxRangeStartA("HotPhase");

#pragma omp barrier

        LAUNCH_BLOCK(CodeBlock_Hot, d);

#pragma omp barrier

        if(tid == 0) roctxRangeStop(hot_id);

#pragma omp barrier
    }

    LAUNCH_BLOCK(CodeBlock_Cooldown, d);

    return 0;
}
