// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "range.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (call);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(                                                                               \
                stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(_err), __FILE__, __LINE__); \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while(0)

// Each dispatch reads what its predecessor wrote, so a pass that skipped, reordered, or started
// from an unrestored state produces a different number. Named to match kKernelName.
__global__ void
rr_local_context_step(int* acc, int add)
{
    if(threadIdx.x == 0) *acc = (*acc * 3) + add;
}

int
main()
{
    auto* begin_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_begin)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_begin"));
    auto* end_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_end)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_end"));

    if(begin_fn == nullptr || end_fn == nullptr)
    {
        fprintf(stderr, "[app] FAIL: the range replay API is not in this process\n");
        return EXIT_FAILURE;
    }

    int* acc = nullptr;
    HIP_CHECK(hipMalloc(&acc, sizeof(int)));
    HIP_CHECK(hipMemset(acc, 0, sizeof(int)));

    // A device allocation inside a range is a decline reason, so everything the range touches is
    // allocated and settled before it opens.
    HIP_CHECK(hipDeviceSynchronize());

    if(const auto status = begin_fn(kRangeId); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: rocprofiler_range_replay_begin returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    for(uint64_t add = 1; add <= kRangeDispatches; ++add)
    {
        rr_local_context_step<<<1, 64>>>(acc, static_cast<int>(add));
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipDeviceSynchronize());

    if(const auto status = end_fn(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: rocprofiler_range_replay_end returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    int acc_h = 0;
    HIP_CHECK(hipMemcpy(&acc_h, acc, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(acc));

    // Whatever the services did during the replayed passes, the application must get back the
    // value its own execution of the range produced.
    printf("[app] acc=%d\n", acc_h);
    if(acc_h != kExpectedResult)
    {
        fprintf(stderr, "[app] FAIL: acc=%d (expected %d)\n", acc_h, kExpectedResult);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
