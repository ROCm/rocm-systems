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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// One range of three order-dependent dispatches, for rocprofv3 --replay-mode range / range-host.
//
// RR_V3_MODE=packet (default) calls rocprofiler_range_replay_begin / _end around the dispatches.
// RR_V3_MODE=host opens the same dispatches with ROCPROFILER_REPLAY_RANGE_BEGIN / _END, so each
// extra pass re-runs this code. 0 -> 1 -> 5 -> 18. The process prints [rr-v3] PASS when the value
// it observes after the range is 18, which is the value one execution produces.

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

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

namespace
{
constexpr uint64_t kRangeId         = 1;
constexpr int      kRangeDispatches = 3;
constexpr int      kExpected        = 18;

__global__ void
rr_v3_step(int* acc, int add)
{
    if(threadIdx.x == 0) *acc = (*acc * 3) + add;
}

int
launch_range(int* acc)
{
    for(int add = 1; add <= kRangeDispatches; ++add)
    {
        rr_v3_step<<<1, 1>>>(acc, add);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());
    return EXIT_SUCCESS;
}
}  // namespace

int
main()
{
    const char* mode_env = std::getenv("RR_V3_MODE");
    const bool  host     = mode_env != nullptr && std::strcmp(mode_env, "host") == 0;

    int* acc = nullptr;
    HIP_CHECK(hipMalloc(&acc, sizeof(int)));
    HIP_CHECK(hipMemset(acc, 0, sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());

    if(host)
    {
        ROCPROFILER_REPLAY_RANGE_BEGIN(kRangeId)
        {
            const auto begin_status = rocp_begin_status_;
            if(begin_status != ROCPROFILER_STATUS_SUCCESS)
            {
                fprintf(stderr, "[rr-v3] FAIL begin status %d\n", static_cast<int>(begin_status));
                return EXIT_FAILURE;
            }
            if(launch_range(acc) != EXIT_SUCCESS) return EXIT_FAILURE;
        }
        ROCPROFILER_REPLAY_RANGE_END()
    }
    else
    {
        if(const auto status = rocprofiler_range_replay_begin(kRangeId);
           status != ROCPROFILER_STATUS_SUCCESS)
        {
            fprintf(stderr, "[rr-v3] FAIL begin status %d\n", static_cast<int>(status));
            return EXIT_FAILURE;
        }
        if(launch_range(acc) != EXIT_SUCCESS) return EXIT_FAILURE;
        if(const auto status = rocprofiler_range_replay_end(); status != ROCPROFILER_STATUS_SUCCESS)
        {
            fprintf(stderr, "[rr-v3] FAIL end status %d\n", static_cast<int>(status));
            return EXIT_FAILURE;
        }
    }

    int acc_h = 0;
    HIP_CHECK(hipMemcpy(&acc_h, acc, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(acc));

    printf("[rr-v3] mode=%s acc=%d\n", host ? "host" : "packet", acc_h);
    if(acc_h != kExpected)
    {
        fprintf(stderr, "[rr-v3] FAIL acc=%d expected %d\n", acc_h, kExpected);
        return EXIT_FAILURE;
    }
    printf("[rr-v3] PASS\n");
    return EXIT_SUCCESS;
}
