// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <array>
#include <cstdlib>
#include <thread>
#include <vector>

#include "hip/hip_runtime.h"

/* Macro for checking GPU API return values */
#define HIP_ASSERT(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        hipError_t gpuErr = call;                                                                  \
        if(hipSuccess != gpuErr)                                                                   \
        {                                                                                          \
            printf(                                                                                \
                "GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr));   \
            exit(1);                                                                               \
        }                                                                                          \
    } while(0)

static void
copy_to_dev(const hipStream_t stream)
{
    unsigned int n   = (32 * 1024);  // 32KB
    double*      A_h = nullptr;
    double*      A_d = nullptr;

    HIP_ASSERT(hipHostMalloc(&A_h, n * sizeof(double)));
    HIP_ASSERT(hipMalloc(&A_d, n * sizeof(double)));

    for(unsigned int i = 0; i < n; ++i)
    {
        A_h[i] = 123.5;
    }
    HIP_ASSERT(hipMemcpyAsync(A_d, A_h, n * sizeof(double), hipMemcpyHostToDevice, stream));
    // Repeat to make sure streams remain the same
    HIP_ASSERT(hipMemcpyAsync(A_d, A_h, n * sizeof(double), hipMemcpyHostToDevice, stream));

    // Release device memory
    HIP_ASSERT(hipFree(A_d));
    // Release host memory
    HIP_ASSERT(hipHostFree(A_h));
}

int
main(int argc, char** argv)
{
    // Test hipStreamPerThread with multiple threads
    const size_t                         num_streams = 3;
    const size_t                         thread_cnt  = argc < 2 ? 9 : atoi(argv[1]);
    std::vector<std::thread>             threads{};
    std::array<hipStream_t, num_streams> streams{};
    threads.reserve(thread_cnt);
    threads.emplace_back(std::thread(copy_to_dev, nullptr));
    for(size_t i = 1, j = 0; i < thread_cnt; ++i)
    {
        if(i % 3 == 0)
        {
            threads.emplace_back(std::thread(copy_to_dev, hipStreamLegacy));
        }
        else if(i % 3 == 1)
        {
            threads.emplace_back(std::thread(copy_to_dev, hipStreamPerThread));
        }
        else
        {
            HIP_ASSERT(hipStreamCreate(&streams[j]));
            threads.emplace_back(std::thread(copy_to_dev, streams[j++]));
        }
    }
    for(auto& thread : threads)
    {
        thread.join();
    }
    return 0;
}
