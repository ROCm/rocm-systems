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

/**
 * @file tests/kernel-replay-multi-agent/main.cpp
 *
 * @brief Workload for the multi-agent kernel replay test.
 *
 * One host thread per GPU, on the first two GPUs, launches an accumulate kernel that the client
 * tool replays. Each launch adds 1 to every element, so a replay that fails to restore memory
 * between passes leaves more than `launches` in it.
 *
 *   kernel-replay-multi-agent-testapp [launches] [MiB per buffer] [MiB to leave free, 0 = all]
 *
 * Leaving little free device memory fills each GPU with fine-grained memory, which kernel replay
 * does not snapshot. The snapshot then no longer fits the GPU budget and is partly kept in pinned
 * host memory, which GPUs on the same CPU share.
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        const hipError_t _status = (call);                                                         \
        if(_status != hipSuccess)                                                                  \
        {                                                                                          \
            std::fprintf(stderr,                                                                   \
                         "[multi-agent] FAIL: %s returned %s\n",                                   \
                         #call,                                                                    \
                         hipGetErrorString(_status));                                              \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while(0)

namespace
{
// The client tool replays only dispatches with this workgroup size.
constexpr unsigned kAccumulateBlockX = 192;

__global__ void
accumulate(float* data, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if(i < n) data[i] += 1.0f;
}

struct result_t
{
    size_t wrong         = 0;
    float  first         = 0;
    size_t ballast_bytes = 0;
};

void
run_device(int device, int launches, size_t bytes, size_t leave_free, result_t& out)
{
    HIP_CHECK(hipSetDevice(device));

    auto ballast = std::vector<void*>{};
    if(leave_free > 0)
    {
        size_t free_bytes  = 0;
        size_t total_bytes = 0;
        HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
        const size_t chunk = size_t{8} << 30;
        while(free_bytes > leave_free + bytes)
        {
            const size_t take = std::min(chunk, free_bytes - leave_free - bytes);
            void*        ptr  = nullptr;
            if(hipExtMallocWithFlags(&ptr, take, hipDeviceMallocFinegrained) != hipSuccess) break;
            ballast.push_back(ptr);
            out.ballast_bytes += take;
            HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
        }
    }

    const size_t n    = bytes / sizeof(float);
    float*       data = nullptr;
    HIP_CHECK(hipMalloc(&data, bytes));
    HIP_CHECK(hipMemset(data, 0, bytes));
    const auto grid = dim3(static_cast<unsigned>((n + kAccumulateBlockX - 1) / kAccumulateBlockX));
    for(int k = 0; k < launches; ++k)
        hipLaunchKernelGGL(accumulate, grid, dim3(kAccumulateBlockX), 0, 0, data, n);
    HIP_CHECK(hipDeviceSynchronize());

    auto host = std::vector<float>(n);
    HIP_CHECK(hipMemcpy(host.data(), data, bytes, hipMemcpyDeviceToHost));
    out.first = host.front();
    out.wrong = static_cast<size_t>(std::count_if(host.begin(), host.end(), [&](float value) {
        return value != static_cast<float>(launches);
    }));

    HIP_CHECK(hipFree(data));
    for(auto* ptr : ballast)
        HIP_CHECK(hipFree(ptr));
}
}  // namespace

int
main(int argc, char** argv)
{
    const int    launches       = argc > 1 ? std::atoi(argv[1]) : 4;
    const size_t buffer_mib     = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 256;
    const size_t leave_free_mib = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 0;

    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    if(count < 2)
    {
        std::printf("[multi-agent] skipped: fewer than two GPUs\n");
        return EXIT_SUCCESS;
    }

    constexpr int devices = 2;
    for(int device = 0; leave_free_mib > 0 && device < devices; ++device)
    {
        int integrated = 0;
        HIP_CHECK(hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, device));
        // An integrated GPU's device memory is the host's, so the ballast would fill host memory.
        if(integrated != 0)
        {
            std::printf("[multi-agent] skipped: filling device memory needs discrete GPUs\n");
            return EXIT_SUCCESS;
        }
    }

    auto results = std::vector<result_t>(devices);
    auto threads = std::vector<std::thread>{};
    for(int device = 0; device < devices; ++device)
        threads.emplace_back(run_device,
                             device,
                             launches,
                             buffer_mib << 20,
                             leave_free_mib << 20,
                             std::ref(results[device]));
    for(auto& thread : threads)
        thread.join();

    auto failures = 0;
    for(int device = 0; device < devices; ++device)
    {
        const auto& result = results[device];
        std::printf("[multi-agent] device %d: element 0 = %.0f, %zu wrong, ballast %zu GiB\n",
                    device,
                    result.first,
                    result.wrong,
                    result.ballast_bytes >> 30);
        if(result.wrong != 0) ++failures;
    }
    if(failures != 0)
    {
        std::printf("[multi-agent] FAIL: %d device(s) kept memory a replay pass changed\n",
                    failures);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
