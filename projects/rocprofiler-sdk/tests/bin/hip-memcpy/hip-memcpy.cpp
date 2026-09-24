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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <rocprofiler-sdk-roctx/roctx.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#define HIP_API_CALL(CALL)                                                                         \
    do                                                                                             \
    {                                                                                              \
        auto _status = (CALL);                                                                     \
        if(_status != hipSuccess)                                                                  \
        {                                                                                          \
            std::cerr << __FILE__ << ":" << __LINE__ << " :: HIP error in " << #CALL << ": "       \
                      << hipGetErrorString(_status) << '\n';                                       \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    } while(false)

namespace
{
constexpr std::size_t kCopiesPerDirection = 4;
constexpr std::size_t kElementsPerCopy    = 1U << 14;  // 16K ints (64 KiB)
constexpr std::size_t kBytesPerCopy       = kElementsPerCopy * sizeof(int);

void
verify_equal(const int* lhs, const int* rhs, std::size_t count)
{
    for(std::size_t i = 0; i < count; ++i)
    {
        if(lhs[i] != rhs[i])
        {
            std::cerr << "[hip-memcpy] mismatch at idx " << i << ": got " << lhs[i] << ", expected "
                      << rhs[i] << '\n';
            throw std::runtime_error("memory copy verification failed");
        }
    }
}
}  // namespace

__global__ void
empty_kernel()
{}

namespace
{
void
run(int tid)
{
    HIP_API_CALL(hipSetDevice(0));

    hipStream_t stream = nullptr;
    HIP_API_CALL(hipStreamCreate(&stream));

    int* h_async_src = nullptr;
    int* h_async_dst = nullptr;
    int* d_async_buf = nullptr;

    int* h_src[kCopiesPerDirection] = {};
    int* h_dst[kCopiesPerDirection] = {};
    int* d_buf[kCopiesPerDirection] = {};

    HIP_API_CALL(hipHostMalloc(&h_async_src, kBytesPerCopy, hipHostMallocDefault));
    HIP_API_CALL(hipHostMalloc(&h_async_dst, kBytesPerCopy, hipHostMallocDefault));
    HIP_API_CALL(hipMalloc(&d_async_buf, kBytesPerCopy));
    for(std::size_t i = 0; i < kElementsPerCopy; ++i)
    {
        h_async_src[i] = static_cast<int>((tid + 1) * kElementsPerCopy + i);
        h_async_dst[i] = -1;
    }

    for(std::size_t i = 0; i < kCopiesPerDirection; ++i)
    {
        // Pinned host allocations keep HIP on the buffer-batch path that reaches
        // hsa_amd_memory_async_batch_copy.
        HIP_API_CALL(hipHostMalloc(&h_src[i], kBytesPerCopy, hipHostMallocDefault));
        HIP_API_CALL(hipHostMalloc(&h_dst[i], kBytesPerCopy, hipHostMallocDefault));
        HIP_API_CALL(hipMalloc(&d_buf[i], kBytesPerCopy));

        for(std::size_t j = 0; j < kElementsPerCopy; ++j)
        {
            h_src[i][j] = static_cast<int>((tid + 1) * 1000003 + i * kElementsPerCopy + j);
            h_dst[i][j] = -1;
        }
    }

    auto range_id = roctxRangeStart("run/hip-memcpy");

    roctxRangePush("run/hip-memcpy/async");
    HIP_API_CALL(
        hipMemcpyAsync(d_async_buf, h_async_src, kBytesPerCopy, hipMemcpyHostToDevice, stream));
    empty_kernel<<<1, 1, 0, stream>>>();
    HIP_API_CALL(hipGetLastError());
    HIP_API_CALL(
        hipMemcpyAsync(h_async_dst, d_async_buf, kBytesPerCopy, hipMemcpyDeviceToHost, stream));
    roctxRangePop();

    std::size_t fail_idx = static_cast<std::size_t>(-1);

    roctxRangePush("run/hip-memcpy/grouped");
    for(std::size_t i = 0; i < kCopiesPerDirection; ++i)
    {
        void*       dst         = d_buf[i];
        void*       src         = h_src[i];
        std::size_t size        = kBytesPerCopy;
        std::size_t attrs_idx[] = {0};
        HIP_API_CALL(
            hipMemcpyBatchAsync(&dst, &src, &size, 1, nullptr, attrs_idx, 0, &fail_idx, stream));
    }

    for(std::size_t i = 0; i < kCopiesPerDirection; ++i)
    {
        void*       dst         = h_dst[i];
        void*       src         = d_buf[i];
        std::size_t size        = kBytesPerCopy;
        std::size_t attrs_idx[] = {0};
        HIP_API_CALL(
            hipMemcpyBatchAsync(&dst, &src, &size, 1, nullptr, attrs_idx, 0, &fail_idx, stream));
    }
    roctxRangePop();

    HIP_API_CALL(hipStreamSynchronize(stream));

    verify_equal(h_async_dst, h_async_src, kElementsPerCopy);
    HIP_API_CALL(hipFree(d_async_buf));
    HIP_API_CALL(hipHostFree(h_async_src));
    HIP_API_CALL(hipHostFree(h_async_dst));

    for(std::size_t i = 0; i < kCopiesPerDirection; ++i)
    {
        verify_equal(h_dst[i], h_src[i], kElementsPerCopy);
        HIP_API_CALL(hipFree(d_buf[i]));
        HIP_API_CALL(hipHostFree(h_src[i]));
        HIP_API_CALL(hipHostFree(h_dst[i]));
    }

    HIP_API_CALL(hipStreamDestroy(stream));
    roctxRangeStop(range_id);
}
}  // namespace

int
main()
{
    int ndevice  = 0;
    int nthreads = 2;

    HIP_API_CALL(hipGetDeviceCount(&ndevice));
    if(ndevice == 0)
    {
        std::cerr << "hip-memcpy requires at least one HIP device\n";
        return EXIT_FAILURE;
    }

    auto threads = std::vector<std::thread>{};
    for(int i = 0; i < nthreads; ++i)
        threads.emplace_back(run, i);

    for(auto& itr : threads)
        itr.join();

    std::cout << "[hip-memcpy] hipMemcpyAsync and hipMemcpyBatchAsync H2D + D2H completed\n";
    return EXIT_SUCCESS;
}
