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

// Stream create/destroy churn with HIP completion handlers in flight (ROCM-33989).
//
// Phase 1 ("poll"): each cycle creates a stream, launches a kernel that spins for kernel_ms,
// polls it with hipStreamQuery until it completes, synchronizes and destroys the stream. Every
// query of an unfinished stream enqueues a HIP marker whose completion handler runs on ROCr's
// async-events thread, so thousands of handlers are pending when the kernel finishes.
// Phase 2 ("burst"): each cycle creates a stream, queues burst_launches tiny kernels,
// synchronizes and destroys the stream; HIP registers a completion handler per command batch.
//
// In both phases the next hipStreamCreate runs while the async-events thread is still draining
// the destroyed stream's handlers, and the last of them releases that stream's device resources
// under HIP's per-device lock. With GPU_MAX_HW_QUEUES raised so that every stream gets its own
// hardware queue, each hipStreamCreate creates one through hsa_amd_queue_create while holding
// that same lock. A profiler that waits for the async-events thread inside that call (for
// example by destroying a queue there) deadlocks hipStreamCreate; the test's TIMEOUT is the
// failure signal.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#define HIP_CHECK(CALL)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d HIP error (%d): %s\n",                                                  \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    static_cast<int>(error_),                                                      \
                    hipGetErrorString(error_));                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

namespace
{
// Busy-waits on the device's constant-rate wall clock for TICKS ticks.
__global__ void
spin_kernel(long long ticks, int* sink)
{
    const long long start = wall_clock64();
    int             iters = 0;
    while(wall_clock64() - start < ticks)
        ++iters;
    if(iters < 0) *sink = iters;  // never true; keeps the loop alive
}

__global__ void
tiny_kernel(int* sink)
{
    if(threadIdx.x == 0 && blockIdx.x == 12345) *sink = 1;  // never true
}

// Every parameter counts something, so zero is rejected along with non-numeric input.
uint64_t
parse_dimension(const char* text, const char* name)
{
    try
    {
        const auto value = std::stoul(text);
        if(value > 0) return value;
    } catch(const std::exception&)
    {}

    std::cerr << "[hip-stream-cycle] " << name << " must be a positive integer, got '" << text
              << "'" << std::endl;
    exit(EXIT_FAILURE);
}
}  // namespace

int
main(int argc, char** argv)
{
    uint64_t poll_cycles    = 64;
    uint64_t kernel_ms      = 20;
    uint64_t burst_cycles   = 8;
    uint64_t burst_launches = 4096;
    if(argc > 1) poll_cycles = parse_dimension(argv[1], "poll_cycles");
    if(argc > 2) kernel_ms = parse_dimension(argv[2], "kernel_ms");
    if(argc > 3) burst_cycles = parse_dimension(argv[3], "burst_cycles");
    if(argc > 4) burst_launches = parse_dimension(argv[4], "burst_launches");

    std::cout << "[hip-stream-cycle] poll_cycles=" << poll_cycles << " kernel_ms=" << kernel_ms
              << " burst_cycles=" << burst_cycles << " burst_launches=" << burst_launches
              << std::endl;

    int rate_khz = 0;
    HIP_CHECK(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, 0));
    const long long ticks = static_cast<long long>(rate_khz) * static_cast<long long>(kernel_ms);
    int*            sink  = nullptr;
    HIP_CHECK(hipMalloc(&sink, sizeof(int)));

    for(uint64_t cycle = 0; cycle < poll_cycles; ++cycle)
    {
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreate(&stream));
        spin_kernel<<<1, 64, 0, stream>>>(ticks, sink);
        HIP_CHECK(hipGetLastError());
        hipError_t query = hipErrorNotReady;
        while((query = hipStreamQuery(stream)) == hipErrorNotReady)
        {}
        HIP_CHECK(query);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipStreamDestroy(stream));
        std::cout << "[hip-stream-cycle] poll cycle " << cycle << " complete" << std::endl;
    }

    for(uint64_t cycle = 0; cycle < burst_cycles; ++cycle)
    {
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreate(&stream));
        for(uint64_t launch = 0; launch < burst_launches; ++launch)
            tiny_kernel<<<1, 64, 0, stream>>>(sink);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipStreamDestroy(stream));
        std::cout << "[hip-stream-cycle] burst cycle " << cycle << " complete" << std::endl;
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(sink));
    std::cout << "[hip-stream-cycle] complete" << std::endl;
    return 0;
}
