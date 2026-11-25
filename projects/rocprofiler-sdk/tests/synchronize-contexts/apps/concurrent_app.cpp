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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// An application offering several independent lanes, for the rocprofiler_synchronize_context tool
// that needs dispatches in flight on other threads while one thread synchronizes.
//
// Each lane carries its own stream and scratch. Lanes backed by the default stream would
// serialize the enqueues and remove the contention that tool exists to create.
//
// No copy entry: the tool here covers the kernel dispatch domain only.

#include "app_support.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace
{
constexpr const char* kAppTag = "synchronize-contexts-concurrent-app";

// A lane is a stream and a small allocation, so the ceiling is set by what a tool might ask for
// rather than by what any one of them uses.
constexpr uint64_t kMaxLanes      = 16;
constexpr size_t   kScratchFloats = 1024;

__global__ void
touch_kernel(float* ptr)
{
    if(threadIdx.x == 0) ptr[0] += 1.0F;
}

// Built and destroyed on the thread that uses it, keeping the stream paired with its enqueuer.
class stream_lane
{
public:
    stream_lane()
    {
        HIP_API_CALL(hipMalloc(&m_scratch, kScratchFloats * sizeof(float)));
        HIP_API_CALL(hipMemset(m_scratch, 0, kScratchFloats * sizeof(float)));
        HIP_API_CALL(hipStreamCreate(&m_stream));
    }

    ~stream_lane()
    {
        HIP_API_CALL(hipStreamDestroy(m_stream));
        HIP_API_CALL(hipFree(m_scratch));
    }

    stream_lane(const stream_lane&) = delete;
    stream_lane& operator=(const stream_lane&) = delete;

    void launch_kernel() const
    {
        touch_kernel<<<1, 64, 0, m_stream>>>(m_scratch);
        HIP_API_CALL(hipGetLastError());
    }

    void wait() const { HIP_API_CALL(hipStreamSynchronize(m_stream)); }

private:
    float*      m_scratch = nullptr;
    hipStream_t m_stream  = nullptr;
};

stream_lane*
lane_of(void* handle)
{
    return static_cast<stream_lane*>(handle);
}

// Every open builds its own stream, so the index selects nothing. It is checked against the
// advertised count to catch a tool asking for more concurrency than it declared.
void*
open_lane(uint64_t index)
{
    if(index >= kMaxLanes)
    {
        fprintf(stderr,
                "[%s] lane number %lu is outside the %lu this application advertises\n",
                kAppTag,
                static_cast<unsigned long>(index),
                static_cast<unsigned long>(kMaxLanes));
        exit(EXIT_FAILURE);
    }
    return new stream_lane{};
}

void
close_lane(void* handle)
{
    delete lane_of(handle);
}

void
launch_kernel(void* handle)
{
    lane_of(handle)->launch_kernel();
}

void
wait(void* handle)
{
    lane_of(handle)->wait();
}
}  // namespace

int
main(int, char**)
{
    // One lane built and dropped before the handshake: the first HIP call brings up the runtime,
    // and the runtime is what loads the tool.
    {
        auto _warmup = stream_lane{};
    }

    auto workload          = sync_test_workload_t{};
    workload.size          = sizeof(sync_test_workload_t);
    workload.lanes         = kMaxLanes;
    workload.open_lane     = &open_lane;
    workload.close_lane    = &close_lane;
    workload.launch_kernel = &launch_kernel;
    workload.launch_copy   = nullptr;
    workload.wait          = &wait;

    return sync_test_app::run_under_tool(workload, kAppTag);
}
