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

// A single-threaded application for the rocprofiler_synchronize_context tools: one lane on the
// default stream, offering kernel dispatches and host-to-device copies. The workload behind every
// tool here except concurrent-windows.
//
// The copy source is pinned host memory and large enough that the runtime uses the copy engine. A
// small copy, or one between two device allocations, is serviced by a blit kernel instead, which
// emits a dispatch record and leaves the memory copy domain untested while appearing to pass.

#include "app_support.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace
{
constexpr const char* kAppTag = "synchronize-contexts-serial-app";

constexpr size_t kCopyBytes     = 8 * 1024 * 1024;
constexpr size_t kScratchFloats = 1024;

__global__ void
touch_kernel(float* ptr)
{
    if(threadIdx.x == 0) ptr[0] += 1.0F;
}

// The one lane, on the default stream. Its allocations outlive any single open/close, which a
// tool performs many times over a run.
class default_stream_lane
{
public:
    default_stream_lane()
    {
        HIP_API_CALL(hipMalloc(&m_scratch, kScratchFloats * sizeof(float)));
        HIP_API_CALL(hipMemset(m_scratch, 0, kScratchFloats * sizeof(float)));
        HIP_API_CALL(hipMalloc(&m_copy_dst, kCopyBytes));
        HIP_API_CALL(hipHostMalloc(reinterpret_cast<void**>(&m_copy_src), kCopyBytes));
    }

    ~default_stream_lane()
    {
        HIP_API_CALL(hipHostFree(m_copy_src));
        HIP_API_CALL(hipFree(m_copy_dst));
        HIP_API_CALL(hipFree(m_scratch));
    }

    default_stream_lane(const default_stream_lane&) = delete;
    default_stream_lane& operator=(const default_stream_lane&) = delete;

    void launch_kernel() const
    {
        touch_kernel<<<1, 64>>>(m_scratch);
        HIP_API_CALL(hipGetLastError());
    }

    void launch_copy() const
    {
        HIP_API_CALL(
            hipMemcpyAsync(m_copy_dst, m_copy_src, kCopyBytes, hipMemcpyHostToDevice, nullptr));
    }

    static void wait() { HIP_API_CALL(hipDeviceSynchronize()); }

private:
    float* m_scratch  = nullptr;
    float* m_copy_dst = nullptr;
    float* m_copy_src = nullptr;
};

default_stream_lane*
lane_of(void* handle)
{
    return static_cast<default_stream_lane*>(handle);
}

// Held for the life of the process. Opening and closing it is bookkeeping: the default stream is
// process-wide and there is nothing per-open to build.
default_stream_lane* g_lane = nullptr;

void* open_lane(uint64_t) { return g_lane; }

void
close_lane(void*)
{}

void
launch_kernel(void* handle)
{
    lane_of(handle)->launch_kernel();
}

void
launch_copy(void* handle)
{
    lane_of(handle)->launch_copy();
}

void
wait(void*)
{
    default_stream_lane::wait();
}
}  // namespace

int
main(int, char**)
{
    // Built before the handshake: the first HIP call brings up the runtime, and the runtime is
    // what loads the tool.
    auto lane = default_stream_lane{};
    g_lane    = &lane;

    auto workload          = sync_test_workload_t{};
    workload.size          = sizeof(sync_test_workload_t);
    workload.lanes         = 1;
    workload.open_lane     = &open_lane;
    workload.close_lane    = &close_lane;
    workload.launch_kernel = &launch_kernel;
    workload.launch_copy   = &launch_copy;
    workload.wait          = &wait;

    return sync_test_app::run_under_tool(workload, kAppTag);
}
