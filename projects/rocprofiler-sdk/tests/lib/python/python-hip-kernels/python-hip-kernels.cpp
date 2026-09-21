// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_KERNELS_PUBLIC_API __attribute__((visibility("default")))

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = (call);                                                                   \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[python-hip-kernels] HIP error %d (%s) at %s:%d\n",                           \
                    (int) err,                                                                     \
                    hipGetErrorString(err),                                                        \
                    __FILE__,                                                                      \
                    __LINE__);                                                                     \
            return -1;                                                                             \
        }                                                                                          \
    } while(0)

namespace
{
// Trivial kernel: each thread increments its element by 1
__global__ void
trivial_increment(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        data[idx] += 1.0f;
    }
}

constexpr int elements = 1024;
constexpr int threads  = 256;
constexpr int blocks   = (elements + threads - 1) / threads;

// Copy buffers are deliberately large: HIP routes tiny copies through an inline
// synchronous path that bypasses hsa_amd_memory_async_copy (so MEMORY_COPY tracing sees
// nothing). A multi-MB copy goes through the async DMA path that is traced.
constexpr int copy_elements = 4 * 1024 * 1024;  // 16 MiB of float

struct stream_state_t
{
    hipStream_t stream    = nullptr;
    float*      d_data    = nullptr;
    float*      d_copybuf = nullptr;  // large device buffer for H2D/D2H copies
    float*      h_copybuf = nullptr;  // large pageable host buffer for H2D/D2H copies
    bool        in_use    = false;
};

constexpr int  max_streams = 64;
stream_state_t streams[max_streams];
int            num_initialized = 0;
bool           hip_initialized = false;

void
ensure_hip_init()
{
    if(!hip_initialized)
    {
        auto err = hipSetDevice(0);
        (void) err;  // hipSetDevice may return an error if HIP is already initialized, but we can
                     // ignore it
        hip_initialized = true;
    }
}
}  // namespace

extern "C" {

/// Initialize num_streams HIP streams and allocate device memory for each.
/// Returns 0 on success, -1 on error.
HIP_KERNELS_PUBLIC_API int
hip_kernels_create_streams(int num_streams)
{
    ensure_hip_init();

    if(num_streams <= 0 || num_streams > max_streams)
    {
        fprintf(stderr,
                "[python-hip-kernels] Invalid num_streams=%d (max=%d)\n",
                num_streams,
                max_streams);
        return -1;
    }

    for(int i = 0; i < num_streams; ++i)
    {
        HIP_CHECK(hipStreamCreate(&streams[i].stream));
        HIP_CHECK(hipMalloc(&streams[i].d_data, elements * sizeof(float)));
        HIP_CHECK(hipMalloc(&streams[i].d_copybuf, copy_elements * sizeof(float)));
        // Pageable (not pinned) host memory: pinned H2D/D2H can take a fast path that
        // bypasses hsa_amd_memory_async_copy and is not captured by MEMORY_COPY tracing.
        streams[i].h_copybuf = static_cast<float*>(malloc(copy_elements * sizeof(float)));
        if(streams[i].h_copybuf == nullptr)
        {
            fprintf(stderr, "[python-hip-kernels] host malloc failed\n");
            return -1;
        }
        HIP_CHECK(
            hipMemsetAsync(streams[i].d_data, 0, elements * sizeof(float), streams[i].stream));
        streams[i].in_use = true;
    }
    num_initialized = num_streams;

    fprintf(stderr, "[python-hip-kernels] Created %d streams\n", num_streams);
    return 0;
}

/// Launch a trivial kernel on the given stream index.
/// Returns 0 on success, -1 on error.
HIP_KERNELS_PUBLIC_API int
hip_kernels_launch(int stream_idx)
{
    if(stream_idx < 0 || stream_idx >= num_initialized || !streams[stream_idx].in_use)
    {
        fprintf(stderr,
                "[python-hip-kernels] Invalid stream_idx=%d (initialized=%d)\n",
                stream_idx,
                num_initialized);
        return -1;
    }

    trivial_increment<<<blocks, threads, 0, streams[stream_idx].stream>>>(
        streams[stream_idx].d_data, elements);

    return 0;
}

/// Issue an asynchronous host-to-device then device-to-host copy on the given stream.
/// Drives the HIP memory-copy API so MEMORY_COPY_* services produce records.
/// Returns 0 on success, -1 on error.
HIP_KERNELS_PUBLIC_API int
hip_kernels_memcpy(int stream_idx)
{
    if(stream_idx < 0 || stream_idx >= num_initialized || !streams[stream_idx].in_use)
    {
        fprintf(stderr,
                "[python-hip-kernels] Invalid stream_idx=%d (initialized=%d)\n",
                stream_idx,
                num_initialized);
        return -1;
    }

    auto& s = streams[stream_idx];
    HIP_CHECK(hipMemcpyAsync(
        s.d_copybuf, s.h_copybuf, copy_elements * sizeof(float), hipMemcpyHostToDevice, s.stream));
    HIP_CHECK(hipMemcpyAsync(
        s.h_copybuf, s.d_copybuf, copy_elements * sizeof(float), hipMemcpyDeviceToHost, s.stream));

    return 0;
}

/// Allocate then free one device buffer. Drives the HIP memory-allocation API so
/// MEMORY_ALLOCATION_* services produce records (one allocate + one free per call).
/// Returns 0 on success, -1 on error.
HIP_KERNELS_PUBLIC_API int
hip_kernels_alloc_free()
{
    ensure_hip_init();

    void* ptr = nullptr;
    HIP_CHECK(hipMalloc(&ptr, elements * sizeof(float)));
    HIP_CHECK(hipFree(ptr));

    return 0;
}

/// Emit one roctx marker range (push + mark + pop). Drives the marker/roctx API so
/// MARKER_API_* services produce records. Returns 0 (always succeeds).
HIP_KERNELS_PUBLIC_API int
hip_kernels_marker()
{
    roctxRangePushA("python-hip-kernels-range");
    roctxMarkA("python-hip-kernels-mark");
    roctxRangePop();
    return 0;
}

/// Synchronize all initialized streams. Returns 0 on success, -1 on error.
HIP_KERNELS_PUBLIC_API int
hip_kernels_synchronize()
{
    for(int i = 0; i < num_initialized; ++i)
    {
        if(streams[i].in_use)
        {
            HIP_CHECK(hipStreamSynchronize(streams[i].stream));
        }
    }
    return 0;
}

/// Destroy all initialized streams and free device memory. Returns 0 on success, -1 on error.
HIP_KERNELS_PUBLIC_API int
hip_kernels_destroy_streams()
{
    for(int i = 0; i < num_initialized; ++i)
    {
        if(streams[i].in_use)
        {
            HIP_CHECK(hipFree(streams[i].d_data));
            HIP_CHECK(hipFree(streams[i].d_copybuf));
            free(streams[i].h_copybuf);
            HIP_CHECK(hipStreamDestroy(streams[i].stream));
            streams[i].d_data    = nullptr;
            streams[i].d_copybuf = nullptr;
            streams[i].h_copybuf = nullptr;
            streams[i].stream    = nullptr;
            streams[i].in_use    = false;
        }
    }
    num_initialized = 0;

    fprintf(stderr, "[python-hip-kernels] Destroyed all streams\n");
    return 0;
}

}  // extern "C"
