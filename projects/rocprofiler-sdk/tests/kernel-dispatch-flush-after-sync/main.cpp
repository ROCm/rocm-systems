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

// A kernel-dispatch record must be in the rocprofiler_flush_buffer() the application issues right
// after it has seen the kernel finish. The SDK writes the record on its own threads once the GPU
// completes the dispatch, a few microseconds after hipDeviceSynchronize() can return, so the flush
// has to wait for it. Each iteration launches one kernel, synchronizes, flushes, and checks that
// the buffer callback has received that kernel's record. The process is its own tool, so the
// flush happens with no delay after the synchronize.

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

#define ROCPROFILER_CHECK(CALL)                                                                    \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t status_ = (CALL);                                                     \
        if(status_ != ROCPROFILER_STATUS_SUCCESS)                                                  \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d %s failed: %s\n",                                                       \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    #CALL,                                                                         \
                    rocprofiler_get_status_string(status_));                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

namespace
{
// A flush that does not wait for the record misses it in nearly every iteration, so this many
// cannot pass by chance.
constexpr int iterations   = 200;
constexpr int num_elements = 1 << 20;
constexpr int block_size   = 256;

auto context = rocprofiler_context_id_t{0};
auto buffer  = rocprofiler_buffer_id_t{0};
auto records = std::atomic<int>{0};

__global__ void
scale(float* data, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = (data[idx] * 1.0001F) + 1.0F;
}

void
buffer_callback(rocprofiler_context_id_t,
                rocprofiler_buffer_id_t,
                rocprofiler_record_header_t** headers,
                size_t                        num_headers,
                void*,
                uint64_t)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        if(headers[i]->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
           headers[i]->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            records.fetch_add(1);
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    ROCPROFILER_CHECK(rocprofiler_create_context(&context));
    ROCPROFILER_CHECK(rocprofiler_create_buffer(context,
                                                64 * 1024,
                                                32 * 1024,
                                                ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                buffer_callback,
                                                nullptr,
                                                &buffer));
    ROCPROFILER_CHECK(rocprofiler_configure_buffer_tracing_service(
        context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, buffer));
    return 0;
}

void
tool_fini(void*)
{}

rocprofiler_tool_configure_result_t*
configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    id->name         = "kernel-dispatch-flush-after-sync";
    static auto _cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &_cfg;
}
}  // namespace

int
main()
{
    ROCPROFILER_CHECK(rocprofiler_force_configure(&configure));

    float* data = nullptr;
    HIP_CHECK(hipMalloc(&data, num_elements * sizeof(float)));
    HIP_CHECK(hipMemset(data, 0, num_elements * sizeof(float)));

    // Outside the context: loads the code object, so the first traced dispatch is like the rest.
    scale<<<num_elements / block_size, block_size>>>(data, num_elements);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    ROCPROFILER_CHECK(rocprofiler_start_context(context));

    // A missed record still arrives with a later flush, so compare the running total: iteration
    // `i` has passed only if every one of the first `i + 1` records has been delivered.
    int missed = 0;
    for(int i = 0; i < iterations; ++i)
    {
        scale<<<num_elements / block_size, block_size>>>(data, num_elements);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ROCPROFILER_CHECK(rocprofiler_flush_buffer(buffer));
        if(records.load() < i + 1) ++missed;
    }

    ROCPROFILER_CHECK(rocprofiler_stop_context(context));
    HIP_CHECK(hipFree(data));

    if(missed > 0)
    {
        fprintf(stderr,
                "FAILED: %d of %d flushes issued right after hipDeviceSynchronize() did not "
                "include the record of the kernel that had just completed\n",
                missed,
                iterations);
        return EXIT_FAILURE;
    }

    printf("PASSED: all %d flushes included the record of the kernel that had just completed\n",
           iterations);
    return EXIT_SUCCESS;
}
