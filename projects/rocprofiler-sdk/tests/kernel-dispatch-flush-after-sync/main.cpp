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
// completes the dispatch, a few microseconds after the synchronize can return, so the flush has to
// wait for it. Each iteration launches one kernel, synchronizes its stream, flushes, and checks
// that the buffer callback has received that kernel's record. The process is its own tool, so the
// flush happens with no delay after the synchronize.
//
// Usage: kernel-dispatch-flush-after-sync [streams [iterations]]
//
// With several streams, each runs the loop on its own thread, so flushes overlap and the records
// they wait for are written out of order. Each stream has its own context and buffer: a flush of a
// buffer shared with another thread can return before the records that thread's flush swapped out
// reach the callback.

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>

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
int           iterations  = 200;
int           num_streams = 1;
constexpr int max_streams = 16;
constexpr int block_size  = 256;
constexpr int num_blocks  = (1 << 20) / block_size;

struct stream_state
{
    hipStream_t              stream    = nullptr;
    float*                   data      = nullptr;
    uint32_t                 grid_size = 0;
    rocprofiler_context_id_t context   = {0};
    rocprofiler_buffer_id_t  buffer    = {0};
    std::atomic<int>         records   = {0};
};

auto streams = std::array<stream_state, max_streams>{};

__global__ void
scale(float* data, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = (data[idx] * 1.0001F) + 1.0F;
}

void
launch(stream_state& state)
{
    scale<<<state.grid_size / block_size, block_size, 0, state.stream>>>(
        state.data, static_cast<int>(state.grid_size));
    HIP_CHECK(hipGetLastError());
}

void
buffer_callback(rocprofiler_context_id_t,
                rocprofiler_buffer_id_t,
                rocprofiler_record_header_t** headers,
                size_t                        num_headers,
                void*                         data,
                uint64_t)
{
    auto* state = static_cast<stream_state*>(data);
    for(size_t i = 0; i < num_headers; ++i)
    {
        if(headers[i]->category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
           headers[i]->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            continue;

        // Every context sees every stream's dispatches. They are told apart by grid size, since a
        // record's thread_id can name the thread of another stream sharing the hardware queue.
        const auto* record =
            static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(headers[i]->payload);
        if(record->dispatch_info.grid_size.x == state->grid_size) state->records.fetch_add(1);
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    for(int k = 0; k < num_streams; ++k)
    {
        auto& state = streams.at(k);
        ROCPROFILER_CHECK(rocprofiler_create_context(&state.context));
        ROCPROFILER_CHECK(rocprofiler_create_buffer(state.context,
                                                    64 * 1024,
                                                    32 * 1024,
                                                    ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                    buffer_callback,
                                                    &state,
                                                    &state.buffer));
        ROCPROFILER_CHECK(rocprofiler_configure_buffer_tracing_service(
            state.context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, state.buffer));
    }
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

// Returns the number of flushes that lacked the record of the kernel just completed on `state`.
int
run_stream(stream_state& state, std::atomic<int>& ready)
{
    // Start together so the flushes of different streams overlap.
    ready.fetch_add(1);
    while(ready.load() < num_streams)
        std::this_thread::yield();

    // A missed record still arrives with a later flush, so compare the running total: iteration
    // `i` has passed only if every one of the first `i + 1` records has been delivered.
    int missed = 0;
    for(int i = 0; i < iterations; ++i)
    {
        launch(state);
        HIP_CHECK(hipStreamSynchronize(state.stream));
        ROCPROFILER_CHECK(rocprofiler_flush_buffer(state.buffer));
        if(state.records.load() < i + 1) ++missed;
    }
    return missed;
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc > 1) num_streams = std::atoi(argv[1]);
    if(argc > 2) iterations = std::atoi(argv[2]);
    if(num_streams < 1 || num_streams > max_streams || iterations < 1)
    {
        fprintf(stderr, "usage: %s [streams (1-%d) [iterations]]\n", argv[0], max_streams);
        return EXIT_FAILURE;
    }

    for(int k = 0; k < num_streams; ++k)
        streams.at(k).grid_size = (num_blocks + k) * block_size;

    ROCPROFILER_CHECK(rocprofiler_force_configure(&configure));

    // Outside the contexts: loads the code object, so the first traced dispatch is like the rest.
    for(int k = 0; k < num_streams; ++k)
    {
        auto& state = streams.at(k);
        HIP_CHECK(hipStreamCreate(&state.stream));
        HIP_CHECK(hipMalloc(&state.data, state.grid_size * sizeof(float)));
        HIP_CHECK(hipMemsetAsync(state.data, 0, state.grid_size * sizeof(float), state.stream));
        launch(state);
        HIP_CHECK(hipStreamSynchronize(state.stream));
    }

    for(int k = 0; k < num_streams; ++k)
        ROCPROFILER_CHECK(rocprofiler_start_context(streams.at(k).context));

    auto ready   = std::atomic<int>{0};
    auto missed  = std::vector<int>(num_streams, 0);
    auto threads = std::vector<std::thread>{};
    for(int k = 0; k < num_streams; ++k)
        threads.emplace_back(
            [k, &ready, &missed]() { missed.at(k) = run_stream(streams.at(k), ready); });
    for(auto& thread : threads)
        thread.join();

    for(int k = 0; k < num_streams; ++k)
    {
        auto& state = streams.at(k);
        ROCPROFILER_CHECK(rocprofiler_stop_context(state.context));
        HIP_CHECK(hipFree(state.data));
        HIP_CHECK(hipStreamDestroy(state.stream));
    }

    const int total_missed = std::accumulate(missed.begin(), missed.end(), 0);
    const int total        = num_streams * iterations;
    if(total_missed > 0)
    {
        fprintf(stderr,
                "FAILED: %d of %d flushes issued right after a stream synchronize did not include "
                "the record of the kernel that had just completed on that stream (%d stream(s))\n",
                total_missed,
                total,
                num_streams);
        return EXIT_FAILURE;
    }

    printf("PASSED: all %d flushes included the record of the kernel that had just completed (%d "
           "stream(s))\n",
           total,
           num_streams);
    return EXIT_SUCCESS;
}
