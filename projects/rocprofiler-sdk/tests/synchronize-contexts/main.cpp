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

// Regression test for rocprofiler_synchronize_contexts.
//
// Reproduces the usage that exposed the gap: a tool that opens a short profiling window, runs one
// GPU operation, synchronizes on it, and reads its records immediately. Records for asynchronous
// operations are built on runtime-owned threads after the operation retires, so without the wait
// the flush returns an empty buffer and the record lands in the following window.
//
// The buffer is one buffer, 256 KiB with a 128 KiB watermark, lossless, carrying both kernel
// dispatch and memory copy. Every window is asserted, never an aggregate, so one late record
// fails the test rather than being absorbed by the window that follows.

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace
{
// Enough windows that passing without the wait is not a realistic outcome, while still running in
// a couple of seconds. Too few and an unloaded machine can pass with the wait removed.
constexpr uint64_t kWindows = 1000;

rocprofiler_context_id_t g_ctx    = {};
rocprofiler_buffer_id_t  g_buffer = {};

std::mutex g_mu;
struct delivered
{
    uint64_t                          window;
    rocprofiler_buffer_tracing_kind_t kind;
};
std::vector<delivered> g_delivered;  // launch window of each record, in delivery order

// Guards against a vacuous pass: if the buffer callback never ran at all, every window would look
// deficient for the wrong reason, and a future change that silently stops delivering would want to
// be told apart from one that delivers late.
std::atomic<bool> g_callback_ran{false};

#define CHECK_ROCP(expr)                                                                           \
    do                                                                                             \
    {                                                                                              \
        auto _status = (expr);                                                                     \
        if(_status != ROCPROFILER_STATUS_SUCCESS)                                                  \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[synchronize-contexts] %s:%d %s failed: %s\n",                                \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    #expr,                                                                         \
                    rocprofiler_get_status_string(_status));                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

#define CHECK_HIP(expr)                                                                            \
    do                                                                                             \
    {                                                                                              \
        auto _err = (expr);                                                                        \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[synchronize-contexts] %s:%d %s failed: %s\n",                                \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    #expr,                                                                         \
                    hipGetErrorString(_err));                                                      \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

void
buffer_callback(rocprofiler_context_id_t /*context*/,
                rocprofiler_buffer_id_t /*buffer_id*/,
                rocprofiler_record_header_t** headers,
                size_t                        num_headers,
                void* /*user_data*/,
                uint64_t /*drop_count*/)
{
    g_callback_ran.store(true);

    auto _lk = std::lock_guard<std::mutex>{g_mu};
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];
        if(header == nullptr || header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;

        if(header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            g_delivered.push_back(
                {static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload)
                     ->correlation_id.external.value,
                 ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH});
        else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
            g_delivered.push_back(
                {static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload)
                     ->correlation_id.external.value,
                 ROCPROFILER_BUFFER_TRACING_MEMORY_COPY});
    }
}

int
tool_init(rocprofiler_client_finalize_t /*fini_func*/, void* /*tool_data*/)
{
    CHECK_ROCP(rocprofiler_create_context(&g_ctx));
    CHECK_ROCP(rocprofiler_create_buffer(g_ctx,
                                         256 * 1024,
                                         128 * 1024,
                                         ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                         buffer_callback,
                                         nullptr,
                                         &g_buffer));
    CHECK_ROCP(rocprofiler_configure_buffer_tracing_service(
        g_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, g_buffer));
    CHECK_ROCP(rocprofiler_configure_buffer_tracing_service(
        g_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0, g_buffer));
    CHECK_ROCP(rocprofiler_start_context(g_ctx));
    return 0;
}

void
tool_fini(void* /*tool_data*/)
{}
}  // namespace

extern "C" __attribute__((visibility("default"))) rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /*version*/,
                      const char* /*runtime_version*/,
                      uint32_t /*priority*/,
                      rocprofiler_client_id_t* id)
{
    id->name         = "synchronize-contexts-test";
    static auto _cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &_cfg;
}

__global__ void
touch_kernel(float* ptr)
{
    if(threadIdx.x == 0) ptr[0] += 1.0F;
}

int
main(int, char**)
{
    rocprofiler_thread_id_t tid = 0;
    CHECK_ROCP(rocprofiler_get_thread_id(&tid));

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    CHECK_HIP(hipMalloc(&dev_a, 1024 * sizeof(float)));
    CHECK_HIP(hipMalloc(&dev_b, 1024 * sizeof(float)));
    CHECK_HIP(hipMemset(dev_a, 0, 1024 * sizeof(float)));

    // Pinned host memory, and large enough to be serviced by the copy engine. A small transfer,
    // or one between two device allocations, is serviced by a blit kernel instead: that emits a
    // dispatch record and leaves the memory copy path untested.
    constexpr size_t kCopyBytes  = 8 * 1024 * 1024;
    float*           host_pinned = nullptr;
    CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&host_pinned), kCopyBytes));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipMalloc(&dev_b, kCopyBytes));

    // Warm up the lazy paths (module load, signal pool, first touch) outside the measured windows,
    // then discard whatever they produced.
    for(int i = 0; i < 10; ++i)
        touch_kernel<<<1, 64>>>(dev_a);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_ROCP(rocprofiler_synchronize_contexts(&g_ctx, 1));
    CHECK_ROCP(rocprofiler_flush_buffer(g_buffer));
    {
        auto _lk = std::lock_guard<std::mutex>{g_mu};
        g_delivered.clear();
    }

    uint64_t deficient = 0;
    for(uint64_t window = 0; window < kWindows; ++window)
    {
        auto _ext  = rocprofiler_user_data_t{};
        _ext.value = window;
        CHECK_ROCP(rocprofiler_push_external_correlation_id(g_ctx, tid, _ext));

        // Alternate the two covered domains so a regression in either one fails this test.
        if(window % 2 == 0)
            touch_kernel<<<1, 64>>>(dev_a);
        else
            CHECK_HIP(
                hipMemcpyAsync(dev_b, host_pinned, kCopyBytes, hipMemcpyHostToDevice, nullptr));

        CHECK_HIP(hipDeviceSynchronize());

        // The behavior under test. Without it, the flush below routinely delivers nothing.
        CHECK_ROCP(rocprofiler_synchronize_contexts(&g_ctx, 1));
        CHECK_ROCP(rocprofiler_flush_buffer(g_buffer));

        // The kind is checked, not just the window: accepting either would let a run pass
        // without ever exercising the copy path, which a same-device copy may satisfy with a
        // blit kernel rather than an async copy.
        const auto expected_kind = (window % 2 == 0) ? ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH
                                                     : ROCPROFILER_BUFFER_TRACING_MEMORY_COPY;
        bool       found         = false;
        bool       wrong_kind    = false;
        {
            auto _lk = std::lock_guard<std::mutex>{g_mu};
            for(const auto& itr : g_delivered)
            {
                if(itr.window != window) continue;
                if(itr.kind == expected_kind)
                    found = true;
                else
                    wrong_kind = true;
            }
            g_delivered.clear();
        }

        if(!found && wrong_kind)
        {
            ++deficient;
            fprintf(stderr,
                    "[synchronize-contexts] window %lu received a record of the wrong kind; the "
                    "workload did not exercise the domain this window was testing\n",
                    static_cast<unsigned long>(window));
        }
        else if(!found)
        {
            ++deficient;
            fprintf(stderr,
                    "[synchronize-contexts] window %lu closed with no record for its own "
                    "operation\n",
                    static_cast<unsigned long>(window));
        }

        rocprofiler_user_data_t _popped = {};
        CHECK_ROCP(rocprofiler_pop_external_correlation_id(g_ctx, tid, &_popped));
    }

    CHECK_HIP(hipHostFree(host_pinned));
    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));

    int _exit_code = EXIT_SUCCESS;

    if(deficient > 0)
    {
        fprintf(stderr,
                "[synchronize-contexts] FAILED: %lu of %lu windows did not receive the record "
                "for an operation they had already synchronized on\n",
                static_cast<unsigned long>(deficient),
                static_cast<unsigned long>(kWindows));
        _exit_code = EXIT_FAILURE;
    }

    if(!g_callback_ran.load())
    {
        fprintf(stderr,
                "[synchronize-contexts] FAILED: the buffer callback never ran, so no record was "
                "delivered by any route and this run proves nothing about ordering\n");
        _exit_code = EXIT_FAILURE;
    }

    if(_exit_code == EXIT_SUCCESS)
        fprintf(stderr,
                "[synchronize-contexts] PASSED: %lu/%lu windows received their own record\n",
                static_cast<unsigned long>(kWindows),
                static_cast<unsigned long>(kWindows));

    return _exit_code;
}
