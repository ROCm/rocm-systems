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

// rocprofiler_synchronize_context, over dispatch counter collection records.
//
// Counter records take a different route out of the runtime than tracing records: a dispatch's
// counter sample is handed to a consumer thread and written there, later than the dispatch record
// tracing produces. It then turns up in the following window. Covering the tracing domains says
// nothing about whether the wait reaches this path, hence a separate test.
//
//   every_window_sees_its_counter_records   each window gets a counter record for every dispatch
//                                           it issued, before the window closes.
//   noflush_still_waits_for_records         NOFLUSH drops the flush but keeps the wait, so a
//                                           later explicit flush produces all of them.
//
// Flag and argument validation common to the API is covered once, in tracing_window.cpp. Every
// window is asserted on its own and never in aggregate: a late record still arrives one window
// later, so totals reconcile even when every window came up short.

#include "counter_config.hpp"
#include "test_support.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/synchronize_contexts.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace
{
constexpr const char* kTestTag = "counter-window";

// A single window is only sometimes short, so a small count passes against a broken build.
constexpr uint64_t kWindows             = 1000;
constexpr uint64_t kDispatchesPerWindow = 2;

// Counted per dispatch on every supported agent, so it is the cheapest counter that reliably
// produces one record per dispatch.
constexpr const char* kCounterName = "SQ_WAVES";

using sync_test::make_options;
using sync_test::make_result;

// Tool

rocprofiler_context_id_t g_context = {};
rocprofiler_buffer_id_t  g_buffer  = {};

// Reset at the top of each window; the window's verdict is what it holds when the window closes.
std::atomic<uint64_t> g_records_this_window{0};
std::atomic<uint64_t> g_records_total{0};

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
        const auto* header = headers[i];
        if(header == nullptr) continue;
        if(header->category != ROCPROFILER_BUFFER_CATEGORY_COUNTERS) continue;
        if(header->kind != ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER) continue;

        g_records_this_window.fetch_add(1, std::memory_order_relaxed);
        g_records_total.fetch_add(1, std::memory_order_relaxed);
    }
}

void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t data,
                  rocprofiler_counter_config_id_t*             config,
                  rocprofiler_user_data_t*,
                  void*)
{
    *config =
        sync_test::counter_config_for_agent(data.dispatch_info.agent_id, kCounterName, kTestTag);
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    CHECK_ROCP(rocprofiler_create_context(&g_context));
    CHECK_ROCP(rocprofiler_create_buffer(g_context,
                                         512 * 1024,
                                         256 * 1024,
                                         ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                         buffer_callback,
                                         nullptr,
                                         &g_buffer));
    CHECK_ROCP(rocprofiler_configure_buffer_dispatch_counting_service(
        g_context, g_buffer, dispatch_callback, nullptr));
    CHECK_ROCP(rocprofiler_start_context(g_context));
    return 0;
}

void
tool_fini(void*)
{}

// Workload

__global__ void
touch_kernel(float* ptr)
{
    ptr[threadIdx.x] = static_cast<float>(threadIdx.x);
}

void
launch_window_dispatches(float* scratch)
{
    for(uint64_t i = 0; i < kDispatchesPerWindow; ++i)
    {
        hipLaunchKernelGGL(touch_kernel, dim3(1), dim3(64), 0, nullptr, scratch);
        CHECK_HIP(hipGetLastError());
    }
    CHECK_HIP(hipDeviceSynchronize());
}

// Checks. Each returns a count of failed expectations.

// The regression guard. The synchronize flushes as part of the call, so the window deliberately
// does not flush the buffer itself -- that would mask a regression in half the contract.
uint64_t
every_window_sees_its_counter_records(float* scratch)
{
    auto short_windows          = uint64_t{0};
    auto first_short_window     = uint64_t{0};
    auto first_short_window_got = uint64_t{0};

    for(uint64_t window = 0; window < kWindows; ++window)
    {
        g_records_this_window.store(0, std::memory_order_relaxed);
        launch_window_dispatches(scratch);

        auto options = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NONE);
        auto result  = make_result();
        CHECK_ROCP(rocprofiler_synchronize_context(g_context, &options, &result));

        const auto delivered = g_records_this_window.load(std::memory_order_relaxed);
        if(delivered < kDispatchesPerWindow)
        {
            if(short_windows == 0)
            {
                first_short_window     = window;
                first_short_window_got = delivered;
            }
            ++short_windows;
        }
    }

    if(short_windows == 0) return 0;

    fprintf(stderr,
            "[%s] %lu of %lu windows were short; the first was window %lu with %lu of %lu "
            "records\n",
            kTestTag,
            static_cast<unsigned long>(short_windows),
            static_cast<unsigned long>(kWindows),
            static_cast<unsigned long>(first_short_window),
            static_cast<unsigned long>(first_short_window_got),
            static_cast<unsigned long>(kDispatchesPerWindow));
    return short_windows;
}

// NOFLUSH drops the flush but keeps the wait, isolating the half this test cares about: if the
// wait reaches the counter consumer thread the records exist on return, so a later explicit flush
// produces all of them.
//
// Deliberately not asserted: that the buffer is still unflushed on return. NOFLUSH promises only
// that this call will not flush, not that nothing else will, so requiring an empty buffer would
// be testing a race. The count is read only after the flush, which covers both sides of it.
uint64_t
noflush_still_waits_for_records(float* scratch)
{
    g_records_this_window.store(0, std::memory_order_relaxed);
    launch_window_dispatches(scratch);

    auto options = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NOFLUSH);
    auto result  = make_result();
    CHECK_ROCP(rocprofiler_synchronize_context(g_context, &options, &result));
    CHECK_ROCP(rocprofiler_flush_buffer(g_buffer));

    const auto delivered = g_records_this_window.load(std::memory_order_relaxed);
    if(delivered >= kDispatchesPerWindow) return 0;

    fprintf(stderr,
            "[%s] a NOFLUSH synchronize followed by an explicit flush produced %lu of %lu "
            "records, so the wait did not reach the counter consumer thread\n",
            kTestTag,
            static_cast<unsigned long>(delivered),
            static_cast<unsigned long>(kDispatchesPerWindow));
    return 1;
}
}  // namespace

extern "C" __attribute__((visibility("default"))) rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    id->name         = "synchronize-contexts-counter-window";
    static auto _cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &_cfg;
}

int
main()
{
    float* scratch = nullptr;
    CHECK_HIP(hipMalloc(&scratch, 64 * sizeof(float)));

    auto failures = uint64_t{0};
    failures += every_window_sees_its_counter_records(scratch);
    failures += noflush_still_waits_for_records(scratch);

    CHECK_HIP(hipFree(scratch));

    // A run producing no counter records at all reports every window short for a reason that has
    // nothing to do with ordering.
    if(g_records_total.load() == 0)
    {
        fprintf(stderr,
                "[%s] FAILED: no counter records were delivered at all, so this run proves "
                "nothing about ordering\n",
                kTestTag);
        return EXIT_FAILURE;
    }

    if(failures > 0)
    {
        fprintf(stderr,
                "[%s] FAILED: %lu failed expectations\n",
                kTestTag,
                static_cast<unsigned long>(failures));
        return EXIT_FAILURE;
    }

    fprintf(stdout,
            "[%s] PASSED: %lu windows, %lu counter records, and NOFLUSH behaved as documented\n",
            kTestTag,
            static_cast<unsigned long>(kWindows),
            static_cast<unsigned long>(g_records_total.load()));
    return EXIT_SUCCESS;
}
