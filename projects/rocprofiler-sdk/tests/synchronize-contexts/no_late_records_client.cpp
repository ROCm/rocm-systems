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

// rocprofiler_synchronize_context as the closing act of a profiling window.
//
// The other tools ask whether a window's own records arrive before it closes. This one asks the
// complementary question, which no count of arrivals can answer: that nothing arrives after. A
// record landing once the window has closed describes work the tool already reported on.
//
// All three routes are enabled on the one context, since a drain reaching two and missing the
// third is the defect being looked for.
//
// The context is deliberately left running across the idle stretch. Stopping it first is what a
// real tool does, but it would also let the runtime discard the in-flight record this is hunting.

#include "counter_config.hpp"
#include "sync_test_workload.h"
#include "test_support.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/synchronize_contexts.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace
{
constexpr const char* kTestTag = "synchronize-contexts";

// Every operation outstanding when the window closes is a separate chance to leak one record.
constexpr uint64_t kOperationPairs = 50;

// How long the run sits idle after closing its window. Insurance, not a measurement: a consumer
// thread that merely lags has caught up many times over by the end of it.
constexpr auto kSettleDuration = std::chrono::seconds{5};

// Counted per dispatch on every supported agent: the cheapest counter that reliably produces one
// record per dispatch.
constexpr const char* kCounterName = "SQ_WAVES";

using sync_test::make_options;
using sync_test::make_result;
using sync_test::scoped_lane;

// Tool: one lossless buffer carrying every route, so one drain has to reach all of them.

rocprofiler_context_id_t g_context = {};
rocprofiler_buffer_id_t  g_buffer  = {};

struct delivery_counts
{
    uint64_t kernel_dispatch = 0;
    uint64_t memory_copy     = 0;
    uint64_t counter         = 0;

    uint64_t total() const { return kernel_dispatch + memory_copy + counter; }
};

std::mutex      g_counts_mutex;
delivery_counts g_inside_window;
delivery_counts g_after_window;

// Everything delivered once this is set is late: no further GPU work is issued.
std::atomic<bool> g_window_closed{false};

void
buffer_callback(rocprofiler_context_id_t /*context*/,
                rocprofiler_buffer_id_t /*buffer_id*/,
                rocprofiler_record_header_t** headers,
                size_t                        num_headers,
                void* /*user_data*/,
                uint64_t /*drop_count*/)
{
    const auto closed = g_window_closed.load();

    auto  _lk   = std::lock_guard<std::mutex>{g_counts_mutex};
    auto& tally = closed ? g_after_window : g_inside_window;

    for(size_t i = 0; i < num_headers; ++i)
    {
        const auto* header = headers[i];
        if(header == nullptr) continue;

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING)
        {
            if(header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
                ++tally.kernel_dispatch;
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
                ++tally.memory_copy;
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
                header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
        {
            ++tally.counter;
        }
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
tool_init(rocprofiler_client_finalize_t /*fini_func*/, void* /*tool_data*/)
{
    ROCPROFILER_CHECK(rocprofiler_create_context(&g_context));
    ROCPROFILER_CHECK(rocprofiler_create_buffer(g_context,
                                                512 * 1024,
                                                256 * 1024,
                                                ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                buffer_callback,
                                                nullptr,
                                                &g_buffer));
    ROCPROFILER_CHECK(rocprofiler_configure_buffer_tracing_service(
        g_context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, g_buffer));
    ROCPROFILER_CHECK(rocprofiler_configure_buffer_tracing_service(
        g_context, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0, g_buffer));
    ROCPROFILER_CHECK(rocprofiler_configure_buffer_dispatch_counting_service(
        g_context, g_buffer, dispatch_callback, nullptr));
    ROCPROFILER_CHECK(rocprofiler_start_context(g_context));
    return 0;
}

void
tool_fini(void* /*tool_data*/)
{}

// Bookkeeping

delivery_counts
read_counts(const delivery_counts& counts)
{
    auto _lk = std::lock_guard<std::mutex>{g_counts_mutex};
    return counts;
}

void
reset_counts()
{
    auto _lk        = std::lock_guard<std::mutex>{g_counts_mutex};
    g_inside_window = delivery_counts{};
    g_after_window  = delivery_counts{};
}

// Report one route that came up short and evaluate to 1. Without these the test passes on a build
// delivering nothing: no records means no late records, and silence is what it looks for.
uint64_t
require_at_least(const char* route, uint64_t delivered, uint64_t expected)
{
    if(delivered >= expected) return 0;

    fprintf(stderr,
            "[%s] the window closed having received %lu of %lu %s records, so this run has "
            "nothing to say about what arrives afterwards\n",
            kTestTag,
            static_cast<unsigned long>(delivered),
            static_cast<unsigned long>(expected),
            route);
    return 1;
}

// Open a window, run a batch, close the window, then go quiet.

uint64_t
nothing_arrives_after_the_window_closes(const scoped_lane& lane)
{
    auto failures = uint64_t{0};

    for(uint64_t i = 0; i < kOperationPairs; ++i)
    {
        lane.launch_kernel();
        lane.launch_copy();
    }

    // A tool reporting on a finished phase waits for it first, leaving the synchronize to drain
    // record production and delivery rather than the operations themselves.
    lane.wait();

    auto options = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NONE);
    auto result  = make_result();
    ROCPROFILER_CHECK(rocprofiler_synchronize_context(g_context, &options, &result));

    // Closed here and never reopened. A record delivered at this exact instant may land on either
    // side of the flip, which can only under-report, never fail a build that drained.
    g_window_closed.store(true);

    // Success means the drain reported nothing outstanding, so a nonzero count is the status and
    // the result disagreeing.
    if(result.num_timeout_records != 0)
    {
        ++failures;
        fprintf(stderr,
                "[%s] synchronize succeeded but reported %lu records still outstanding\n",
                kTestTag,
                static_cast<unsigned long>(result.num_timeout_records));
    }

    const auto inside = read_counts(g_inside_window);
    failures += require_at_least("kernel dispatch", inside.kernel_dispatch, kOperationPairs);
    failures += require_at_least("memory copy", inside.memory_copy, kOperationPairs);
    failures += require_at_least("dispatch counter", inside.counter, kOperationPairs);

    std::this_thread::sleep_for(kSettleDuration);

    // A record written to the buffer after the synchronize flushed it is still sitting there, so
    // the idle stretch alone would not reveal it.
    ROCPROFILER_CHECK(rocprofiler_flush_buffer(g_buffer));

    const auto late = read_counts(g_after_window);
    if(late.total() == 0) return failures;

    ++failures;
    fprintf(stderr,
            "[%s] %lu records arrived after the window closed (%lu kernel dispatch, %lu memory "
            "copy, %lu dispatch counter), so the synchronize returned while the context still had "
            "records to produce\n",
            kTestTag,
            static_cast<unsigned long>(late.total()),
            static_cast<unsigned long>(late.kernel_dispatch),
            static_cast<unsigned long>(late.memory_copy),
            static_cast<unsigned long>(late.counter));
    return failures;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /*version*/,
                      const char* /*runtime_version*/,
                      uint32_t /*priority*/,
                      rocprofiler_client_id_t* id)
{
    id->name         = "synchronize-contexts-no-late-records";
    static auto _cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &_cfg;
}

extern "C" __attribute__((visibility("default"))) uint64_t
sync_test_run(const sync_test_workload_t* workload)
{
    if(!sync_test::workload_is_usable(
           workload, /* required_lanes = */ 1, /* requires_copy = */ true, kTestTag))
        return 1;

    auto lane = scoped_lane{*workload};

    // Warm up the lazy paths -- module load, signal pool, the counter configuration built on the
    // first dispatch -- then start from a known-empty slate.
    for(int i = 0; i < 10; ++i)
        lane.launch_kernel();
    lane.launch_copy();
    lane.wait();
    ROCPROFILER_CHECK(rocprofiler_synchronize_context(g_context, nullptr, nullptr));
    reset_counts();

    const auto failures = nothing_arrives_after_the_window_closes(lane);

    if(failures > 0)
    {
        fprintf(stderr,
                "[%s] FAILED: %lu failed expectations\n",
                kTestTag,
                static_cast<unsigned long>(failures));
        return failures;
    }

    fprintf(stdout,
            "[%s] PASSED: a window of %lu dispatches and %lu copies closed with every record "
            "delivered, and nothing arrived in the %lu seconds after\n",
            kTestTag,
            static_cast<unsigned long>(kOperationPairs),
            static_cast<unsigned long>(kOperationPairs),
            static_cast<unsigned long>(kSettleDuration.count()));
    return 0;
}
