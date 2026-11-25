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

// rocprofiler_synchronize_context with dispatches in flight on other threads. The sibling window
// tools are single-threaded, so nothing else covers the drain's cross-thread behaviour.
//
//   each_thread_sees_its_own_dispatch_record  threads synchronize concurrently; each requires its
//                                             own dispatch's record inside its own window.
//   one_sync_after_join_sees_every_record     all threads joined, then one synchronize must
//                                             account for every dispatch.
//
// Nothing is asserted about a dispatch still being enqueued elsewhere when the call begins: that
// work is concurrent with the synchronize, not ordered before it. Each check establishes a
// happens-before edge first -- a device wait, or a join -- and asserts only across it.
//
// Each thread takes a lane of its own. Lanes sharing a stream would serialize the enqueues and
// remove the contention this exists to create, hence the multi-lane application and the checked
// lane count.
//
// Argument validation common to the API is covered in tracing_window_client.cpp.

#include "sync_test_workload.h"
#include "test_support.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/experimental/synchronize_contexts.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
constexpr const char* kTestTag = "synchronize-contexts";

// Reported, not required: the tool runs under both routes, so this only tells the log which one.
constexpr const char* kSignalLessEnvVar = "ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS";

constexpr uint64_t kThreads             = 4;
constexpr uint64_t kWindowsPerThread    = 100;
constexpr uint64_t kDispatchesPerThread = 50;

// Tag 0 means untagged, so both ranges start at 1 and never overlap.
constexpr uint64_t kWindowTagBase = 1;
constexpr uint64_t kJoinTagBase   = kWindowTagBase + (kThreads * kWindowsPerThread);
constexpr uint64_t kTagSpace      = kJoinTagBase + (kThreads * kDispatchesPerThread);

using sync_test::make_options;
using sync_test::make_result;
using sync_test::scoped_lane;

rocprofiler_context_id_t g_context = {};
rocprofiler_buffer_id_t  g_buffer  = {};

// One counter per tag, never cleared: threads read their own tags while others are still written.
std::array<std::atomic<uint64_t>, kTagSpace> g_delivered{};

// Guards against a vacuous pass: a callback that never ran fails every check for the wrong reason.
std::atomic<bool> g_callback_ran{false};

// Missing at window close, re-checked after the drain: a record present by then was late rather
// than lost, and the failure message should say which.
std::mutex            g_late_mutex;
std::vector<uint64_t> g_missed_at_window_close;

void
buffer_callback(rocprofiler_context_id_t /*context*/,
                rocprofiler_buffer_id_t /*buffer_id*/,
                rocprofiler_record_header_t** headers,
                size_t                        num_headers,
                void* /*user_data*/,
                uint64_t /*drop_count*/)
{
    g_callback_ran.store(true);

    for(size_t i = 0; i < num_headers; ++i)
    {
        const auto* header = headers[i];
        if(header == nullptr || header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;
        if(header->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) continue;

        const auto* record =
            static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
        const auto tag = record->correlation_id.external.value;
        if(tag > 0 && tag < kTagSpace) g_delivered[tag].fetch_add(1, std::memory_order_relaxed);
    }
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
    ROCPROFILER_CHECK(rocprofiler_start_context(g_context));
    return 0;
}

void
tool_fini(void* /*tool_data*/)
{}

// Tagging. Every thread pushes against its own thread id, so the tags do not interfere.

void
push_tag(uint64_t tag)
{
    auto thread_id = rocprofiler_thread_id_t{0};
    ROCPROFILER_CHECK(rocprofiler_get_thread_id(&thread_id));

    auto external  = rocprofiler_user_data_t{};
    external.value = tag;
    ROCPROFILER_CHECK(rocprofiler_push_external_correlation_id(g_context, thread_id, external));
}

void
pop_tag()
{
    auto thread_id = rocprofiler_thread_id_t{0};
    ROCPROFILER_CHECK(rocprofiler_get_thread_id(&thread_id));

    auto popped = rocprofiler_user_data_t{};
    ROCPROFILER_CHECK(rocprofiler_pop_external_correlation_id(g_context, thread_id, &popped));
}

uint64_t
delivered(uint64_t tag)
{
    return g_delivered[tag].load(std::memory_order_relaxed);
}

// Checks. Each returns a count of failed expectations, so one run reports every problem.

// The cross-thread regression guard. Each thread waits for its own dispatch before synchronizing,
// so its record is owed; the others are mid-enqueue throughout, which is the contention.
uint64_t
each_thread_sees_its_own_dispatch_record(const sync_test_workload_t& workload)
{
    auto failures = std::atomic<uint64_t>{0};
    auto threads  = std::vector<std::thread>{};
    threads.reserve(kThreads);

    for(uint64_t t = 0; t < kThreads; ++t)
    {
        threads.emplace_back([t, &failures, &workload]() {
            // Opened on this thread, so the stream and the thread enqueuing onto it stay paired.
            auto lane = scoped_lane{workload, t};

            for(uint64_t window = 0; window < kWindowsPerThread; ++window)
            {
                const auto tag = kWindowTagBase + (t * kWindowsPerThread) + window;

                push_tag(tag);
                lane.launch_kernel();
                lane.wait();

                auto       options = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NONE);
                auto       result  = make_result();
                const auto status  = rocprofiler_synchronize_context(g_context, &options, &result);
                pop_tag();

                if(status != ROCPROFILER_STATUS_SUCCESS)
                {
                    failures.fetch_add(1);
                    fprintf(stderr,
                            "[%s] thread %lu window %lu: synchronize returned %s\n",
                            kTestTag,
                            static_cast<unsigned long>(t),
                            static_cast<unsigned long>(window),
                            rocprofiler_get_status_string(status));
                    continue;
                }

                if(delivered(tag) == 0)
                {
                    failures.fetch_add(1);
                    {
                        auto _lk = std::lock_guard<std::mutex>{g_late_mutex};
                        g_missed_at_window_close.push_back(tag);
                    }
                    // A peek of zero means the drain never saw the record as outstanding, which
                    // is a different defect from a wait that gave up on it.
                    fprintf(stderr,
                            "[%s] thread %lu window %lu: closed with no record for its own "
                            "completed dispatch (synchronize reported num_records=%lu "
                            "num_timeout_records=%lu)\n",
                            kTestTag,
                            static_cast<unsigned long>(t),
                            static_cast<unsigned long>(window),
                            static_cast<unsigned long>(result.num_records),
                            static_cast<unsigned long>(result.num_timeout_records));
                }
            }
        });
    }

    for(auto& itr : threads)
        itr.join();

    return failures.load();
}

// The completeness half. Joining the threads orders all of their work before the call, so one
// synchronize owes a record for every dispatch any of them issued.
uint64_t
one_sync_after_join_sees_every_record(const sync_test_workload_t& workload)
{
    auto threads = std::vector<std::thread>{};
    threads.reserve(kThreads);

    for(uint64_t t = 0; t < kThreads; ++t)
    {
        threads.emplace_back([t, &workload]() {
            auto lane = scoped_lane{workload, t};

            for(uint64_t i = 0; i < kDispatchesPerThread; ++i)
            {
                push_tag(kJoinTagBase + (t * kDispatchesPerThread) + i);
                lane.launch_kernel();
                pop_tag();
            }
            lane.wait();
        });
    }

    for(auto& itr : threads)
        itr.join();

    auto options  = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NONE);
    auto result   = make_result();
    auto failures = EXPECT_STATUS(ROCPROFILER_STATUS_SUCCESS,
                                  rocprofiler_synchronize_context(g_context, &options, &result));

    auto missing = uint64_t{0};
    for(uint64_t tag = kJoinTagBase; tag < kTagSpace; ++tag)
        if(delivered(tag) == 0) ++missing;

    if(missing > 0)
    {
        ++failures;
        fprintf(stderr,
                "[%s] one synchronize after joining %lu threads left %lu of %lu dispatch records "
                "undelivered\n",
                kTestTag,
                static_cast<unsigned long>(kThreads),
                static_cast<unsigned long>(missing),
                static_cast<unsigned long>(kThreads * kDispatchesPerThread));
    }

    return failures;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /*version*/,
                      const char* /*runtime_version*/,
                      uint32_t /*priority*/,
                      rocprofiler_client_id_t* id)
{
    id->name         = "synchronize-contexts-concurrent-test";
    static auto _cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &_cfg;
}

extern "C" __attribute__((visibility("default"))) uint64_t
sync_test_run(const sync_test_workload_t* workload)
{
    if(!sync_test::workload_is_usable(
           workload, /* required_lanes = */ kThreads, /* requires_copy = */ false, kTestTag))
        return 1;

    const auto* _signal_less = ::getenv(kSignalLessEnvVar);
    fprintf(stderr,
            "[%s] completion route: %s\n",
            kTestTag,
            (_signal_less != nullptr) ? "signal-less requested" : "signal-based");

    // Warm up the lazy paths -- module load, dispatch-log session, first touch -- on one lane, so
    // the first window does not measure them.
    {
        auto warmup = scoped_lane{*workload, 0};
        for(int i = 0; i < 10; ++i)
            warmup.launch_kernel();
        warmup.wait();
        ROCPROFILER_CHECK(rocprofiler_synchronize_context(g_context, nullptr, nullptr));
    }

    auto failures = uint64_t{0};
    failures += each_thread_sees_its_own_dispatch_record(*workload);
    failures += one_sync_after_join_sees_every_record(*workload);

    if(!g_callback_ran.load())
    {
        ++failures;
        fprintf(stderr,
                "[%s] the buffer callback never ran, so no record was delivered by any route and "
                "this run proves nothing about ordering\n",
                kTestTag);
    }

    if(failures > 0)
    {
        // Settle everything, then re-examine the tags missing at window close: present now means
        // delivered late, still absent means dropped.
        ROCPROFILER_CHECK(rocprofiler_synchronize_context(g_context, nullptr, nullptr));

        auto late = uint64_t{0};
        auto lost = uint64_t{0};
        {
            auto _lk = std::lock_guard<std::mutex>{g_late_mutex};
            for(auto tag : g_missed_at_window_close)
                (delivered(tag) > 0) ? ++late : ++lost;
        }

        fprintf(stderr,
                "[%s] FAILED: %lu failed expectations across %lu threads (%lu record(s) arrived "
                "late, %lu never arrived)\n",
                kTestTag,
                static_cast<unsigned long>(failures),
                static_cast<unsigned long>(kThreads),
                static_cast<unsigned long>(late),
                static_cast<unsigned long>(lost));
        return failures;
    }

    fprintf(stderr,
            "[%s] PASSED: %lu threads each saw their own record across %lu windows, and one "
            "synchronize after the join accounted for all %lu dispatches\n",
            kTestTag,
            static_cast<unsigned long>(kThreads),
            static_cast<unsigned long>(kWindowsPerThread),
            static_cast<unsigned long>(kThreads * kDispatchesPerThread));
    return 0;
}
