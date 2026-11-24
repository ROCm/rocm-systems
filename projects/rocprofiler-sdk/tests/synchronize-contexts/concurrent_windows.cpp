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
// tests are single-threaded, so nothing else covers the drain's cross-thread behaviour.
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
// Argument validation common to the API is covered once, in tracing_window.cpp.

#include "test_support.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/experimental/synchronize_contexts.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
constexpr const char* kTestTag = "synchronize-contexts-concurrent";

// Reported, not required: the binary runs under both routes, so this only tells the log which one.
constexpr const char* kSignalLessEnvVar = "ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS";

constexpr uint64_t kThreads             = 4;
constexpr uint64_t kWindowsPerThread    = 100;
constexpr uint64_t kDispatchesPerThread = 50;
constexpr size_t   kScratchFloats       = 1024;

// Tag 0 means untagged, so both ranges start at 1 and never overlap.
constexpr uint64_t kWindowTagBase = 1;
constexpr uint64_t kJoinTagBase   = kWindowTagBase + (kThreads * kWindowsPerThread);
constexpr uint64_t kTagSpace      = kJoinTagBase + (kThreads * kDispatchesPerThread);

using sync_test::make_options;
using sync_test::make_result;

rocprofiler_context_id_t g_context = {};
rocprofiler_buffer_id_t  g_buffer  = {};

// One counter per tag, never cleared: threads read their own tags while others are still written.
std::array<std::atomic<uint64_t>, kTagSpace> g_delivered{};

// Guards against a vacuous pass: a callback that never ran fails every check for the wrong reason.
std::atomic<bool> g_callback_ran{false};

// Missing at window close, re-checked after the drain: a record present by then was late rather
// than lost -- a different defect, and the failure message should say which.
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
    CHECK_ROCP(rocprofiler_create_context(&g_context));
    CHECK_ROCP(rocprofiler_create_buffer(g_context,
                                         512 * 1024,
                                         256 * 1024,
                                         ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                         buffer_callback,
                                         nullptr,
                                         &g_buffer));
    CHECK_ROCP(rocprofiler_configure_buffer_tracing_service(
        g_context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, g_buffer));
    CHECK_ROCP(rocprofiler_start_context(g_context));
    return 0;
}

void
tool_fini(void* /*tool_data*/)
{}

// Workload

__global__ void
touch_kernel(float* ptr)
{
    if(threadIdx.x == 0) ptr[0] += 1.0F;
}

// Per-thread scratch and stream. A shared null stream would serialise the enqueues and remove the
// contention this test exists to create.
class thread_workload
{
public:
    thread_workload()
    {
        CHECK_HIP(hipMalloc(&m_scratch, kScratchFloats * sizeof(float)));
        CHECK_HIP(hipMemset(m_scratch, 0, kScratchFloats * sizeof(float)));
        CHECK_HIP(hipStreamCreate(&m_stream));
    }

    ~thread_workload()
    {
        CHECK_HIP(hipStreamDestroy(m_stream));
        CHECK_HIP(hipFree(m_scratch));
    }

    thread_workload(const thread_workload&) = delete;
    thread_workload& operator=(const thread_workload&) = delete;

    void launch() const
    {
        touch_kernel<<<1, 64, 0, m_stream>>>(m_scratch);
        CHECK_HIP(hipGetLastError());
    }

    void wait() const { CHECK_HIP(hipStreamSynchronize(m_stream)); }

private:
    float*      m_scratch = nullptr;
    hipStream_t m_stream  = nullptr;
};

// Tagging. Every thread pushes against its own thread id, so the tags do not interfere.

void
push_tag(uint64_t tag)
{
    auto thread_id = rocprofiler_thread_id_t{0};
    CHECK_ROCP(rocprofiler_get_thread_id(&thread_id));

    auto external  = rocprofiler_user_data_t{};
    external.value = tag;
    CHECK_ROCP(rocprofiler_push_external_correlation_id(g_context, thread_id, external));
}

void
pop_tag()
{
    auto thread_id = rocprofiler_thread_id_t{0};
    CHECK_ROCP(rocprofiler_get_thread_id(&thread_id));

    auto popped = rocprofiler_user_data_t{};
    CHECK_ROCP(rocprofiler_pop_external_correlation_id(g_context, thread_id, &popped));
}

uint64_t
delivered(uint64_t tag)
{
    return g_delivered[tag].load(std::memory_order_relaxed);
}

// Checks. Each returns a count of failed expectations so main can report every problem at once.

// The cross-thread regression guard. Each thread waits for its own dispatch before synchronizing,
// so its record is owed; the others are mid-enqueue throughout, which is the contention.
uint64_t
each_thread_sees_its_own_dispatch_record()
{
    auto failures = std::atomic<uint64_t>{0};
    auto threads  = std::vector<std::thread>{};
    threads.reserve(kThreads);

    for(uint64_t t = 0; t < kThreads; ++t)
    {
        threads.emplace_back([t, &failures]() {
            auto workload = thread_workload{};

            for(uint64_t window = 0; window < kWindowsPerThread; ++window)
            {
                const auto tag = kWindowTagBase + (t * kWindowsPerThread) + window;

                push_tag(tag);
                workload.launch();
                workload.wait();

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
one_sync_after_join_sees_every_record()
{
    auto threads = std::vector<std::thread>{};
    threads.reserve(kThreads);

    for(uint64_t t = 0; t < kThreads; ++t)
    {
        threads.emplace_back([t]() {
            auto workload = thread_workload{};

            for(uint64_t i = 0; i < kDispatchesPerThread; ++i)
            {
                push_tag(kJoinTagBase + (t * kDispatchesPerThread) + i);
                workload.launch();
                pop_tag();
            }
            workload.wait();
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

extern "C" __attribute__((visibility("default"))) rocprofiler_tool_configure_result_t*
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

int
main(int, char**)
{
    const auto* _signal_less = ::getenv(kSignalLessEnvVar);
    fprintf(stderr,
            "[%s] completion route: %s\n",
            kTestTag,
            (_signal_less != nullptr) ? "signal-less requested" : "signal-based");

    // Warm up the lazy paths -- module load, dispatch-log session, first touch -- on one thread,
    // so the first window does not measure them.
    {
        auto warmup = thread_workload{};
        for(int i = 0; i < 10; ++i)
            warmup.launch();
        warmup.wait();
        CHECK_ROCP(rocprofiler_synchronize_context(g_context, nullptr, nullptr));
    }

    auto failures = uint64_t{0};
    failures += each_thread_sees_its_own_dispatch_record();
    failures += one_sync_after_join_sees_every_record();

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
        // Late or lost: settle everything, then re-examine the tags missing at window close. A
        // record present now was delivered late -- an ordering defect. One still absent was
        // dropped.
        CHECK_ROCP(rocprofiler_synchronize_context(g_context, nullptr, nullptr));

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
        return EXIT_FAILURE;
    }

    fprintf(stderr,
            "[%s] PASSED: %lu threads each saw their own record across %lu windows, and one "
            "synchronize after the join accounted for all %lu dispatches\n",
            kTestTag,
            static_cast<unsigned long>(kThreads),
            static_cast<unsigned long>(kWindowsPerThread),
            static_cast<unsigned long>(kThreads * kDispatchesPerThread));
    return EXIT_SUCCESS;
}
