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

// rocprofiler_synchronize_context, over signal-less kernel-dispatch completion.
//
// An eligible batch publishes its packets untouched and completes from a firmware record rather
// than an SDK completion signal -- a third route out of the runtime, alongside tracing_window.cpp
// and counter_window.cpp. The feature is opt-in through ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS,
// read once per process, and changes completion semantics process-wide, so it cannot be turned on
// partway through a test: hence a binary and a ctest environment of its own.
//
//   every_window_sees_its_dispatch_record   many short windows, each requiring its own dispatch's
//                                           record back inside the window.
//   nohang_never_reports_timeout            NOHANG returns a reading instead of waiting, so it
//                                           cannot report a timeout. The peek is separate code
//                                           from the wait, and one blind to a signal-less packet
//                                           in flight is wrong in a way the wait never shows.
//
// Option and argument validation common to the API is covered once, in tracing_window.cpp.
//
// COVERAGE BOUNDARY. Signal-less engages only on a kernel whose dispatch-log ABI matches the one
// the SDK was built against; elsewhere the SDK falls back to signal-based completion and these
// checks re-prove what tracing_window.cpp already proves rather than fail. Nothing inside the
// process distinguishes the two, so the ctest registration gates on the SDK's teardown summary
// and skips a run that registered no batch.

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

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace
{
constexpr const char* kTestTag = "synchronize-contexts-signal-less";

// Read here only to confirm it reached the process: the SDK reads it once, early, and never
// reports what it decided, so a run that silently went without it looks like a passing one.
constexpr const char* kFeatureEnvVar = "ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS";

// Fewer than the tracing window loop: every dispatch here waits on a firmware record rather than
// on a signal the SDK controls, which is the slower of the two.
constexpr uint64_t kWindows = 500;

constexpr uint64_t kNohangTag     = kWindows + 1;
constexpr size_t   kScratchFloats = 1024;

using sync_test::make_options;
using sync_test::make_result;

// Tool: kernel dispatch only, the one domain signal-less completion changes.

rocprofiler_context_id_t g_context = {};
rocprofiler_buffer_id_t  g_buffer  = {};

std::mutex            g_delivered_mutex;
std::vector<uint64_t> g_delivered;  ///< external correlation id of each dispatch record delivered

// Guards against a vacuous pass: if the callback never ran, every window looks deficient for a
// reason unrelated to ordering.
std::atomic<bool> g_callback_ran{false};

void
buffer_callback(rocprofiler_context_id_t /*context*/,
                rocprofiler_buffer_id_t /*buffer_id*/,
                rocprofiler_record_header_t** headers,
                size_t                        num_headers,
                void* /*user_data*/,
                uint64_t /*drop_count*/)
{
    g_callback_ran.store(true);

    auto _lk = std::lock_guard<std::mutex>{g_delivered_mutex};
    for(size_t i = 0; i < num_headers; ++i)
    {
        const auto* header = headers[i];
        if(header == nullptr || header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;
        if(header->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) continue;

        const auto* record =
            static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
        g_delivered.push_back(record->correlation_id.external.value);
    }
}

int
tool_init(rocprofiler_client_finalize_t /*fini_func*/, void* /*tool_data*/)
{
    CHECK_ROCP(rocprofiler_create_context(&g_context));
    CHECK_ROCP(rocprofiler_create_buffer(g_context,
                                         256 * 1024,
                                         128 * 1024,
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

class gpu_workload
{
public:
    gpu_workload()
    {
        CHECK_HIP(hipMalloc(&m_scratch, kScratchFloats * sizeof(float)));
        CHECK_HIP(hipMemset(m_scratch, 0, kScratchFloats * sizeof(float)));
    }

    ~gpu_workload() { CHECK_HIP(hipFree(m_scratch)); }

    gpu_workload(const gpu_workload&) = delete;
    gpu_workload& operator=(const gpu_workload&) = delete;

    void launch() const
    {
        touch_kernel<<<1, 64>>>(m_scratch);
        CHECK_HIP(hipGetLastError());
    }

    static void wait_on_device() { CHECK_HIP(hipDeviceSynchronize()); }

private:
    float* m_scratch = nullptr;
};

// Window bookkeeping

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

// Consume everything delivered and report how much carried `tag`. Draining is what keeps each
// window's verdict independent of the ones before it.
uint64_t
take_delivered(uint64_t tag)
{
    auto matching = uint64_t{0};
    auto _lk      = std::lock_guard<std::mutex>{g_delivered_mutex};

    for(auto itr : g_delivered)
        if(itr == tag) ++matching;

    g_delivered.clear();
    return matching;
}

void
discard_delivered()
{
    auto _lk = std::lock_guard<std::mutex>{g_delivered_mutex};
    g_delivered.clear();
}

// Checks. Each returns a count of failed expectations, so main can report every problem at once.

// The regression guard. The synchronize flushes as part of the call, so the window deliberately
// does not flush the buffer itself -- that would mask a regression in half the contract.
uint64_t
every_window_sees_its_dispatch_record(const gpu_workload& workload)
{
    auto failures = uint64_t{0};

    for(uint64_t window = 0; window < kWindows; ++window)
    {
        push_tag(window);
        workload.launch();
        gpu_workload::wait_on_device();

        auto options = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NONE);
        auto result  = make_result();
        CHECK_ROCP(rocprofiler_synchronize_context(g_context, &options, &result));
        pop_tag();

        // The wait was unbounded, so a success reporting outstanding records would mean the
        // status and the result disagree.
        if(result.num_timeout_records != 0)
        {
            ++failures;
            fprintf(stderr,
                    "[%s] window %lu: synchronize succeeded but reported %lu records still "
                    "outstanding\n",
                    kTestTag,
                    static_cast<unsigned long>(window),
                    static_cast<unsigned long>(result.num_timeout_records));
        }

        if(take_delivered(window) == 0)
        {
            ++failures;
            fprintf(stderr,
                    "[%s] window %lu: closed with no record for its own dispatch\n",
                    kTestTag,
                    static_cast<unsigned long>(window));
        }
    }

    return failures;
}

// NOHANG reads the outstanding work and returns instead of waiting, so it cannot report a timeout
// even under a deadline far too short for a real wait. The dispatch is left in flight on purpose:
// an in-flight signal-less dispatch is exactly what the peek has to account for.
uint64_t
nohang_never_reports_timeout(const gpu_workload& workload)
{
    auto failures = uint64_t{0};

    push_tag(kNohangTag);
    workload.launch();

    auto options = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NOHANG, /* timeout_ns = */ 1);
    auto result  = make_result();
    failures += EXPECT_STATUS(ROCPROFILER_STATUS_SUCCESS,
                              rocprofiler_synchronize_context(g_context, &options, &result));
    pop_tag();

    if(result.num_timeout_records != 0)
    {
        ++failures;
        fprintf(stderr,
                "[%s] NOHANG reported %lu records outstanding after a timeout, but it never "
                "waited and so can never have timed out\n",
                kTestTag,
                static_cast<unsigned long>(result.num_timeout_records));
    }

    // Settle the dispatch left in flight, taking the documented both-arguments-NULL path.
    gpu_workload::wait_on_device();
    CHECK_ROCP(rocprofiler_synchronize_context(g_context, nullptr, nullptr));

    if(take_delivered(kNohangTag) == 0)
    {
        ++failures;
        fprintf(stderr,
                "[%s] the dispatch left in flight across the NOHANG call never produced a record, "
                "even after an unbounded synchronize\n",
                kTestTag);
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
    id->name         = "synchronize-contexts-signal-less-test";
    static auto _cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &_cfg;
}

int
main(int, char**)
{
    // The SDK read the variable before main, so an environment that never carried it produces a
    // run that passes while testing the signal path. Fail rather than skip: that is a broken test
    // registration, not an unsuitable machine.
    if(::getenv(kFeatureEnvVar) == nullptr)
    {
        fprintf(stderr,
                "[%s] %s is unset, so this run would exercise signal-based completion under the "
                "name of the signal-less test\n",
                kTestTag,
                kFeatureEnvVar);
        return EXIT_FAILURE;
    }

    auto workload = gpu_workload{};

    // Warm up the lazy paths -- module load, dispatch-log session, first touch -- then start
    // clean.
    for(int i = 0; i < 10; ++i)
        workload.launch();
    gpu_workload::wait_on_device();
    CHECK_ROCP(rocprofiler_synchronize_context(g_context, nullptr, nullptr));
    discard_delivered();

    auto failures = uint64_t{0};
    failures += every_window_sees_its_dispatch_record(workload);
    failures += nohang_never_reports_timeout(workload);

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
        fprintf(stderr,
                "[%s] FAILED: %lu failed expectations across %lu windows\n",
                kTestTag,
                static_cast<unsigned long>(failures),
                static_cast<unsigned long>(kWindows));
        return EXIT_FAILURE;
    }

    fprintf(stderr,
            "[%s] PASSED: %lu windows received their own dispatch record with signal-less "
            "completion requested\n",
            kTestTag,
            static_cast<unsigned long>(kWindows));
    return EXIT_SUCCESS;
}
