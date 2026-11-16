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

// rocprofiler_synchronize_context, over buffer tracing records.
//
// Records for asynchronous operations are written by the runtime's own threads once the operation
// retires, so a tool that runs one, waits on the device and flushes can get nothing back: the
// record lands in the window after the one that earned it. This call waits for record production
// to catch up, then flushes the context's buffers unless told not to.
//
// Five checks, each a function named for the property it proves:
//
//   every_window_sees_its_own_record    the regression guard: many short windows, each requiring
//                                       its own operation's record back inside the window.
//   unknown_context_is_rejected         a context the runtime never handed out is refused.
//   nohang_without_result_is_rejected   NOHANG with nowhere to put the answer is refused.
//   noflush_still_waits_for_records     NOFLUSH drops the flush but keeps the wait.
//   nohang_never_reports_timeout        NOHANG returns a reading instead of waiting, so it
//                                       cannot report a timeout.
//
// The window loop alternates kernel dispatch and memory copy so a regression in either tracing
// domain fails. Every window is asserted on its own and never in aggregate: a late record still
// arrives one window later, so totals reconcile even when every window came up short.

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
constexpr const char* kTestTag = "synchronize-contexts";

// Enough that passing with the wait removed is not a realistic outcome, while still running in a
// couple of seconds. Too few and an unloaded machine passes against a broken build.
constexpr uint64_t kWindows = 1000;

// Window labels for the checks that run after the loop. Past kWindows so a record left over from
// the loop can never be mistaken for one of theirs.
constexpr uint64_t kNoflushTag = kWindows + 1;
constexpr uint64_t kNohangTag  = kWindows + 2;

constexpr size_t kCopyBytes     = 8 * 1024 * 1024;
constexpr size_t kScratchFloats = 1024;

using sync_test::make_options;
using sync_test::make_result;

// Tool: one lossless buffer carrying both of the traced domains.

rocprofiler_context_id_t g_context = {};
rocprofiler_buffer_id_t  g_buffer  = {};

// `tag` is the external correlation id pushed when the operation was launched, which is how a
// record is attributed back to the window that earned it.
struct delivered_record
{
    uint64_t                          tag;
    rocprofiler_buffer_tracing_kind_t kind;
};

std::mutex                    g_delivered_mutex;
std::vector<delivered_record> g_delivered;

// Guards against a vacuous pass: a build that stops delivering records entirely looks like one
// that merely delivers them late, since every window comes up short either way.
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

        if(header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            const auto* record =
                static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
            g_delivered.push_back({record->correlation_id.external.value,
                                   ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH});
        }
        else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
        {
            const auto* record =
                static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);
            g_delivered.push_back(
                {record->correlation_id.external.value, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY});
        }
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
    CHECK_ROCP(rocprofiler_configure_buffer_tracing_service(
        g_context, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0, g_buffer));
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

enum class operation
{
    kernel_dispatch,
    memory_copy,
};

rocprofiler_buffer_tracing_kind_t
record_kind_of(operation op)
{
    return (op == operation::kernel_dispatch) ? ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH
                                              : ROCPROFILER_BUFFER_TRACING_MEMORY_COPY;
}

// The copy source is pinned host memory and large enough that the runtime uses the copy engine.
// A small copy, or one between two device allocations, is serviced by a blit kernel instead,
// which emits a dispatch record and leaves the memory copy domain untested while appearing to
// pass.
class gpu_workload
{
public:
    gpu_workload()
    {
        CHECK_HIP(hipMalloc(&m_scratch, kScratchFloats * sizeof(float)));
        CHECK_HIP(hipMemset(m_scratch, 0, kScratchFloats * sizeof(float)));
        CHECK_HIP(hipMalloc(&m_copy_dst, kCopyBytes));
        CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&m_copy_src), kCopyBytes));
    }

    ~gpu_workload()
    {
        CHECK_HIP(hipHostFree(m_copy_src));
        CHECK_HIP(hipFree(m_copy_dst));
        CHECK_HIP(hipFree(m_scratch));
    }

    gpu_workload(const gpu_workload&) = delete;
    gpu_workload& operator=(const gpu_workload&) = delete;

    void launch(operation op) const
    {
        if(op == operation::kernel_dispatch)
        {
            touch_kernel<<<1, 64>>>(m_scratch);
            CHECK_HIP(hipGetLastError());
        }
        else
        {
            CHECK_HIP(
                hipMemcpyAsync(m_copy_dst, m_copy_src, kCopyBytes, hipMemcpyHostToDevice, nullptr));
        }
    }

    static void wait_on_device() { CHECK_HIP(hipDeviceSynchronize()); }

private:
    float* m_scratch  = nullptr;
    float* m_copy_dst = nullptr;
    float* m_copy_src = nullptr;
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

struct delivery_tally
{
    uint64_t matching   = 0;  ///< records carrying `tag` and the expected kind
    uint64_t wrong_kind = 0;  ///< records carrying `tag` but some other kind
};

// Consume everything delivered and report what belonged to `tag`. Draining is what keeps each
// window's verdict independent of the ones before it.
delivery_tally
take_delivered(uint64_t tag, rocprofiler_buffer_tracing_kind_t expected_kind)
{
    auto tally = delivery_tally{};
    auto _lk   = std::lock_guard<std::mutex>{g_delivered_mutex};

    for(const auto& record : g_delivered)
    {
        if(record.tag != tag) continue;
        if(record.kind == expected_kind)
            ++tally.matching;
        else
            ++tally.wrong_kind;
    }
    g_delivered.clear();
    return tally;
}

// Throw away everything delivered so far, so the next check starts from a known-empty slate.
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
every_window_sees_its_own_record(const gpu_workload& workload)
{
    auto failures = uint64_t{0};

    for(uint64_t window = 0; window < kWindows; ++window)
    {
        const auto op = (window % 2 == 0) ? operation::kernel_dispatch : operation::memory_copy;
        const auto expected_kind = record_kind_of(op);

        push_tag(window);
        workload.launch(op);
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

        // The kind is checked, not just the tag: accepting either would let a run pass without
        // exercising the copy path, which the runtime may satisfy with a blit kernel.
        const auto tally = take_delivered(window, expected_kind);
        if(tally.matching == 0)
        {
            ++failures;
            if(tally.wrong_kind > 0)
                fprintf(stderr,
                        "[%s] window %lu: received a record of the wrong kind, so the workload "
                        "did not exercise the domain this window was testing\n",
                        kTestTag,
                        static_cast<unsigned long>(window));
            else
                fprintf(stderr,
                        "[%s] window %lu: closed with no record for its own operation\n",
                        kTestTag,
                        static_cast<unsigned long>(window));
        }
    }

    return failures;
}

// A context id the runtime never handed out has no records to wait for and no buffers to flush.
uint64_t
unknown_context_is_rejected()
{
    auto failures = uint64_t{0};
    auto options  = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NONE);
    auto result   = make_result();

    const auto never_issued = rocprofiler_context_id_t{~uint64_t{0}};
    failures += EXPECT_STATUS(ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND,
                              rocprofiler_synchronize_context(never_issued, &options, &result));
    failures +=
        EXPECT_STATUS(ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND,
                      rocprofiler_synchronize_context(ROCPROFILER_CONTEXT_NONE, &options, &result));
    return failures;
}

// NOHANG's entire output is the result struct, so asking for it without somewhere to put the
// answer is a caller error rather than a silently ignored flag.
uint64_t
nohang_without_result_is_rejected()
{
    auto options = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NOHANG);
    return EXPECT_STATUS(ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT,
                         rocprofiler_synchronize_context(g_context, &options, nullptr));
}

// NOFLUSH drops the flush but keeps the wait, and the wait is the part that can be asserted: the
// record has been written by the time the call returns, so a later explicit flush produces it.
//
// Deliberately not asserted: that the buffer is still unflushed on return. NOFLUSH promises only
// that this call will not flush, not that nothing else will, so requiring an empty buffer would
// be testing a race. Delivery is counted on both sides of the flush and the total is what matters.
uint64_t
noflush_still_waits_for_records(const gpu_workload& workload)
{
    push_tag(kNoflushTag);
    workload.launch(operation::kernel_dispatch);
    gpu_workload::wait_on_device();

    auto options = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NOFLUSH);
    auto result  = make_result();
    CHECK_ROCP(rocprofiler_synchronize_context(g_context, &options, &result));
    pop_tag();

    auto delivered =
        take_delivered(kNoflushTag, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH).matching;
    CHECK_ROCP(rocprofiler_flush_buffer(g_buffer));
    delivered += take_delivered(kNoflushTag, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH).matching;

    if(delivered == 1) return 0;

    fprintf(stderr,
            "[%s] a NOFLUSH synchronize followed by an explicit flush produced %lu records for "
            "one dispatch, so the call did not wait for the record to be written\n",
            kTestTag,
            static_cast<unsigned long>(delivered));
    return 1;
}

// NOHANG reads the outstanding work and returns instead of waiting, so it cannot report a timeout
// even under a deadline far too short for a real wait. The operation is left in flight on
// purpose: a device synchronize first would remove the thing being measured.
uint64_t
nohang_never_reports_timeout(const gpu_workload& workload)
{
    auto failures = uint64_t{0};

    push_tag(kNohangTag);
    workload.launch(operation::kernel_dispatch);

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

    // Settle the operation left in flight, taking the documented both-arguments-NULL path.
    gpu_workload::wait_on_device();
    CHECK_ROCP(rocprofiler_synchronize_context(g_context, nullptr, nullptr));
    discard_delivered();

    return failures;
}
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

int
main(int, char**)
{
    auto workload = gpu_workload{};

    // Warm up the lazy paths -- module load, signal pool, first touch -- then start clean.
    for(int i = 0; i < 10; ++i)
        workload.launch(operation::kernel_dispatch);
    gpu_workload::wait_on_device();
    CHECK_ROCP(rocprofiler_synchronize_context(g_context, nullptr, nullptr));
    discard_delivered();

    auto failures = uint64_t{0};
    failures += every_window_sees_its_own_record(workload);
    failures += unknown_context_is_rejected();
    failures += nohang_without_result_is_rejected();
    failures += noflush_still_waits_for_records(workload);
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
                "[%s] FAILED: %lu failed expectations across %lu windows and 4 option checks\n",
                kTestTag,
                static_cast<unsigned long>(failures),
                static_cast<unsigned long>(kWindows));
        return EXIT_FAILURE;
    }

    fprintf(stderr,
            "[%s] PASSED: %lu windows received their own record, and NONE, NOHANG and NOFLUSH all "
            "behaved as documented\n",
            kTestTag,
            static_cast<unsigned long>(kWindows));
    return EXIT_SUCCESS;
}
