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

// rocprofiler_synchronize_context, over device counting samples.
//
// Device counting is not driven by kernel dispatch: the tool asks an agent for a reading, the
// runtime submits a read packet and a barrier to that agent's profile queue, and a completion
// handler writes the sample afterwards. A route of its own, drained by different code from the
// dispatch routes. Samples are requested asynchronously, since a synchronous one waits for its
// own handler and leaves the drain nothing to do.
//
//   every_window_sees_its_own_sample   the regression guard: many short windows, each requiring
//                                      the sample it asked for back inside the window.
//   noflush_still_waits_for_sample     NOFLUSH drops the flush but keeps the wait.
//   nohang_never_reports_timeout       the peek is separate code from the wait, and one blind to
//                                      an agent still owing a sample is wrong in a way the wait
//                                      never shows.
//
// Option and argument validation common to the API is covered in tracing_window_client.cpp.
// Windows are asserted individually, never in aggregate: a late sample still arrives one window
// later, so totals reconcile even when every window came up short.
//
// Coverage boundary: the service needs an agent the runtime will profile. Where it never engages
// nothing is delivered at all, so the ctest registration gates on the summary line below.

#include "counter_config.hpp"
#include "sync_test_workload.h"
#include "test_support.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/experimental/synchronize_contexts.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace
{
constexpr const char* kTestTag = "synchronize-contexts";

// Enough windows that a sample arriving one late cannot pass unnoticed, at a profile queue
// round trip apiece.
constexpr uint64_t kWindows = 500;

// Window labels for the checks that run after the loop. Past kWindows, so a sample left over from
// the loop cannot be mistaken for one of theirs.
constexpr uint64_t kWarmupTag  = kWindows + 1;
constexpr uint64_t kNoflushTag = kWindows + 2;
constexpr uint64_t kNohangTag  = kWindows + 3;

// The cheapest counter a profiling-capable agent supports, as in counter_window_client.cpp. The
// values are never read: what is under test is when the sample arrives, not what it says.
constexpr const char* kCounterName = "SQ_WAVES";

using sync_test::make_options;
using sync_test::make_result;
using sync_test::scoped_lane;

// Tool: one lossless buffer carrying the samples of a single agent.

rocprofiler_context_id_t g_context = {};
rocprofiler_buffer_id_t  g_buffer  = {};
rocprofiler_agent_id_t   g_agent   = {};

// `user_data` is the value handed to the sample call, attributing a record back to the window
// that asked for it.
std::mutex            g_delivered_mutex;
std::vector<uint64_t> g_delivered;
uint64_t              g_samples_total = 0;

void
buffer_callback(rocprofiler_context_id_t /*context*/,
                rocprofiler_buffer_id_t /*buffer_id*/,
                rocprofiler_record_header_t** headers,
                size_t                        num_headers,
                void* /*user_data*/,
                uint64_t /*drop_count*/)
{
    auto _lk = std::lock_guard<std::mutex>{g_delivered_mutex};
    for(size_t i = 0; i < num_headers; ++i)
    {
        const auto* header = headers[i];
        if(header == nullptr) continue;
        if(header->category != ROCPROFILER_BUFFER_CATEGORY_COUNTERS) continue;
        if(header->kind != ROCPROFILER_COUNTER_RECORD_VALUE) continue;

        const auto* record = static_cast<rocprofiler_counter_record_t*>(header->payload);
        g_delivered.push_back(record->user_data.value);
        ++g_samples_total;
    }
}

// Called once per configured agent when the context starts, which is the earliest the counter
// configuration can be built.
void
set_profile(rocprofiler_context_id_t               context,
            rocprofiler_agent_id_t                 agent,
            rocprofiler_device_counting_agent_cb_t set_config,
            void* /*user_data*/)
{
    set_config(context, sync_test::counter_config_for_agent(agent, kCounterName, kTestTag));
}

// The first GPU both runtimes can see, or a zero handle where there is none. One is enough: the
// service permits only one context profiling a given agent.
rocprofiler_agent_id_t
find_profiling_agent()
{
    auto agent = rocprofiler_agent_id_t{};

    ROCPROFILER_CHECK(rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0,
        [](rocprofiler_agent_version_t version, const void** agents, size_t num_agents, void* out)
            -> rocprofiler_status_t {
            if(version != ROCPROFILER_AGENT_INFO_VERSION_0)
                return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

            auto* _out = static_cast<rocprofiler_agent_id_t*>(out);
            for(size_t i = 0; i < num_agents; ++i)
            {
                const auto* _agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
                if(_agent->type == ROCPROFILER_AGENT_TYPE_GPU && _agent->runtime_visibility.hsa &&
                   _agent->runtime_visibility.hip)
                {
                    *_out = _agent->id;
                    return ROCPROFILER_STATUS_SUCCESS;
                }
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        sizeof(rocprofiler_agent_t),
        &agent));

    return agent;
}

int
tool_init(rocprofiler_client_finalize_t /*fini_func*/, void* /*tool_data*/)
{
    ROCPROFILER_CHECK(rocprofiler_create_context(&g_context));
    ROCPROFILER_CHECK(rocprofiler_create_buffer(g_context,
                                                256 * 1024,
                                                128 * 1024,
                                                ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                buffer_callback,
                                                nullptr,
                                                &g_buffer));

    g_agent = find_profiling_agent();
    if(g_agent.handle == 0) return 0;

    ROCPROFILER_CHECK(rocprofiler_configure_device_counting_service(
        g_context, g_buffer, g_agent, set_profile, nullptr));

    // Deliberately not started here. Starting the service submits packets to the agent's profile
    // queue, and this runs while the runtime that owns that queue is still coming up.
    return 0;
}

void
tool_fini(void* /*tool_data*/)
{}

// Window bookkeeping

void
request_sample(uint64_t tag)
{
    auto user_data  = rocprofiler_user_data_t{};
    user_data.value = tag;

    // ASYNC forbids output_records, so the sample can only come back through the buffer.
    ROCPROFILER_CHECK(rocprofiler_sample_device_counting_service(
        g_context, user_data, ROCPROFILER_COUNTER_FLAG_ASYNC, nullptr, nullptr));
}

// Consume everything delivered and report how much of it belonged to `tag`. Draining keeps each
// window's verdict independent of the ones before it.
uint64_t
take_delivered(uint64_t tag)
{
    auto matching = uint64_t{0};
    auto _lk      = std::lock_guard<std::mutex>{g_delivered_mutex};

    for(auto value : g_delivered)
        if(value == tag) ++matching;

    g_delivered.clear();
    return matching;
}

void
discard_delivered()
{
    auto _lk = std::lock_guard<std::mutex>{g_delivered_mutex};
    g_delivered.clear();
}

uint64_t
samples_total()
{
    auto _lk = std::lock_guard<std::mutex>{g_delivered_mutex};
    return g_samples_total;
}

// Checks. Each returns a count of failed expectations, so one run reports every problem.

// The regression guard. The synchronize flushes as part of the call, so the window deliberately
// does not flush the buffer itself -- that would mask a regression in half the contract.
//
// Each window drives a dispatch first: a sample covers whatever the agent did since the last
// one, and a reading taken over an idle agent measures nothing.
uint64_t
every_window_sees_its_own_sample(const scoped_lane& lane)
{
    auto failures = uint64_t{0};

    for(uint64_t window = 0; window < kWindows; ++window)
    {
        lane.launch_kernel();
        lane.wait();
        request_sample(window);

        auto options = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NONE);
        auto result  = make_result();
        ROCPROFILER_CHECK(rocprofiler_synchronize_context(g_context, &options, &result));

        // Success means the drain reported nothing outstanding, so a nonzero count is the
        // status and the result disagreeing.
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

        // Counted as present or absent rather than by number: one sample produces a record per
        // counter instance the agent reports, and how many that is belongs to the agent.
        if(take_delivered(window) == 0)
        {
            ++failures;
            fprintf(stderr,
                    "[%s] window %lu: closed with no sample of its own\n",
                    kTestTag,
                    static_cast<unsigned long>(window));
        }
    }

    return failures;
}

// NOFLUSH drops the flush but keeps the wait: the sample has been written by the time the call
// returns, so a later explicit flush produces it.
//
// Deliberately not asserted: that the buffer is still unflushed on return. NOFLUSH promises only
// that this call will not flush, not that nothing else will, so an empty-buffer assertion would
// be a race. Delivery is counted on both sides of the flush and the total is what matters.
uint64_t
noflush_still_waits_for_sample(const scoped_lane& lane)
{
    lane.launch_kernel();
    lane.wait();
    request_sample(kNoflushTag);

    auto options = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NOFLUSH);
    auto result  = make_result();
    ROCPROFILER_CHECK(rocprofiler_synchronize_context(g_context, &options, &result));

    auto delivered = take_delivered(kNoflushTag);
    ROCPROFILER_CHECK(rocprofiler_flush_buffer(g_buffer));
    delivered += take_delivered(kNoflushTag);

    if(delivered > 0) return 0;

    fprintf(stderr,
            "[%s] a NOFLUSH synchronize followed by an explicit flush produced no record for a "
            "requested sample, so the call did not wait for the sample to be written\n",
            kTestTag);
    return 1;
}

// NOHANG reads the outstanding work and returns instead of waiting, so it cannot report a timeout
// even under a deadline far too short for a real wait. The sample is left in flight on purpose:
// an agent still owing one is exactly what the peek has to account for.
//
// Deliberately not asserted: the number of outstanding records. The sample may land between the
// request and the peek, so zero and nonzero are both honest readings.
uint64_t
nohang_never_reports_timeout(const scoped_lane& lane)
{
    auto failures = uint64_t{0};

    lane.launch_kernel();
    lane.wait();
    request_sample(kNohangTag);

    auto options = make_options(ROCPROFILER_SYNCHRONIZE_CONTEXT_NOHANG, /* timeout_ns = */ 1);
    auto result  = make_result();
    failures += EXPECT_STATUS(ROCPROFILER_STATUS_SUCCESS,
                              rocprofiler_synchronize_context(g_context, &options, &result));

    if(result.num_timeout_records != 0)
    {
        ++failures;
        fprintf(stderr,
                "[%s] NOHANG reported %lu records outstanding after a timeout, but it never "
                "waited and so can never have timed out\n",
                kTestTag,
                static_cast<unsigned long>(result.num_timeout_records));
    }

    // Settle the sample left in flight, taking the documented both-arguments-NULL path.
    ROCPROFILER_CHECK(rocprofiler_synchronize_context(g_context, nullptr, nullptr));

    if(take_delivered(kNohangTag) == 0)
    {
        ++failures;
        fprintf(stderr,
                "[%s] the sample left in flight across the NOHANG call never arrived, even after "
                "an unbounded synchronize\n",
                kTestTag);
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
    id->name         = "synchronize-contexts-device-counting-test";
    static auto _cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &_cfg;
}

extern "C" __attribute__((visibility("default"))) uint64_t
sync_test_run(const sync_test_workload_t* workload)
{
    if(!sync_test::workload_is_usable(
           workload, /* required_lanes = */ 1, /* requires_copy = */ false, kTestTag))
        return 1;

    if(g_agent.handle == 0)
    {
        fprintf(stderr,
                "[%s] device counting summary: samples=0, windows=0 (no agent both runtimes can "
                "see)\n",
                kTestTag);
        return 0;
    }

    // Started here, not in tool_init: by now the runtime owning the profile queue is up.
    ROCPROFILER_CHECK(rocprofiler_start_context(g_context));

    auto lane = scoped_lane{*workload};

    // Warm up the lazy paths -- module load, profile queue, counter configuration -- then start
    // clean.
    for(int i = 0; i < 10; ++i)
        lane.launch_kernel();
    lane.wait();
    request_sample(kWarmupTag);
    ROCPROFILER_CHECK(rocprofiler_synchronize_context(g_context, nullptr, nullptr));
    discard_delivered();

    auto failures = uint64_t{0};
    failures += every_window_sees_its_own_sample(lane);
    failures += noflush_still_waits_for_sample(lane);
    failures += nohang_never_reports_timeout(lane);

    // Printed on every path: a run that delivered nothing reports every window short for reasons
    // unrelated to ordering, and the test registration reads this line to tell the two apart.
    fprintf(stderr,
            "[%s] device counting summary: samples=%lu, windows=%lu\n",
            kTestTag,
            static_cast<unsigned long>(samples_total()),
            static_cast<unsigned long>(kWindows));

    if(samples_total() == 0) return 0;

    if(failures > 0)
    {
        fprintf(stderr,
                "[%s] FAILED: %lu failed expectations across %lu windows\n",
                kTestTag,
                static_cast<unsigned long>(failures),
                static_cast<unsigned long>(kWindows));
        return failures;
    }

    fprintf(stderr,
            "[%s] PASSED: %lu windows received the device counting sample they asked for, and "
            "NOHANG and NOFLUSH behaved as documented\n",
            kTestTag,
            static_cast<unsigned long>(kWindows));
    return 0;
}
