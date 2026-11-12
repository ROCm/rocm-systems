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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// rocprofiler_range_replay_host_run against a real tool: a context configured for range replay
// through rocprofiler_force_configure decides the pass count and the replay mode at CONFIG, and
// records the PASS and CLOSE callbacks. With no GPU agent the device fence, snapshot and restore
// are empty, so the pass loop, the host-region restore and the callback sequence run unchanged.
// The no-context cases are in host_replay.cpp, whose process never registers a context.

#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>

#include <csetjmp>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rr = ::rocprofiler::range_replay;

namespace
{
struct tool_plan_t
{
    uint64_t passes      = 1;
    bool     host_replay = false;
};

struct observed_t
{
    int                                            configs     = 0;
    std::vector<uint64_t>                          pass_enters = {};
    std::vector<uint64_t>                          pass_exits  = {};
    std::vector<rocprofiler_range_replay_status_t> closes      = {};
    std::vector<uint64_t>                          close_ids   = {};
};

tool_plan_t              g_plan     = {};
observed_t               g_observed = {};
rocprofiler_context_id_t g_context  = {.handle = 0};

uint64_t pass_count(uint64_t, rocprofiler_user_data_t) { return g_plan.passes; }

void
tool_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;
    auto* data = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    const bool enter = (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER);

    switch(record.operation)
    {
        case ROCPROFILER_RANGE_REPLAY_CONFIG:
            if(!enter) break;
            ++g_observed.configs;
            data->pass_count_cb = pass_count;
            data->host_replay   = g_plan.host_replay ? 1 : 0;
            break;
        case ROCPROFILER_RANGE_REPLAY_PASS:
            (enter ? g_observed.pass_enters : g_observed.pass_exits).push_back(data->current_pass);
            break;
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            if(!enter) break;
            g_observed.closes.push_back(data->status);
            g_observed.close_ids.push_back(data->range_id);
            break;
        default: break;
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    if(rocprofiler_create_context(&g_context) != ROCPROFILER_STATUS_SUCCESS) return 1;
    if(rocprofiler_configure_callback_tracing_service(g_context,
                                                      ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                                      nullptr,
                                                      0,
                                                      tool_callback,
                                                      nullptr) != ROCPROFILER_STATUS_SUCCESS)
        return 1;
    return (rocprofiler_start_context(g_context) == ROCPROFILER_STATUS_SUCCESS) ? 0 : 1;
}

bool
tool_configured()
{
    static const bool configured = []() {
        static auto result = rocprofiler_tool_configure_result_t{
            sizeof(rocprofiler_tool_configure_result_t), tool_init, [](void*) {}, nullptr};
        rocprofiler_configure_func_t configure =
            [](uint32_t, const char*, uint32_t, rocprofiler_client_id_t* client_id) {
                client_id->name = "range-replay-host-run-test";
                return &result;
            };
        return rocprofiler_force_configure(configure) == ROCPROFILER_STATUS_SUCCESS &&
               g_context.handle != 0;
    }();
    return configured;
}

// Stands in for a kernel launch: a host range that dispatched nothing declines NO_DISPATCH after
// pass 0. Counted dispatches are not recorded packets, so packet replay still sees none.
void
note_dispatch()
{
    if(auto* ctx = rr::current_range(); ctx != nullptr) ctx->record.count_dispatches(1);
}

struct body_log_t
{
    std::vector<uint64_t> passes      = {};
    std::vector<uint64_t> host_passes = {};
    std::vector<bool>     host_mode   = {};
};

void
log_pass(uint64_t pass, void* user_data)
{
    auto& log       = *static_cast<body_log_t*>(user_data);
    auto  host_pass = uint64_t{~0ULL};
    EXPECT_EQ(rocprofiler_range_replay_get_host_pass(&host_pass), ROCPROFILER_STATUS_SUCCESS);
    const auto* ctx = rr::current_range();
    log.passes.emplace_back(pass);
    log.host_passes.emplace_back(host_pass);
    log.host_mode.emplace_back(ctx != nullptr && ctx->host.enabled);
    note_dispatch();
}

class range_replay_host_run : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_TRUE(tool_configured());
        g_plan     = {};
        g_observed = {};
    }

    void TearDown() override
    {
        EXPECT_EQ(rr::current_range(), nullptr) << "host_run left a range open";
        EXPECT_TRUE(rr::take_pending_host_regions().empty()) << "host_run left pending regions";
    }
};
}  // namespace

TEST_F(range_replay_host_run, host_replay_calls_the_body_once_per_pass)
{
    g_plan   = {4, true};
    auto log = body_log_t{};

    EXPECT_EQ(rocprofiler_range_replay_host_run(11, log_pass, &log), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(log.passes, (std::vector<uint64_t>{0, 1, 2, 3}));
    EXPECT_EQ(log.host_passes, log.passes) << "the body's pass is the one get_host_pass reports";
    EXPECT_EQ(log.host_mode, (std::vector<bool>{true, true, true, true}));
    EXPECT_EQ(g_observed.configs, 1);
    EXPECT_EQ(g_observed.pass_enters, (std::vector<uint64_t>{1, 2, 3}));
    EXPECT_EQ(g_observed.pass_exits, (std::vector<uint64_t>{1, 2, 3}));
    EXPECT_EQ(
        g_observed.closes,
        (std::vector<rocprofiler_range_replay_status_t>{ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED}));
    EXPECT_EQ(g_observed.close_ids, (std::vector<uint64_t>{11}));
}

TEST_F(range_replay_host_run, a_tool_that_opts_out_gets_one_call_and_a_close)
{
    g_plan   = {1, true};
    auto log = body_log_t{};

    EXPECT_EQ(rocprofiler_range_replay_host_run(12, log_pass, &log), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(log.passes, (std::vector<uint64_t>{0}));
    EXPECT_TRUE(g_observed.pass_enters.empty());
    EXPECT_EQ(g_observed.closes,
              (std::vector<rocprofiler_range_replay_status_t>{
                  ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT}));
}

TEST_F(range_replay_host_run, packet_replay_calls_the_body_once_and_lets_end_replay_it)
{
    g_plan   = {4, false};
    auto log = body_log_t{};

    EXPECT_EQ(rocprofiler_range_replay_host_run(13, log_pass, &log), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(log.passes, (std::vector<uint64_t>{0}));
    EXPECT_EQ(log.host_mode, (std::vector<bool>{false}));
    EXPECT_EQ(g_observed.configs, 1);
    EXPECT_TRUE(g_observed.pass_enters.empty());
    // Host replay would have replayed the counted dispatch; the packet executor sees no recorded
    // packet.
    EXPECT_EQ(g_observed.closes,
              (std::vector<rocprofiler_range_replay_status_t>{
                  ROCPROFILER_RANGE_REPLAY_STATUS_NO_DISPATCH}));
}

TEST_F(range_replay_host_run, a_stopped_context_runs_the_body_once_without_a_range)
{
    g_plan   = {4, true};
    auto log = body_log_t{};

    ASSERT_EQ(rocprofiler_stop_context(g_context), ROCPROFILER_STATUS_SUCCESS);
    const auto status = rocprofiler_range_replay_host_run(14, log_pass, &log);
    ASSERT_EQ(rocprofiler_start_context(g_context), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(status, ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED);
    EXPECT_EQ(log.passes, (std::vector<uint64_t>{0}));
    EXPECT_EQ(log.host_mode, (std::vector<bool>{false}));
    EXPECT_EQ(g_observed.configs, 0);
    EXPECT_TRUE(g_observed.closes.empty());
}

TEST_F(range_replay_host_run, a_null_body_is_rejected_without_opening_a_range)
{
    g_plan = {4, true};

    EXPECT_EQ(rocprofiler_range_replay_host_run(15, nullptr, nullptr),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(g_observed.configs, 0);
    EXPECT_TRUE(g_observed.closes.empty());
}

namespace
{
struct nested_log_t
{
    std::vector<uint64_t>             passes       = {};
    std::vector<uint64_t>             inner_passes = {};
    std::vector<rocprofiler_status_t> statuses     = {};
};

void
inner_body(uint64_t pass, void* user_data)
{
    static_cast<nested_log_t*>(user_data)->inner_passes.emplace_back(pass);
}

void
nesting_body(uint64_t pass, void* user_data)
{
    auto& log = *static_cast<nested_log_t*>(user_data);
    log.passes.emplace_back(pass);
    note_dispatch();

    log.statuses.emplace_back(rocprofiler_range_replay_begin(91));
    log.statuses.emplace_back(rocprofiler_range_replay_end());
    log.statuses.emplace_back(rocprofiler_range_replay_host_run(92, inner_body, &log));

    // The macro arms the jump buffer; the rejected begin must consume the arming without
    // retargeting anything, so this frame is never jumped to.
    auto status = ROCPROFILER_STATUS_SUCCESS;
    ROCPROFILER_RANGE_REPLAY_HOST_BEGIN(93, status);
    log.statuses.emplace_back(status);
}
}  // namespace

TEST_F(range_replay_host_run, nested_entry_points_are_rejected_and_the_range_completes)
{
    g_plan   = {3, true};
    auto log = nested_log_t{};

    EXPECT_EQ(rocprofiler_range_replay_host_run(16, nesting_body, &log),
              ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(log.passes, (std::vector<uint64_t>{0, 1, 2}));
    EXPECT_EQ(log.inner_passes, log.passes)
        << "a rejected nested host_run runs its body once, as the enclosing range's pass";
    EXPECT_EQ(log.statuses,
              std::vector<rocprofiler_status_t>(4 * 3, ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT));
    EXPECT_FALSE(rr::take_host_jmp_armed());
    EXPECT_EQ(g_observed.configs, 1);
    EXPECT_EQ(g_observed.pass_enters, (std::vector<uint64_t>{1, 2}));
    EXPECT_EQ(
        g_observed.closes,
        (std::vector<rocprofiler_range_replay_status_t>{ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED}));
    EXPECT_EQ(g_observed.close_ids, (std::vector<uint64_t>{16}));
}

namespace
{
struct region_log_t
{
    int              registered_before = 10;
    int              registered_inside = 0;
    std::vector<int> seen_before       = {};
    std::vector<int> seen_inside       = {};
};

// Non-idempotent host computation over both regions: a pass over un-restored state differs.
void
mutate_regions(uint64_t pass, void* user_data)
{
    auto& log = *static_cast<region_log_t*>(user_data);
    note_dispatch();
    if(pass == 0) log.registered_inside = 100;
    EXPECT_EQ(rocprofiler_range_replay_host_track_memory(&log.registered_inside,
                                                         sizeof(log.registered_inside)),
              ROCPROFILER_STATUS_SUCCESS);

    log.seen_before.emplace_back(log.registered_before);
    log.seen_inside.emplace_back(log.registered_inside);
    log.registered_before += 1;
    log.registered_inside *= 2;
}
}  // namespace

TEST_F(range_replay_host_run, registered_host_regions_are_restored_before_every_pass)
{
    g_plan   = {4, true};
    auto log = region_log_t{};
    ASSERT_EQ(rocprofiler_range_replay_host_track_memory(&log.registered_before,
                                                         sizeof(log.registered_before)),
              ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(rocprofiler_range_replay_host_run(17, mutate_regions, &log),
              ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(log.seen_before, (std::vector<int>{10, 10, 10, 10}));
    EXPECT_EQ(log.seen_inside, (std::vector<int>{100, 100, 100, 100}));
    EXPECT_EQ(log.registered_before, 11) << "the final state is one execution's";
    EXPECT_EQ(log.registered_inside, 200);
    EXPECT_EQ(
        g_observed.closes,
        (std::vector<rocprofiler_range_replay_status_t>{ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED}));
}

namespace
{
struct throwing_log_t
{
    uint64_t              throw_pass = 0;
    int                   value      = 0;
    std::vector<uint64_t> passes     = {};
};

void
throw_on_pass(uint64_t pass, void* user_data)
{
    auto& log = *static_cast<throwing_log_t*>(user_data);
    log.passes.emplace_back(pass);
    note_dispatch();
    if(pass == log.throw_pass) throw std::runtime_error{"range body failed"};
}
}  // namespace

TEST_F(range_replay_host_run, an_exception_on_a_replayed_pass_closes_the_range_and_propagates)
{
    g_plan         = {4, true};
    auto log       = throwing_log_t{};
    log.throw_pass = 2;
    ASSERT_EQ(rocprofiler_range_replay_host_track_memory(&log.value, sizeof(log.value)),
              ROCPROFILER_STATUS_SUCCESS);

    EXPECT_THROW(rocprofiler_range_replay_host_run(18, throw_on_pass, &log), std::runtime_error);

    EXPECT_EQ(log.passes, (std::vector<uint64_t>{0, 1, 2}));
    EXPECT_EQ(g_observed.pass_enters, (std::vector<uint64_t>{1, 2}));
    EXPECT_EQ(g_observed.pass_exits, (std::vector<uint64_t>{1, 2}))
        << "the interrupted pass's PASS enter is paired";
    EXPECT_EQ(g_observed.closes,
              (std::vector<rocprofiler_range_replay_status_t>{
                  ROCPROFILER_RANGE_REPLAY_STATUS_BODY_EXCEPTION}));
    EXPECT_EQ(rr::current_range(), nullptr);

    // The thread is free for the next range.
    g_observed = {};
    auto next  = body_log_t{};
    EXPECT_EQ(rocprofiler_range_replay_host_run(19, log_pass, &next), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(next.passes, (std::vector<uint64_t>{0, 1, 2, 3}));
    EXPECT_EQ(
        g_observed.closes,
        (std::vector<rocprofiler_range_replay_status_t>{ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED}));
}

TEST_F(range_replay_host_run, an_exception_on_the_first_pass_closes_the_range_in_either_mode)
{
    for(const bool host_replay : {true, false})
    {
        g_plan         = {4, host_replay};
        g_observed     = {};
        auto log       = throwing_log_t{};
        log.throw_pass = 0;

        EXPECT_THROW(rocprofiler_range_replay_host_run(20, throw_on_pass, &log),
                     std::runtime_error);

        EXPECT_EQ(log.passes, (std::vector<uint64_t>{0}));
        EXPECT_TRUE(g_observed.pass_enters.empty());
        EXPECT_EQ(g_observed.closes,
                  (std::vector<rocprofiler_range_replay_status_t>{
                      ROCPROFILER_RANGE_REPLAY_STATUS_BODY_EXCEPTION}))
            << "host_replay=" << host_replay;
        EXPECT_EQ(rr::current_range(), nullptr);
    }
}

TEST_F(range_replay_host_run, an_exception_from_an_unprofiled_body_propagates_and_drops_regions)
{
    g_plan   = {4, true};
    auto log = throwing_log_t{};

    ASSERT_EQ(rocprofiler_stop_context(g_context), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_THROW(rocprofiler_range_replay_host_run(
                     21,
                     [](uint64_t pass, void* user_data) {
                         auto& body_log = *static_cast<throwing_log_t*>(user_data);
                         EXPECT_EQ(rocprofiler_range_replay_host_track_memory(
                                       &body_log.value, sizeof(body_log.value)),
                                   ROCPROFILER_STATUS_SUCCESS);
                         throw_on_pass(pass, user_data);
                     },
                     &log),
                 std::runtime_error);
    ASSERT_EQ(rocprofiler_start_context(g_context), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(log.passes, (std::vector<uint64_t>{0}));
    EXPECT_TRUE(g_observed.closes.empty());
}

namespace
{
std::vector<uint64_t> g_function_passes = {};

void
plain_function_body(uint64_t pass)
{
    g_function_passes.emplace_back(pass);
    note_dispatch();
}
}  // namespace

TEST_F(range_replay_host_run, cxx_wrapper_forwards_callables_with_captures)
{
    g_plan = {3, true};

    auto passes = std::vector<uint64_t>{};
    EXPECT_EQ(rr::host_run(22,
                           [&passes](uint64_t pass) {
                               passes.emplace_back(pass);
                               note_dispatch();
                           }),
              ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(passes, (std::vector<uint64_t>{0, 1, 2}));

    // A nullary mutable lambda, passed by lvalue: called by reference, so its state is shared by
    // every pass.
    auto total   = 0;
    auto counted = [calls = 0, &total]() mutable {
        total = ++calls;
        note_dispatch();
    };
    EXPECT_EQ(rr::host_run(23, counted), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(total, 3);

    auto       const_calls = 0;
    const auto const_body  = [&const_calls](uint64_t) {
        ++const_calls;
        note_dispatch();
    };
    EXPECT_EQ(rr::host_run(24, const_body), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(const_calls, 3);

    g_function_passes.clear();
    EXPECT_EQ(rr::host_run(25, plain_function_body), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(g_function_passes, (std::vector<uint64_t>{0, 1, 2}));

    g_observed = {};
    EXPECT_THROW(rr::host_run(26,
                              []() {
                                  note_dispatch();
                                  throw std::logic_error{"lambda body failed"};
                              }),
                 std::logic_error);
    EXPECT_EQ(g_observed.closes,
              (std::vector<rocprofiler_range_replay_status_t>{
                  ROCPROFILER_RANGE_REPLAY_STATUS_BODY_EXCEPTION}));
}
