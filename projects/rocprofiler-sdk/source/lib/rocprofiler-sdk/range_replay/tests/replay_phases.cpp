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

// Unit tests for the range-replay pass loop and callback lifecycle, the counterpart of
// kernel_replay/tests/replay_phases.cpp.
//
// Two things are pinned here.
//
// The pass-loop decision. should_continue_replay() is driven entirely from range_plan_t, so it is
// testable outright: no contexts, no tracing dispatch, no device. It decides how many times a
// recorded range re-executes, which is the difference between a tool getting the passes it asked
// for and a tool getting silently truncated or looping forever.
//
// The user_data lifecycle. The value a tool writes during CONFIG PHASE_ENTER becomes the range's
// sequence-wide user_data (execute_config_callback captures it into plan.user_data), and every PASS
// and the CLOSE callback are re-seeded from it. A write inside one pass is therefore scoped to that
// pass, and never reaches CLOSE. CLOSE is also delivered to the contexts resolved at CONFIG rather
// than the ones active at close time, so a tool that stops its context mid-range still gets the
// range's terminal notification -- the same rule that lets an in-flight dispatch complete after its
// context stops.
//
// execute_pass_phase_exit() and execute_close_callback() take their contexts from arguments, so the
// tests below call the real functions. execute_pass_phase_enter() resolves its contexts from the
// global active-context registry, which a unit test has no way to populate, so its ENTER is driven
// through the same tracing helpers it uses (as kernel_replay/tests/replay_phases.cpp does).
//
// No GPU, HSA, or runtime registration is involved, so this runs unconditionally.

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"
#include "lib/rocprofiler-sdk/range_replay/replay_callbacks.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <vector>

namespace ctxc = rocprofiler::context;
namespace trc  = rocprofiler::tracing;
namespace rr   = rocprofiler::range_replay;

namespace
{
constexpr uint64_t SENTINEL_VALUE     = 0xDEADDEADDEADDEADull;  // "callback never ran" sentinel
constexpr uint64_t CONFIG_ENTER_VALUE = 0x00C0FFEEull;          // tool's CONFIG-ENTER user_data
constexpr uint64_t PASS_1_ENTER_VALUE = 0x0000BEEFull;  // tool's per-pass override on pass 1
constexpr uint64_t RANGE_ID           = 0x5A5Aull;

// ---------------------------------------------------------------------------
// should_continue_replay: the pass loop
// ---------------------------------------------------------------------------

struct continue_call
{
    uint64_t range_id{};
    uint64_t current_pass{};
    uint64_t total_passes{};
    uint64_t user_data{};
};

// Plain function pointers cannot capture, so the continue callbacks record into a file-scope
// pointer the test sets before invoking the decision.
std::vector<continue_call>* g_continue_calls = nullptr;

void
record_continue_call(uint64_t                range_id,
                     uint64_t                current_pass,
                     uint64_t                total_passes,
                     rocprofiler_user_data_t user_data)
{
    if(g_continue_calls)
        g_continue_calls->push_back({range_id, current_pass, total_passes, user_data.value});
}

int
continue_always(uint64_t                range_id,
                uint64_t                current_pass,
                uint64_t                total_passes,
                rocprofiler_user_data_t user_data)
{
    record_continue_call(range_id, current_pass, total_passes, user_data);
    return 1;
}

int
stop_after_pass_two(uint64_t                range_id,
                    uint64_t                current_pass,
                    uint64_t                total_passes,
                    rocprofiler_user_data_t user_data)
{
    record_continue_call(range_id, current_pass, total_passes, user_data);
    return (current_pass < 2) ? 1 : 0;
}

rr::range_plan_t
make_plan(uint64_t total_passes, bool indefinite)
{
    auto plan                 = rr::range_plan_t{};
    plan.config_data.range_id = RANGE_ID;
    plan.total_passes         = total_passes;
    plan.indefinite           = indefinite;
    plan.replay_requested     = indefinite || total_passes > 1;
    plan.user_data.value      = CONFIG_ENTER_VALUE;
    return plan;
}

// Run the executor's pass loop against a plan and report which passes it would have run. This is
// the loop in executor.cpp::execute_range, with the device work removed: passes are numbered from
// 1 because pass 0 is the application's own execution.
std::vector<uint64_t>
passes_run(const rr::range_plan_t& plan, uint64_t safety_limit = 32)
{
    auto ran = std::vector<uint64_t>{};
    for(uint64_t pass = 1; pass <= safety_limit; ++pass)
    {
        const bool is_final = !plan.indefinite && (pass == plan.total_passes - 1);
        ran.emplace_back(pass);
        if(!rr::should_continue_replay(plan, pass, is_final)) break;
    }
    return ran;
}

// ---------------------------------------------------------------------------
// user_data lifecycle
// ---------------------------------------------------------------------------

struct pass_obs
{
    uint64_t enter_seen = SENTINEL_VALUE;
    uint64_t exit_seen  = SENTINEL_VALUE;
};

struct observations
{
    uint64_t config_write     = CONFIG_ENTER_VALUE;  // value the tool writes in CONFIG PHASE_ENTER
    uint64_t config_exit_seen = SENTINEL_VALUE;

    std::vector<pass_obs>        passes{};
    std::map<uint64_t, uint64_t> pass_enter_writes{};  // pass index -> value written at its ENTER

    uint64_t                          close_enter_seen  = SENTINEL_VALUE;
    uint64_t                          close_exit_seen   = SENTINEL_VALUE;
    uint64_t                          close_range_id    = 0;
    uint64_t                          close_dispatches  = 0;
    uint64_t                          close_divergence  = 0;
    rocprofiler_range_replay_status_t close_status      = ROCPROFILER_RANGE_REPLAY_STATUS_LAST;
    int                               close_enter_count = 0;
    int                               close_exit_count  = 0;
};

// Stand-in for a tool's RANGE_REPLAY callback. Reads and writes user_data exactly where a real tool
// would: sets it once in CONFIG ENTER, records what it sees everywhere, and optionally overrides it
// in a specific pass's ENTER to prove scoping.
void
tool_callback(rocprofiler_callback_tracing_record_t record,
              rocprofiler_user_data_t*              user_data,
              void*                                 data)
{
    auto& obs = *static_cast<observations*>(data);
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY)
    {
        ADD_FAILURE() << "unexpected record.kind " << record.kind
                      << ", expected ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY";
        return;
    }

    if(record.operation == ROCPROFILER_RANGE_REPLAY_CONFIG)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
            user_data->value = obs.config_write;  // sequence-wide value, set once
        else
            obs.config_exit_seen = user_data->value;
        return;
    }

    if(record.operation == ROCPROFILER_RANGE_REPLAY_PASS)
    {
        const auto* payload =
            static_cast<const rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
        const auto pass = payload->current_pass;

        if(pass >= obs.passes.size())
        {
            ADD_FAILURE() << "unexpected pass index " << pass;
            return;
        }

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            // Record what the pass inherited *before* the tool touches it this pass.
            obs.passes[pass].enter_seen = user_data->value;

            auto itr = obs.pass_enter_writes.find(pass);
            if(itr != obs.pass_enter_writes.end()) user_data->value = itr->second;
        }
        else
        {
            obs.passes[pass].exit_seen = user_data->value;
        }
        return;
    }

    if(record.operation == ROCPROFILER_RANGE_REPLAY_CLOSE)
    {
        const auto* payload =
            static_cast<const rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            ++obs.close_enter_count;
            obs.close_enter_seen = user_data->value;
            obs.close_range_id   = payload->range_id;
            obs.close_dispatches = payload->dispatch_count;
            obs.close_divergence = payload->divergence_count;
            obs.close_status     = payload->status;
        }
        else
        {
            ++obs.close_exit_count;
            obs.close_exit_seen = user_data->value;
        }
    }
}

// Build a callback context subscribed to every RANGE_REPLAY operation with tool_callback.
void
enable_range_replay_domains(ctxc::context& ctx, observations& obs)
{
    ctx.context_idx     = 1;
    ctx.callback_tracer = std::make_unique<ctxc::callback_tracing_service>();

    ASSERT_EQ(
        ctxc::add_domain(ctx.callback_tracer->domains, ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY),
        ROCPROFILER_STATUS_SUCCESS);

    for(auto operation : {ROCPROFILER_RANGE_REPLAY_CONFIG,
                          ROCPROFILER_RANGE_REPLAY_PASS,
                          ROCPROFILER_RANGE_REPLAY_CLOSE})
    {
        ASSERT_EQ(
            ctxc::add_domain_op(
                ctx.callback_tracer->domains, ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY, operation),
            ROCPROFILER_STATUS_SUCCESS);
    }

    ctx.callback_tracer->callback_data.at(ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) = {
        &tool_callback, &obs};
}

trc::callback_context_data_vec_t
one_context(ctxc::context& ctx)
{
    auto out = trc::callback_context_data_vec_t{};
    out.emplace_back(trc::callback_context_data{&ctx, rocprofiler_callback_tracing_record_t{}});
    return out;
}

trc::external_correlation_id_map_t
one_correlation(ctxc::context& ctx)
{
    auto out = trc::external_correlation_id_map_t{};
    out.emplace(&ctx, trc::empty_user_data);
    return out;
}

// One range: a CONFIG enter/exit pair, n_passes of PASS enter/exit, then CLOSE. Mirrors the SDK's
// user_data handling around the real dispatch helpers, and calls the real execute_pass_phase_exit
// and execute_close_callback.
void
run_range(observations& obs, uint64_t n_passes, rocprofiler_range_replay_status_t close_status)
{
    // Passes are numbered from 1, so index n_passes must be addressable.
    obs.passes.assign(n_passes + 1, pass_obs{});

    auto ctx = ctxc::context{};
    enable_range_replay_domains(ctx, obs);

    auto plan = make_plan(n_passes + 1, /*indefinite=*/false);

    // CONFIG: the tool writes user_data in ENTER; capture it as the sequence-wide value the way
    // execute_config_callback does. EXIT then observes it.
    {
        auto cfg  = one_context(ctx);
        auto corr = one_correlation(ctx);

        trc::execute_phase_enter_callbacks(cfg,
                                           0 /*thr_id*/,
                                           0 /*internal_corr_id*/,
                                           corr,
                                           0 /*ancestor_corr_id*/,
                                           ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                           ROCPROFILER_RANGE_REPLAY_CONFIG,
                                           plan.config_data);

        plan.user_data = cfg.front().user_data;

        trc::execute_phase_exit_callbacks(cfg,
                                          corr,
                                          ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                          ROCPROFILER_RANGE_REPLAY_CONFIG,
                                          plan.config_data);
    }

    // CLOSE's contexts are resolved here, during CONFIG, exactly as execute_config_callback does.
    plan.close_contexts        = one_context(ctx);
    plan.close_correlation_ids = one_correlation(ctx);

    for(uint64_t pass = 1; pass <= n_passes; ++pass)
    {
        auto pass_state                     = rr::pass_context_state_t{};
        pass_state.contexts                 = one_context(ctx);
        pass_state.external_correlation_ids = one_correlation(ctx);

        // Re-seed from the CONFIG value every pass, as execute_pass_phase_enter does.
        for(auto& itr : pass_state.contexts)
            itr.user_data = plan.user_data;

        auto pass_data         = rocprofiler_callback_tracing_range_replay_data_t{};
        pass_data.range_id     = plan.config_data.range_id;
        pass_data.current_pass = pass;
        pass_data.total_passes = plan.total_passes;

        trc::execute_phase_enter_callbacks(pass_state.contexts,
                                           0 /*thr_id*/,
                                           0 /*internal_corr_id*/,
                                           pass_state.external_correlation_ids,
                                           0 /*ancestor_corr_id*/,
                                           ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                           ROCPROFILER_RANGE_REPLAY_PASS,
                                           pass_data);

        rr::execute_pass_phase_exit(
            plan, pass, rocprofiler_agent_id_t{.handle = 0}, /*dispatch_count=*/4, pass_state);
    }

    rr::execute_close_callback(plan,
                               close_status,
                               rocprofiler_agent_id_t{.handle = 0},
                               /*dispatch_count=*/4,
                               /*divergence_count=*/0,
                               0 /*thr_id*/,
                               0 /*internal_corr_id*/,
                               0 /*ancestor_corr_id*/);
}
}  // namespace

// ---------------------------------------------------------------------------
// Pass loop
// ---------------------------------------------------------------------------

// A fixed pass count of N means the application's run plus N-1 re-executions. The loop must stop on
// its own at the last one, without a continue callback: a tool that only supplies pass_count_cb
// gets exactly the passes it asked for.
TEST(range_replay_phases, fixed_pass_count_runs_every_pass_but_the_application_run)
{
    EXPECT_EQ(passes_run(make_plan(3, false)), (std::vector<uint64_t>{1, 2}));
    EXPECT_EQ(passes_run(make_plan(2, false)), (std::vector<uint64_t>{1}));
    EXPECT_EQ(passes_run(make_plan(5, false)), (std::vector<uint64_t>{1, 2, 3, 4}));
}

// The final pass of a fixed loop stops without consulting the continue callback. A tool that
// returns "keep going" from every call must still not exceed the count it asked for.
TEST(range_replay_phases, continue_callback_cannot_extend_a_fixed_loop)
{
    auto calls       = std::vector<continue_call>{};
    g_continue_calls = &calls;

    auto plan               = make_plan(3, /*indefinite=*/false);
    plan.replay_continue_cb = &continue_always;

    EXPECT_EQ(passes_run(plan), (std::vector<uint64_t>{1, 2}));

    // Consulted for pass 1 only; pass 2 is final and short-circuits.
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].current_pass, 1u);

    g_continue_calls = nullptr;
}

// The continue callback may end a fixed loop early, and it receives the range's id, the pass just
// finished, the total it was told about, and the sequence-wide user_data from CONFIG.
TEST(range_replay_phases, continue_callback_can_end_a_fixed_loop_early)
{
    auto calls       = std::vector<continue_call>{};
    g_continue_calls = &calls;

    auto plan               = make_plan(10, /*indefinite=*/false);
    plan.replay_continue_cb = &stop_after_pass_two;

    EXPECT_EQ(passes_run(plan), (std::vector<uint64_t>{1, 2}));

    ASSERT_EQ(calls.size(), 2u);
    for(const auto& call : calls)
    {
        EXPECT_EQ(call.range_id, RANGE_ID);
        EXPECT_EQ(call.total_passes, 10u);
        EXPECT_EQ(call.user_data, CONFIG_ENTER_VALUE);
    }
    EXPECT_EQ(calls[0].current_pass, 1u);
    EXPECT_EQ(calls[1].current_pass, 2u);

    g_continue_calls = nullptr;
}

// An indefinite loop (pass_count_cb returned 0) runs until the continue callback says stop. The
// total it reports must be 0 rather than a stale count, so the tool can tell the two modes apart
// from inside the callback.
TEST(range_replay_phases, indefinite_loop_runs_until_the_callback_stops_it)
{
    auto calls       = std::vector<continue_call>{};
    g_continue_calls = &calls;

    auto plan               = make_plan(0, /*indefinite=*/true);
    plan.replay_continue_cb = &stop_after_pass_two;

    EXPECT_EQ(passes_run(plan), (std::vector<uint64_t>{1, 2}));

    ASSERT_EQ(calls.size(), 2u);
    for(const auto& call : calls)
        EXPECT_EQ(call.total_passes, 0u) << "indefinite loops report 0, not a count";

    g_continue_calls = nullptr;
}

// Without a continue callback an indefinite loop never terminates on its own. execute_config_
// callback refuses that combination (pass_count_cb returning 0 with no replay_continue_cb leaves
// replay_requested false), so this pins the decision function's half of that contract: it is the
// admission check, not the loop, that prevents the spin.
TEST(range_replay_phases, indefinite_loop_without_a_callback_never_stops_itself)
{
    auto plan = make_plan(0, /*indefinite=*/true);
    ASSERT_EQ(plan.replay_continue_cb, nullptr);

    for(uint64_t pass = 1; pass <= 8; ++pass)
        EXPECT_TRUE(rr::should_continue_replay(plan, pass, /*is_final_pass=*/false));
}

// ---------------------------------------------------------------------------
// user_data lifecycle
// ---------------------------------------------------------------------------

// The value set in CONFIG PHASE_ENTER is what CONFIG PHASE_EXIT sees, what seeds every pass, and
// what CLOSE receives.
TEST(range_replay_phases, config_user_data_flows_to_every_pass_and_to_close)
{
    observations obs{};
    obs.config_write = CONFIG_ENTER_VALUE;

    run_range(obs, /*n_passes=*/3, ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);

    EXPECT_EQ(obs.config_exit_seen, CONFIG_ENTER_VALUE);

    for(uint64_t pass = 1; pass <= 3; ++pass)
    {
        EXPECT_EQ(obs.passes[pass].enter_seen, CONFIG_ENTER_VALUE) << "pass " << pass << " enter";
        EXPECT_EQ(obs.passes[pass].exit_seen, CONFIG_ENTER_VALUE) << "pass " << pass << " exit";
    }

    EXPECT_EQ(obs.close_enter_seen, CONFIG_ENTER_VALUE);
    EXPECT_EQ(obs.close_exit_seen, CONFIG_ENTER_VALUE);
}

// A write during a PASS ENTER is visible in that same PASS EXIT (they share one slot) but is scoped
// to the pass: the next pass is re-seeded from the CONFIG value, and CLOSE never sees it either.
TEST(range_replay_phases, pass_user_data_write_is_scoped_to_its_own_pass)
{
    observations obs{};
    obs.config_write         = CONFIG_ENTER_VALUE;
    obs.pass_enter_writes[1] = PASS_1_ENTER_VALUE;

    run_range(obs, /*n_passes=*/3, ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);

    // pass 1: ENTER still inherits the CONFIG seed (recorded before the tool writes); EXIT sees the
    // write because ENTER and EXIT share the same slot.
    EXPECT_EQ(obs.passes[1].enter_seen, CONFIG_ENTER_VALUE);
    EXPECT_EQ(obs.passes[1].exit_seen, PASS_1_ENTER_VALUE);

    // pass 2: re-seeded from CONFIG -> the pass-1 write did not persist.
    EXPECT_EQ(obs.passes[2].enter_seen, CONFIG_ENTER_VALUE);
    EXPECT_EQ(obs.passes[2].exit_seen, CONFIG_ENTER_VALUE);

    EXPECT_EQ(obs.close_enter_seen, CONFIG_ENTER_VALUE);
    EXPECT_EQ(obs.close_exit_seen, CONFIG_ENTER_VALUE);
}

// CLOSE is the range's terminal notification and reports the outcome the tool acts on. It is
// delivered exactly once, in both phases, carrying the range id and the decline reason.
TEST(range_replay_phases, close_reports_the_outcome_once)
{
    observations obs{};

    run_range(obs, /*n_passes=*/1, ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE);

    EXPECT_EQ(obs.close_enter_count, 1);
    EXPECT_EQ(obs.close_exit_count, 1);
    EXPECT_EQ(obs.close_range_id, RANGE_ID);
    EXPECT_EQ(obs.close_status, ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE);
    EXPECT_EQ(obs.close_dispatches, 4u);
    EXPECT_EQ(obs.close_divergence, 0u);
}

// A range that was never replayed still gets a CLOSE: the tool asked for the range and is owed the
// reason it did not run, otherwise a declined range is indistinguishable from a range whose
// callbacks were dropped.
TEST(range_replay_phases, declined_range_still_reports_close)
{
    observations obs{};

    run_range(obs, /*n_passes=*/0, ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT);

    EXPECT_EQ(obs.close_enter_count, 1);
    EXPECT_EQ(obs.close_status, ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT);
}

// A plan whose CLOSE contexts were never resolved delivers nothing rather than dereferencing an
// empty context vector. This is the path a range takes when no context subscribed to CLOSE.
TEST(range_replay_phases, close_without_subscribed_contexts_is_a_no_op)
{
    auto plan = make_plan(3, /*indefinite=*/false);
    ASSERT_TRUE(plan.close_contexts.empty());

    rr::execute_close_callback(plan,
                               ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED,
                               rocprofiler_agent_id_t{.handle = 0},
                               /*dispatch_count=*/1,
                               /*divergence_count=*/0,
                               0 /*thr_id*/,
                               0 /*internal_corr_id*/,
                               0 /*ancestor_corr_id*/);
}
