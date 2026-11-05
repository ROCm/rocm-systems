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

// Unit tests for the kernel-replay CONFIG/PASS callback user_data lifecycle.
//
// The value a tool writes into user_data during CONFIG PHASE_ENTER is captured as the
// sequence-wide user_data (execute_config_phase_enter -> plan.user_data,
// kernel_replay/replay_callbacks.cpp:232-233). Every pass is then handed a fresh copy of that
// value (execute_pass_phase_enter re-seeds each pass's context slot from plan.user_data,
// replay_callbacks.cpp:285 + :292-293), and a pass's PHASE_ENTER and PHASE_EXIT share that one
// slot because the SDK reuses the same pass_context_state_t for both (execute_pass_phase_exit).
// Consequences the tests below pin down:
//   * CONFIG EXIT and every PASS callback observe the CONFIG-ENTER value by default.
//   * A write during a PASS ENTER is visible in that same PASS EXIT.
//   * That write does NOT survive into the next pass (re-seeded from the CONFIG value).
//
// This drives the real generic tracing callback dispatch (tracing::execute_phase_enter_callbacks /
// execute_phase_exit_callbacks) -- the actual mechanism that hands &user_data to the tool and lets
// a write persist in the caller's context vector -- against a hand-built callback context. The
// per-pass seed/reset around those calls mirrors the replay loop exactly (see the line references
// above). No GPU / HSA / runtime registration is involved, so the test runs unconditionally, like
// kernel_replay/tests/local_context.cpp.

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/replay_callbacks.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <vector>

/*
replay {
    user_data_t user_data = {};                  // starts 0

    CONFIG PHASE_ENTER {
        CONFIG_PHASE_ENTER_callback(..., &user_data)  // tool sets it -> X
    }

    ...

    for (each pass) {
        PASS_LOCAL_OVERRIDE = user_data;         // re-seed to X every pass

        PASS PHASE_ENTER {
            // sees X; tool may overwrite PASS_LOCAL_OVERRIDE -> Y (this pass only)
        }

        ...

        PASS PHASE_EXIT {
            // sees X, or Y if this pass (PASS PHASE_ENTER) wrote it
        }
    }                                            // Y never carries to the next pass

    CONFIG_PHASE_EXIT {}                         // sees X (pass writes don't reach here)
}
**/

namespace ctxc = rocprofiler::context;
namespace trc  = rocprofiler::tracing;
namespace kr   = rocprofiler::kernel_replay;

namespace
{
constexpr uint64_t SENTINEL_VALUE     = 0xDEADDEADDEADDEADull;  // "callback never ran" sentinel
constexpr uint64_t CONFIG_ENTER_VALUE = 0x00C0FFEEull;          // tool's CONFIG-ENTER user_data
constexpr uint64_t PASS_1_ENTER_VALUE = 0x0000BEEFull;  // tool's per-pass override on pass 1

struct pass_obs
{
    uint64_t enter_seen = SENTINEL_VALUE;
    uint64_t exit_seen  = SENTINEL_VALUE;
};

// Drives the fake tool and records what user_data value each callback observed.
struct observations
{
    uint64_t config_write     = CONFIG_ENTER_VALUE;  // value the tool writes in CONFIG PHASE_ENTER
    uint64_t config_exit_seen = SENTINEL_VALUE;      // value CONFIG PHASE_EXIT observed

    std::vector<pass_obs> passes{};
    std::map<uint64_t, uint64_t>
        pass_enter_writes{};  // pass index -> value tool writes at its ENTER
};

// Stand-in for a tool's KERNEL_REPLAY callback. Reads/writes user_data exactly where a real tool
// would: sets it once in CONFIG ENTER, records what it sees everywhere, and optionally overrides it
// in a specific pass's ENTER to prove scoping.
void
tool_callback(rocprofiler_callback_tracing_record_t record,
              rocprofiler_user_data_t*              user_data,
              void*                                 data)
{
    auto& obs = *static_cast<observations*>(data);
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY)
    {
        fprintf(
            stderr,
            "Unexpected record.kind '%d', expected ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY\n",
            record.kind);
        std::abort();
    };

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            user_data->value = obs.config_write;  // sequence-wide value, set once
        }
        else
        {
            obs.config_exit_seen = user_data->value;
        }
    }

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_PASS)
    {
        const auto* payload =
            static_cast<const rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);
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
            auto itr                    = obs.pass_enter_writes.find(pass);

            if(itr != obs.pass_enter_writes.end())
            {
                user_data->value = itr->second;
            }
        }
        else
        {
            obs.passes[pass].exit_seen = user_data->value;
        }
    }
}

// Build a callback context that subscribes to KERNEL_REPLAY CONFIG + PASS with tool_callback.
void
enable_replay_domains(ctxc::context& ctx, observations& obs)
{
    ctx.context_idx     = 1;
    ctx.callback_tracer = std::make_unique<ctxc::callback_tracing_service>();

    EXPECT_EQ(
        ctxc::add_domain(ctx.callback_tracer->domains, ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY),
        ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(ctxc::add_domain_op(ctx.callback_tracer->domains,
                                  ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                  ROCPROFILER_KERNEL_REPLAY_CONFIG),
              ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(ctxc::add_domain_op(ctx.callback_tracer->domains,
                                  ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                  ROCPROFILER_KERNEL_REPLAY_PASS),
              ROCPROFILER_STATUS_SUCCESS);

    ctx.callback_tracer->callback_data.at(ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) = {
        &tool_callback, &obs};
}

// One replay "dispatch": a CONFIG enter/exit pair, then n_passes of PASS enter/exit. Mirrors the
// SDK's user_data handling around the real tracing dispatch helpers (line references in the file
// header) so the test exercises the real persistence mechanism, not a re-implementation of it.
void
run_replay(observations& obs, uint64_t n_passes)
{
    obs.passes.assign(n_passes, pass_obs{});

    auto ctx = ctxc::context{};
    enable_replay_domains(ctx, obs);

    // CONFIG: tool writes user_data in ENTER; capture it as the sequence-wide value the way
    // execute_config_phase_enter does (replay_callbacks.cpp:232-233). EXIT then observes it.
    uint64_t plan_user_data = 0;
    {
        auto cfg = trc::callback_context_data_vec_t{};
        cfg.emplace_back(trc::callback_context_data{&ctx, rocprofiler_callback_tracing_record_t{}});
        auto corr = trc::external_correlation_id_map_t{};
        corr.emplace(&ctx, trc::empty_user_data);

        auto payload = rocprofiler_callback_tracing_kernel_replay_data_t{};

        trc::execute_phase_enter_callbacks(cfg,
                                           0 /*thr_id=*/,
                                           0 /*internal_corr_id=*/,
                                           corr,
                                           0 /*ancestor_corr_id=*/,
                                           ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                           ROCPROFILER_KERNEL_REPLAY_CONFIG,
                                           payload);

        plan_user_data = cfg.front().user_data.value;  // capture (replay_callbacks.cpp:232-233)

        trc::execute_phase_exit_callbacks(cfg,
                                          corr,
                                          ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                          ROCPROFILER_KERNEL_REPLAY_CONFIG,
                                          payload);
    }

    // PASS loop. Each pass gets a fresh context vector (execute_pass_phase_enter's
    // out_pass_state = {}, replay_callbacks.cpp:285) re-seeded from plan.user_data (:292-293), and
    // ENTER + EXIT reuse that same vector (execute_pass_phase_exit reuses pass_state.contexts).
    for(uint64_t pass = 0; pass < n_passes; ++pass)
    {
        auto pc = trc::callback_context_data_vec_t{};
        pc.emplace_back(trc::callback_context_data{&ctx, rocprofiler_callback_tracing_record_t{}});
        auto corr = trc::external_correlation_id_map_t{};
        corr.emplace(&ctx, trc::empty_user_data);

        for(auto& itr : pc)
            itr.user_data.value = plan_user_data;  // seed from the CONFIG value every pass

        auto payload         = rocprofiler_callback_tracing_kernel_replay_data_t{};
        payload.current_pass = pass;
        payload.total_passes = n_passes;

        trc::execute_phase_enter_callbacks(pc,
                                           0 /*thr_id=*/,
                                           0 /*internal_corr_id=*/,
                                           corr,
                                           0 /*ancestor_corr_id=*/,
                                           ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                           ROCPROFILER_KERNEL_REPLAY_PASS,
                                           payload);
        trc::execute_phase_exit_callbacks(pc,
                                          corr,
                                          ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                          ROCPROFILER_KERNEL_REPLAY_PASS,
                                          payload);
    }
}

// Function-pointer callbacks for the replay_continue decision. Plain function pointers can't
// capture, so they record into a file-scope pointer the test sets before invoking the decision.
struct continue_call
{
    uint64_t current_pass{};
    uint64_t user_data{};
    int      returned{};
};

std::vector<continue_call>* g_continue_calls = nullptr;

int
continue_always(rocprofiler_kernel_dispatch_info_t /*dispatch_info*/,
                uint64_t current_pass,
                uint64_t /*total_passes*/,
                rocprofiler_user_data_t user_data)
{
    if(g_continue_calls) g_continue_calls->push_back({current_pass, user_data.value, 1});
    return 1;  // continue
}

int
stop_now(rocprofiler_kernel_dispatch_info_t /*dispatch_info*/,
         uint64_t current_pass,
         uint64_t /*total_passes*/,
         rocprofiler_user_data_t user_data)
{
    if(g_continue_calls) g_continue_calls->push_back({current_pass, user_data.value, 0});
    return 0;  // stop
}
}  // namespace

// The value set in CONFIG PHASE_ENTER is what CONFIG PHASE_EXIT sees and what seeds every pass's
// ENTER and EXIT when the tool does not touch user_data inside a pass.
TEST(kernel_replay_phases, config_user_data_flows_to_config_exit_and_every_pass)
{
    observations obs{};
    obs.config_write = CONFIG_ENTER_VALUE;

    run_replay(obs, /*n_passes=*/3);

    EXPECT_EQ(obs.config_exit_seen, CONFIG_ENTER_VALUE);
    ASSERT_EQ(obs.passes.size(), 3u);
    for(uint64_t p = 0; p < obs.passes.size(); ++p)
    {
        EXPECT_EQ(obs.passes[p].enter_seen, CONFIG_ENTER_VALUE) << "pass " << p << " enter";
        EXPECT_EQ(obs.passes[p].exit_seen, CONFIG_ENTER_VALUE) << "pass " << p << " exit";
    }
}

// A write during a PASS ENTER is visible in that same PASS EXIT (shared slot) but is scoped to the
// pass: the next pass is re-seeded from the CONFIG value and never sees the prior pass's write.
TEST(kernel_replay_phases, pass_enter_user_data_write_scoped_to_its_own_pass)
{
    observations obs{};
    obs.config_write = CONFIG_ENTER_VALUE;
    obs.pass_enter_writes[1] =
        PASS_1_ENTER_VALUE;  // tool overrides user_data during pass 1's ENTER

    run_replay(obs, /*n_passes=*/3);

    // pass 0: untouched -> CONFIG value on both ends.
    EXPECT_EQ(obs.passes[0].enter_seen, CONFIG_ENTER_VALUE);
    EXPECT_EQ(obs.passes[0].exit_seen, CONFIG_ENTER_VALUE);

    // pass 1: ENTER still inherits the CONFIG seed (recorded before the tool writes); EXIT sees the
    // write because ENTER and EXIT share the same slot.
    EXPECT_EQ(obs.passes[1].enter_seen, CONFIG_ENTER_VALUE);
    EXPECT_EQ(obs.passes[1].exit_seen, PASS_1_ENTER_VALUE);

    // pass 2: re-seeded from CONFIG -> the pass-1 write did not persist.
    EXPECT_EQ(obs.passes[2].enter_seen, CONFIG_ENTER_VALUE);
    EXPECT_EQ(obs.passes[2].exit_seen, CONFIG_ENTER_VALUE);
}

// should_continue_replay uses THIS pass's replay_continue (the config default unless the tool
// overrode it for the pass in PASS PHASE_EXIT) and hands it the pass-scoped user_data -- the copy
// that pass's PASS PHASE_EXIT saw, not the sequence-wide CONFIG value. The final pass of a fixed
// loop stops without consulting the callback.
TEST(kernel_replay_phases, replay_continue_per_pass_override_and_pass_scoped_user_data)
{
    std::vector<continue_call> calls{};
    g_continue_calls = &calls;

    auto plan            = kr::replay_plan_t{};
    plan.total_passes    = 5;
    plan.indefinite      = false;
    plan.replay_continue = &continue_always;    // config default: keep going
    plan.user_data.value = CONFIG_ENTER_VALUE;  // sequence-wide value (must NOT be what cb sees)

    // pass 0: no per-pass override -> the config default runs and sees this pass's user_data.
    {
        auto ps            = kr::pass_context_state_t{};
        ps.replay_continue = plan.replay_continue;  // seeded from config at PASS ENTER
        ps.user_data.value = CONFIG_ENTER_VALUE;    // pass-scoped copy (equals config for pass 0)
        EXPECT_TRUE(kr::should_continue_replay(plan, ps, /*current_pass=*/0, /*is_final=*/false));
    }

    // pass 1: tool overrode replay_continue in PASS EXIT -> stop; it sees the pass-scoped value.
    {
        auto ps            = kr::pass_context_state_t{};
        ps.replay_continue = &stop_now;           // per-pass override
        ps.user_data.value = PASS_1_ENTER_VALUE;  // pass-scoped copy for this pass
        EXPECT_FALSE(kr::should_continue_replay(plan, ps, /*current_pass=*/1, /*is_final=*/false));
    }

    // final pass of a fixed loop: stops regardless, and the callback is never consulted.
    {
        auto ps            = kr::pass_context_state_t{};
        ps.replay_continue = &continue_always;
        ps.user_data.value = CONFIG_ENTER_VALUE;
        EXPECT_FALSE(kr::should_continue_replay(plan, ps, /*current_pass=*/4, /*is_final=*/true));
    }

    // Only pass 0 (continue) and pass 1 (stop) consulted the callback; the final pass
    // short-circuited. Each callback received its own pass's user_data -- pass 1 proves it is
    // pass-scoped, not config.
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0].current_pass, 0u);
    EXPECT_EQ(calls[0].user_data, CONFIG_ENTER_VALUE);
    EXPECT_EQ(calls[0].returned, 1);
    EXPECT_EQ(calls[1].current_pass, 1u);
    EXPECT_EQ(calls[1].user_data, PASS_1_ENTER_VALUE);  // pass-scoped, not plan.user_data
    EXPECT_EQ(calls[1].returned, 0);

    g_continue_calls = nullptr;
}
