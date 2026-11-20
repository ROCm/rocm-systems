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

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/types.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

#include <memory>

using namespace rocprofiler;

namespace
{
struct replay_session_fixture
{
    rocprofiler_context_id_t                      ctx_id{.handle = 0};
    context::context*                             ctx = nullptr;
    rocprofiler_agent_id_t                        agent_id{.handle = 424242};
    std::shared_ptr<pc_sampling::PCSAgentSession> session{};
    bool                                          prev_hsa_ready = false;

    replay_session_fixture()
    {
        registration::init_logging();
        registration::set_init_status(-1);
        context::push_client(1);

        auto allocated = context::allocate_context();
        EXPECT_TRUE(allocated.has_value());
        ctx_id = *allocated;
        ctx    = context::get_mutable_registered_context(ctx_id);
        EXPECT_NE(ctx, nullptr);

        ctx->pc_sampler = std::make_unique<context::pc_sampling_service>();
        ctx->pc_sampler->enabled.store(true);

        session             = std::make_shared<pc_sampling::PCSAgentSession>();
        session->context_id = ctx_id;
        session->client_idx = ctx->client_idx;
        session->active.store(true);
        // Leave hsa_agent unset so start/stop are no-ops and these tests stay host-only.

        pc_sampling::get_global_pc_sampling_sessions().wlock(
            [this](auto& sessions) { sessions[agent_id] = session; });
        ctx->pc_sampler->agent_sessions[agent_id] = session;
        prev_hsa_ready = pc_sampling::is_hsa_initialized().exchange(true);
    }

    ~replay_session_fixture()
    {
        pc_sampling::get_global_pc_sampling_sessions().wlock(
            [this](auto& sessions) { sessions.erase(agent_id); });
        if(ctx && ctx->pc_sampler) ctx->pc_sampler->agent_sessions.clear();
        pc_sampling::is_hsa_initialized().store(prev_hsa_ready);
        registration::set_init_status(1);
        registration::finalize();
        context::pop_client(1);
    }

    context::context_array_t as_active() const
    {
        context::context_array_t active{};
        active.emplace_back(ctx);
        return active;
    }
};
}  // namespace

TEST(pc_sampling, sync_replay_service_suspends_session_on_local_stop)
{
    replay_session_fixture                      fx{};
    auto                                        active = fx.as_active();
    kernel_replay::scoped_local_context_control loop{active};

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_disable_context(fx.ctx_id), ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);

    EXPECT_EQ(pc_sampling::sync_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_TRUE(fx.session->replay_suspended.load());
    EXPECT_FALSE(fx.session->active.load());
    EXPECT_TRUE(fx.ctx->pc_sampler->enabled.load())
        << "Pass sync must not mutate the context's global start state";
}

TEST(pc_sampling, sync_replay_service_restarts_session_on_local_start)
{
    replay_session_fixture                      fx{};
    auto                                        active = fx.as_active();
    kernel_replay::scoped_local_context_control loop{active};

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_disable_context(fx.ctx_id), ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);
    EXPECT_EQ(pc_sampling::sync_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_TRUE(fx.session->replay_suspended.load());
    EXPECT_FALSE(fx.session->active.load());

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_enable_context(fx.ctx_id), ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);
    EXPECT_EQ(pc_sampling::sync_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_FALSE(fx.session->replay_suspended.load());
    EXPECT_TRUE(fx.session->active.load());
}

TEST(pc_sampling, restore_replay_service_clears_suspend_and_restores_global_intent)
{
    replay_session_fixture                      fx{};
    auto                                        active = fx.as_active();
    kernel_replay::scoped_local_context_control loop{active};

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_disable_context(fx.ctx_id), ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);
    EXPECT_EQ(pc_sampling::sync_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
    ASSERT_TRUE(fx.session->replay_suspended.load());
    ASSERT_FALSE(fx.session->active.load());

    EXPECT_EQ(pc_sampling::restore_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_FALSE(fx.session->replay_suspended.load());
    EXPECT_TRUE(fx.session->active.load())
        << "Restore must bring the session back when the context is still globally started";
}

TEST(pc_sampling, restore_replay_service_keeps_session_off_when_globally_stopped)
{
    replay_session_fixture                      fx{};
    auto                                        active = fx.as_active();
    kernel_replay::scoped_local_context_control loop{active};

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_disable_context(fx.ctx_id), ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);
    EXPECT_EQ(pc_sampling::sync_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);

    fx.ctx->pc_sampler->enabled.store(false);
    EXPECT_EQ(pc_sampling::restore_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_FALSE(fx.session->replay_suspended.load());
    EXPECT_FALSE(fx.session->active.load())
        << "Restore must follow the globally stopped context, not the pre-replay session state";
}

TEST(pc_sampling, replay_suspend_blocks_global_start_service)
{
    replay_session_fixture                      fx{};
    auto                                        active = fx.as_active();
    kernel_replay::scoped_local_context_control loop{active};

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_disable_context(fx.ctx_id), ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);
    EXPECT_EQ(pc_sampling::sync_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
    ASSERT_TRUE(fx.session->replay_suspended.load());
    ASSERT_FALSE(fx.session->active.load());

    EXPECT_EQ(pc_sampling::stop_service(fx.ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_FALSE(fx.ctx->pc_sampler->enabled.load());
    EXPECT_EQ(pc_sampling::start_service(fx.ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_TRUE(fx.ctx->pc_sampler->enabled.load());
    EXPECT_FALSE(fx.session->active.load())
        << "A global start must not restart hardware while the replay pass keeps PCS suspended";

    EXPECT_EQ(pc_sampling::restore_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_TRUE(fx.session->active.load());
}

TEST(pc_sampling, sync_replay_service_keeps_sticky_local_stop_across_passes)
{
    replay_session_fixture                      fx{};
    auto                                        active = fx.as_active();
    kernel_replay::scoped_local_context_control loop{active};

    // Pass 0: no toggle → session stays active.
    EXPECT_EQ(pc_sampling::sync_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_FALSE(fx.session->replay_suspended.load());
    EXPECT_TRUE(fx.session->active.load());

    // Pass 1 enter: sticky local stop.
    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_disable_context(fx.ctx_id), ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);
    EXPECT_EQ(pc_sampling::sync_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_TRUE(fx.session->replay_suspended.load());
    EXPECT_FALSE(fx.session->active.load());

    // Passes 2 and 3: no new toggle; sticky stop keeps the session suspended.
    for(int pass = 2; pass < 4; ++pass)
    {
        (void) pass;
        EXPECT_EQ(pc_sampling::sync_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_TRUE(fx.session->replay_suspended.load());
        EXPECT_FALSE(fx.session->active.load());
    }

    EXPECT_EQ(pc_sampling::restore_replay_service(fx.agent_id), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_FALSE(fx.session->replay_suspended.load());
    EXPECT_TRUE(fx.session->active.load());
}

TEST(pc_sampling, sync_replay_service_is_noop_without_session)
{
    EXPECT_EQ(pc_sampling::sync_replay_service({.handle = 0xdeadbeef}), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(pc_sampling::restore_replay_service({.handle = 0xdeadbeef}),
              ROCPROFILER_STATUS_SUCCESS);
}

TEST(pc_sampling, sync_replay_service_reports_missing_context)
{
    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    const rocprofiler_agent_id_t agent_id{.handle = 777777};
    auto                         session = std::make_shared<pc_sampling::PCSAgentSession>();
    session->context_id                  = rocprofiler_context_id_t{.handle = 0xffffffff};
    session->active.store(true);

    pc_sampling::get_global_pc_sampling_sessions().wlock(
        [&](auto& sessions) { sessions[agent_id] = session; });

    EXPECT_EQ(pc_sampling::sync_replay_service(agent_id),
              ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND);
    EXPECT_EQ(pc_sampling::restore_replay_service(agent_id),
              ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND);

    pc_sampling::get_global_pc_sampling_sessions().wlock(
        [&](auto& sessions) { sessions.erase(agent_id); });
    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
}
