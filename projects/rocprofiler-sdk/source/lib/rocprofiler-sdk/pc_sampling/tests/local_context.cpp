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

#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/types.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

#include <memory>

using namespace rocprofiler;

namespace
{
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
context::context_array_t
as_active(const context::context& ctx)
{
    context::context_array_t active{};
    active.emplace_back(&ctx);
    return active;
}

context::context
make_pcs_context(uint64_t handle, bool globally_enabled, rocprofiler_agent_id_t agent)
{
    context::context ctx{};
    ctx.context_idx = handle;
    ctx.pc_sampler  = std::make_unique<context::pc_sampling_service>();
    ctx.pc_sampler->enabled.store(globally_enabled);
    auto session                          = std::make_shared<pc_sampling::PCSAgentSession>();
    session->context_id                   = rocprofiler_context_id_t{.handle = handle};
    ctx.pc_sampler->agent_sessions[agent] = session;
    return ctx;
}
#endif
}  // namespace

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

TEST(pc_sampling, local_stop_does_not_flip_global_enabled)
{
    const rocprofiler_agent_id_t agent{.handle = 1};
    auto                         ctx = make_pcs_context(72, true, agent);

    {
        auto                                        active = as_active(ctx);
        kernel_replay::scoped_local_context_control loop{active};
        kernel_replay::set_toggles_armed(true);
        EXPECT_EQ(kernel_replay::replay_local_stop_context({.handle = ctx.context_idx}),
                  ROCPROFILER_STATUS_SUCCESS);
        kernel_replay::set_toggles_armed(false);

        EXPECT_FALSE(*kernel_replay::local_context_override({.handle = ctx.context_idx}));
        EXPECT_TRUE(ctx.pc_sampler->enabled.load())
            << "local stop must not flip the context's global enabled flag";
        EXPECT_FALSE(pc_sampling::replay_context_should_sample(&ctx, agent))
            << "the replaying agent must honor the local stop";
    }

    EXPECT_TRUE(ctx.pc_sampler->enabled.load());
}

TEST(pc_sampling, local_start_can_select_globally_stopped_service)
{
    const rocprofiler_agent_id_t agent{.handle = 9};
    auto                         ctx = make_pcs_context(73, false, agent);

    auto                                        active = as_active(ctx);
    kernel_replay::scoped_local_context_control loop{active};

    EXPECT_FALSE(pc_sampling::replay_context_should_sample(&ctx, agent));

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_start_context({.handle = ctx.context_idx}),
              ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);

    EXPECT_FALSE(ctx.pc_sampler->enabled.load())
        << "local start must not flip the context's global enabled flag";
    EXPECT_TRUE(pc_sampling::replay_context_should_sample(&ctx, agent))
        << "kernel replay may promote PC sampling on one agent for a selected pass";
}

TEST(pc_sampling, sticky_toggle_is_idempotent)
{
    const rocprofiler_agent_id_t agent{.handle = 4};
    auto                         ctx = make_pcs_context(74, false, agent);

    auto                                        active = as_active(ctx);
    kernel_replay::scoped_local_context_control loop{active};

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_start_context({.handle = ctx.context_idx}),
              ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(kernel_replay::replay_local_start_context({.handle = ctx.context_idx}),
              ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);

    EXPECT_TRUE(pc_sampling::replay_context_should_sample(&ctx, agent));

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_stop_context({.handle = ctx.context_idx}),
              ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(kernel_replay::replay_local_stop_context({.handle = ctx.context_idx}),
              ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);

    EXPECT_FALSE(pc_sampling::replay_context_should_sample(&ctx, agent));
}

TEST(pc_sampling, reconcile_without_hsa_is_a_no_op)
{
    const rocprofiler_agent_id_t agent{.handle = 11};
    EXPECT_EQ(pc_sampling::reconcile_replay_context(agent), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(pc_sampling::restore_replay_context(agent), ROCPROFILER_STATUS_SUCCESS);
}

#endif  // ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
