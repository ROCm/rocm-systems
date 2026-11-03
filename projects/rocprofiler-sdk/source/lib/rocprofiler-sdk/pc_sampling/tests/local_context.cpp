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
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

#include <memory>

using namespace rocprofiler;

namespace
{
context::context_array_t
as_active(const context::context& ctx)
{
    context::context_array_t active{};
    active.emplace_back(&ctx);
    return active;
}
}  // namespace

// A replay-local stop changes the effective request without mutating the context's global
// start/stop state. The replay loop applies that request to the agent's hardware session.
TEST(pc_sampling, local_context_override_suspends_effective_request)
{
    context::context ctx{};
    ctx.context_idx = 72;
    ctx.pc_sampler  = std::make_unique<context::pc_sampling_service>();
    ctx.pc_sampler->enabled.store(true);

    {
        auto                                        active = as_active(ctx);
        kernel_replay::scoped_local_context_control loop{active};
        kernel_replay::set_toggles_armed(true);
        EXPECT_EQ(kernel_replay::replay_local_disable_context({.handle = ctx.context_idx}),
                  ROCPROFILER_STATUS_SUCCESS);
        kernel_replay::set_toggles_armed(false);

        ASSERT_TRUE(kernel_replay::local_context_override({.handle = ctx.context_idx}).has_value());
        EXPECT_FALSE(*kernel_replay::local_context_override({.handle = ctx.context_idx}));
        EXPECT_FALSE(pc_sampling::is_pc_sample_service_requested(&ctx));
        EXPECT_TRUE(ctx.pc_sampler->enabled.load())
            << "A replay-local stop must not change the context's global state";
    }

    EXPECT_TRUE(ctx.pc_sampler->enabled.load());
    EXPECT_TRUE(pc_sampling::is_pc_sample_service_requested(&ctx));
}

TEST(pc_sampling, local_context_override_restart_restores_effective_request)
{
    context::context ctx{};
    ctx.context_idx = 73;
    ctx.pc_sampler  = std::make_unique<context::pc_sampling_service>();
    ctx.pc_sampler->enabled.store(true);

    auto                                        active = as_active(ctx);
    kernel_replay::scoped_local_context_control loop{active};

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_disable_context({.handle = ctx.context_idx}),
              ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);
    EXPECT_FALSE(*kernel_replay::local_context_override({.handle = ctx.context_idx}));
    EXPECT_FALSE(pc_sampling::is_pc_sample_service_requested(&ctx));
    EXPECT_TRUE(ctx.pc_sampler->enabled.load())
        << "A replay-local stop must not change the context's global state";

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_enable_context({.handle = ctx.context_idx}),
              ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);
    EXPECT_TRUE(*kernel_replay::local_context_override({.handle = ctx.context_idx}));
    EXPECT_TRUE(pc_sampling::is_pc_sample_service_requested(&ctx));
    EXPECT_TRUE(ctx.pc_sampler->enabled.load())
        << "A replay-local start must not change the context's global state";

    ctx.pc_sampler->enabled.store(false);
    EXPECT_FALSE(pc_sampling::is_pc_sample_service_requested(&ctx))
        << "A replay-local start must not promote a globally stopped context";
}
