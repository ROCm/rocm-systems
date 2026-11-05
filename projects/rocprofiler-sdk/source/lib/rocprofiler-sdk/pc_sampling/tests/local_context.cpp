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

// PC sampling is agent-wide and does not consult local_context_override(). A recorded
// local stop must succeed (the TLS map is service-agnostic) but must not flip the
// sampler's enabled flag. Reprogramming PCS hardware per pass is exactly what the
// sticky override is meant to avoid; until a consumer is wired, collection is a no-op.
TEST(pc_sampling, local_context_override_does_not_toggle_enabled)
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
        EXPECT_TRUE(ctx.pc_sampler->enabled.load())
            << "PC sampling is agent-wide; local stop must not flip the service enabled flag";
    }

    EXPECT_TRUE(ctx.pc_sampler->enabled.load());
}

TEST(pc_sampling, local_context_override_restart_does_not_toggle_enabled)
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
    EXPECT_TRUE(ctx.pc_sampler->enabled.load())
        << "PC sampling ignores a local stop; the sampler stays globally active";

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_enable_context({.handle = ctx.context_idx}),
              ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);
    EXPECT_TRUE(*kernel_replay::local_context_override({.handle = ctx.context_idx}));
    EXPECT_TRUE(ctx.pc_sampler->enabled.load())
        << "PC sampling ignores a local start; the sampler stays globally active";
}
