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
#include "lib/common/environment.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/spm/core.hpp"
#include "lib/rocprofiler-sdk/spm/dispatch_handlers.hpp"

#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <atomic>
#include <memory>

using namespace rocprofiler;

// Provided by core.cpp in this test binary.
AmdExtTable&
get_ext_table();
CoreApiTable&
get_api_table();

namespace
{
class FakeQueue : public hsa::Queue
{
public:
    FakeQueue(const hsa::AgentCache& a, rocprofiler_queue_id_t id)
    : hsa::Queue(a, get_api_table())
    , _agent(a)
    , _id(id)
    {}
    const hsa::AgentCache& get_agent() const final { return _agent; }
    rocprofiler_queue_id_t get_id() const final { return _id; }

    ~FakeQueue() override = default;

private:
    const hsa::AgentCache& _agent;
    rocprofiler_queue_id_t _id = {};
};

void
test_init()
{
    HsaApiTable table;
    table.amd_ext_ = &get_ext_table();
    table.core_    = &get_api_table();
    agent::construct_agent_cache(&table);
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
}

void
spm_dispatch_cb(const rocprofiler_spm_dispatch_counting_service_data_t*,
                rocprofiler_counter_config_id_t*,
                rocprofiler_user_data_t*,
                void* args)
{
    static_cast<std::atomic<int>*>(args)->fetch_add(1);
}

void
invoke_pre_kernel_call(context::context& ctx, std::atomic<int>* hits)
{
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_FALSE(agents.empty());
    const auto&            agent = agents.begin()->second;
    rocprofiler_queue_id_t qid{.handle = 1};
    FakeQueue              fq(agent, qid);

    auto info           = std::make_shared<spm::spm_counter_callback_info>();
    info->user_cb       = spm_dispatch_cb;
    info->callback_args = hits;

    context::correlation_id                           corr{};
    hsa::rocprofiler_packet                           pkt{};
    rocprofiler_user_data_t                           user_data{};
    hsa::queue_info_session_t::external_corr_id_map_t extern_ids{};

    auto ret = spm::pre_kernel_call(
        &ctx, info, fq, pkt, /*kernel_id=*/1, /*dispatch_id=*/1, &user_data, extern_ids, &corr);
    ASSERT_TRUE(ret.packet);
}

context::context_array_t
as_active(const context::context& ctx)
{
    context::context_array_t active{};
    active.emplace_back(&ctx);
    return active;
}
}  // namespace

TEST(spm_core, local_context_override_stops_pre_kernel_call)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    context::context ctx{};
    ctx.context_idx  = 52;
    ctx.dispatch_spm = std::make_unique<context::spm_dispatch_counter_collection_service>();
    ctx.dispatch_spm->enabled.wlock([](auto& data) { data = true; });

    std::atomic<int> hits{0};
    invoke_pre_kernel_call(ctx, &hits);
    EXPECT_EQ(hits.load(), 1);

    {
        auto                                        active = as_active(ctx);
        kernel_replay::scoped_local_context_control loop{active};
        kernel_replay::set_toggles_armed(true);
        EXPECT_EQ(kernel_replay::replay_local_disable_context({.handle = ctx.context_idx}),
                  ROCPROFILER_STATUS_SUCCESS);
        kernel_replay::set_toggles_armed(false);

        hits.store(0);
        invoke_pre_kernel_call(ctx, &hits);
        EXPECT_EQ(hits.load(), 0);
    }

    hits.store(0);
    invoke_pre_kernel_call(ctx, &hits);
    EXPECT_EQ(hits.load(), 1);

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
}

TEST(spm_core, local_context_override_restarts_pre_kernel_call)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    context::context ctx{};
    ctx.context_idx  = 53;
    ctx.dispatch_spm = std::make_unique<context::spm_dispatch_counter_collection_service>();
    ctx.dispatch_spm->enabled.wlock([](auto& data) { data = true; });

    std::atomic<int>                            hits{0};
    auto                                        active = as_active(ctx);
    kernel_replay::scoped_local_context_control loop{active};

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_disable_context({.handle = ctx.context_idx}),
              ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);
    invoke_pre_kernel_call(ctx, &hits);
    EXPECT_EQ(hits.load(), 0);

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_enable_context({.handle = ctx.context_idx}),
              ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);
    hits.store(0);
    invoke_pre_kernel_call(ctx, &hits);
    EXPECT_EQ(hits.load(), 1);

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
}

// A local start override must not promote a context whose global enabled flag is false.
TEST(spm_core, local_context_start_cannot_promote_globally_stopped)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    context::context ctx{};
    ctx.context_idx  = 54;
    ctx.dispatch_spm = std::make_unique<context::spm_dispatch_counter_collection_service>();
    ctx.dispatch_spm->enabled.wlock([](auto& data) { data = false; });

    std::atomic<int>                            hits{0};
    auto                                        active = as_active(ctx);
    kernel_replay::scoped_local_context_control loop{active};

    kernel_replay::set_toggles_armed(true);
    EXPECT_EQ(kernel_replay::replay_local_enable_context({.handle = ctx.context_idx}),
              ROCPROFILER_STATUS_SUCCESS);
    kernel_replay::set_toggles_armed(false);

    invoke_pre_kernel_call(ctx, &hits);
    EXPECT_EQ(hits.load(), 0) << "local start must not promote a globally stopped context";

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
}
