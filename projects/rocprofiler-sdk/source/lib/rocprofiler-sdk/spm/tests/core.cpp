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

#include "lib/rocprofiler-sdk/spm/core.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/spm/decode.hpp"
#include "lib/rocprofiler-sdk/spm/dispatch_handlers.hpp"

#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>
#include <sstream>

using namespace rocprofiler::counters;
using namespace rocprofiler;

AmdExtTable&
get_ext_table()
{
    static auto _v = []() {
        auto val                                  = AmdExtTable{};
        val.version.major_id                      = HSA_AMD_EXT_API_TABLE_MAJOR_VERSION;
        val.version.minor_id                      = sizeof(AmdExtTable);
        val.version.step_id                       = HSA_AMD_EXT_API_TABLE_STEP_VERSION;
        val.hsa_amd_memory_pool_get_info_fn       = hsa_amd_memory_pool_get_info;
        val.hsa_amd_agent_iterate_memory_pools_fn = hsa_amd_agent_iterate_memory_pools;
        val.hsa_amd_memory_pool_allocate_fn       = hsa_amd_memory_pool_allocate;
        val.hsa_amd_memory_pool_free_fn           = hsa_amd_memory_pool_free;
        val.hsa_amd_agent_memory_pool_get_info_fn = hsa_amd_agent_memory_pool_get_info;
        val.hsa_amd_agents_allow_access_fn        = hsa_amd_agents_allow_access;
        val.hsa_amd_memory_fill_fn                = hsa_amd_memory_fill;
        val.hsa_amd_signal_create_fn              = hsa_amd_signal_create;
        val.hsa_amd_spm_acquire_fn                = hsa_amd_spm_acquire;
        val.hsa_amd_spm_release_fn                = hsa_amd_spm_release;
        val.hsa_amd_signal_async_handler_fn       = hsa_amd_signal_async_handler;
        return val;
    }();
    return _v;
}

CoreApiTable&
get_api_table()
{
    static auto _v = []() {
        auto val                          = CoreApiTable{};
        val.version.major_id              = HSA_CORE_API_TABLE_MAJOR_VERSION;
        val.version.minor_id              = sizeof(CoreApiTable);
        val.version.step_id               = HSA_CORE_API_TABLE_STEP_VERSION;
        val.hsa_iterate_agents_fn         = hsa_iterate_agents;
        val.hsa_agent_get_info_fn         = hsa_agent_get_info;
        val.hsa_queue_create_fn           = hsa_queue_create;
        val.hsa_queue_destroy_fn          = hsa_queue_destroy;
        val.hsa_signal_wait_relaxed_fn    = hsa_signal_wait_relaxed;
        val.hsa_memory_copy_fn            = hsa_memory_copy;
        val.hsa_signal_create_fn          = hsa_signal_create;
        val.hsa_signal_destroy_fn         = hsa_signal_destroy;
        val.hsa_signal_store_relaxed_fn   = hsa_signal_store_relaxed;
        val.hsa_signal_store_screlease_fn = hsa_signal_store_screlease;
        return val;
    }();
    return _v;
}

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            ASSERT_EQ(CHECKSTATUS, ROCPROFILER_STATUS_SUCCESS) << errmsg.str();                    \
        }                                                                                          \
    }

namespace
{
bool
is_spm_supported_arch(const hsa::AgentCache& agent)
{
    auto rocp_agent = agent.get_rocp_agent();
    if(!rocp_agent) return false;
    auto v = rocp_agent->gfx_target_version;
    return (v >= 90400 && v <= 90402) || v == 90500;
}

auto
findSPMDeviceMetrics(const hsa::AgentCache& agent, const std::unordered_set<std::string>& metrics)
{
    std::vector<counters::Metric> ret;
    auto                          mets         = counters::loadMetrics();
    const auto&                   all_counters = mets->arch_to_metric;

    ROCP_INFO << "Looking up counters for " << std::string(agent.name());
    const auto* gfx_metrics = common::get_val(all_counters, std::string(agent.name()));
    if(!gfx_metrics)
    {
        ROCP_ERROR << "No counters found for " << std::string(agent.name());
        return ret;
    }

    for(const auto& counter : *gfx_metrics)
    {
        auto rocp_agent = CHECK_NOTNULL(agent.get_rocp_agent());

        if((metrics.count(counter.name()) > 0 || metrics.empty()) &&
           rocprofiler::counters::has_spm_support(counter, rocp_agent->id))
        {
            ret.push_back(counter);
        }
    }
    return ret;
}

std::vector<counters::Metric>
getSPMMetrics(const hsa::AgentCache& agent)
{
    auto metrics = findSPMDeviceMetrics(agent, {});
    EXPECT_FALSE(metrics.empty()) << "SPM metrics should not be empty for " << agent.name();
    return metrics;
}

void
test_init()
{
    HsaApiTable table;
    table.amd_ext_ = &get_ext_table();
    table.core_    = &get_api_table();
    rocprofiler::hsa::copy_table(table.core_, 0);
    rocprofiler::hsa::copy_table(table.amd_ext_, 0);
    agent::construct_agent_cache(&table);
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);

    hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
}

}  // namespace

namespace
{
rocprofiler_context_id_t&
get_client_ctx()
{
    static rocprofiler_context_id_t ctx{0};
    return ctx;
}

void
set_client_ctx(rocprofiler_context_id_t& ctx)
{
    ctx = rocprofiler_context_id_t{0};
}

void
null_dispatch_callback(const rocprofiler_spm_dispatch_counting_service_data_t*,
                       rocprofiler_counter_config_id_t*,
                       rocprofiler_user_data_t*,
                       void*)
{}

void
null_record_callback(const rocprofiler_spm_dispatch_counting_service_data_t*,
                     const rocprofiler_spm_counter_record_t**,
                     size_t,
                     rocprofiler_spm_record_flag_t,
                     rocprofiler_user_data_t,
                     void*)
{}

void
null_buffered_callback(rocprofiler_context_id_t,
                       rocprofiler_buffer_id_t,
                       rocprofiler_record_header_t**,
                       size_t,
                       void*,
                       uint64_t)
{}
}  // namespace

TEST(spm_core, check_packet_generation)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    bool any_spm_agent = false;
    for(const auto& [_, agent] : agents)
    {
        auto rocp_agent = CHECK_NOTNULL(agent.get_rocp_agent());
        if(rocp_agent->runtime_visibility.hsa && rocp_agent->runtime_visibility.hip)
        {
            if(!is_spm_supported_arch(agent)) continue;
            any_spm_agent = true;
            auto metrics  = getSPMMetrics(agent);
            ASSERT_TRUE(agent.get_rocp_agent());
            for(auto& metric : metrics)
            {
                /**
                 * Check profile construction
                 */
                rocprofiler_counter_config_id_t cfg_id = {.handle = 0};
                rocprofiler_counter_id_t        id     = {.handle = metric.id()};
                ROCP_ERROR << fmt::format("Generating packet for {}", metric);

                std::vector<rocprofiler_spm_parameters_t*> input_params{};
                rocprofiler_spm_parameters_t               param{
                    .size = sizeof(rocprofiler_spm_parameters_t),
                    .type = ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
                    .value = 1200};
                input_params.push_back(&param);

                ROCPROFILER_CALL(rocprofiler_spm_create_counter_config(agent.get_rocp_agent()->id,
                                                                       &id,
                                                                       1,
                                                                       input_params.data(),
                                                                       input_params.size(),
                                                                       &cfg_id),
                                 "Unable to create profile");
                auto profile = spm::get_spm_counter_config(cfg_id);
                ASSERT_TRUE(profile);

                /**
                 * Check that a packet generator was created
                 */

                /**
                 * Check packet generation
                 */
                std::unique_ptr<hsa::AQLPacket> pkt = nullptr;
                EXPECT_EQ(get_spm_packet(pkt, profile), ROCPROFILER_STATUS_SUCCESS)
                    << "Unable to generate packet";
                EXPECT_TRUE(pkt) << "Expected a packet to be generated";
                ROCPROFILER_CALL(rocprofiler_spm_destroy_counter_config(cfg_id),
                                 "Could not delete profile id");
            }
        }
    }
    if(!any_spm_agent) ROCP_ERROR << "SPM unavailable";
}

namespace rocprofiler
{
namespace hsa
{
class FakeQueue : public Queue
{
public:
    FakeQueue(const AgentCache& a, rocprofiler_queue_id_t id)
    : Queue(a, get_api_table())
    , _agent(a)
    , _id(id)
    {}
    const AgentCache&      get_agent() const final { return _agent; };
    rocprofiler_queue_id_t get_id() const final { return _id; };

    ~FakeQueue() override = default;

private:
    const AgentCache&      _agent;
    rocprofiler_queue_id_t _id = {};
};

}  // namespace hsa
}  // namespace rocprofiler

namespace
{
struct expected_dispatch
{
    // To pass back
    rocprofiler_counter_config_id_t    id             = {.handle = 0};
    rocprofiler_queue_id_t             queue_id       = {.handle = 0};
    rocprofiler_agent_id_t             agent_id       = {.handle = 0};
    uint64_t                           kernel_id      = 0;
    uint64_t                           dispatch_id    = 0;
    rocprofiler_async_correlation_id_t correlation_id = {.internal = 0, .external = {.value = 0}};
    rocprofiler_dim3_t                 workgroup_size = {0, 0, 0};
    rocprofiler_dim3_t                 grid_size      = {0, 0, 0};
    rocprofiler_counter_config_id_t*   config         = nullptr;
};

void
user_dispatch_cb(const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
                 rocprofiler_counter_config_id_t*                        config,
                 rocprofiler_user_data_t*                                user_data,
                 void*                                                   callback_data_args)
{
    expected_dispatch& expected = *static_cast<expected_dispatch*>(callback_data_args);

    auto agent_id       = dispatch_data->dispatch_info.agent_id;
    auto queue_id       = dispatch_data->dispatch_info.queue_id;
    auto correlation_id = dispatch_data->correlation_id;
    auto kernel_id      = dispatch_data->dispatch_info.kernel_id;
    auto dispatch_id    = dispatch_data->dispatch_info.dispatch_id;

    EXPECT_EQ(sizeof(rocprofiler_spm_dispatch_counting_service_data_t), dispatch_data->size);
    EXPECT_EQ(expected.kernel_id, kernel_id);
    EXPECT_EQ(expected.dispatch_id, dispatch_id);
    EXPECT_EQ(expected.agent_id, agent_id);
    EXPECT_EQ(expected.queue_id.handle, queue_id.handle);
    EXPECT_EQ(expected.correlation_id.internal, correlation_id.internal);
    EXPECT_EQ(expected.correlation_id.external.ptr, correlation_id.external.ptr);
    EXPECT_EQ(expected.correlation_id.external.value, correlation_id.external.value);
    EXPECT_EQ(expected.workgroup_size, dispatch_data->dispatch_info.workgroup_size);
    EXPECT_EQ(expected.grid_size, dispatch_data->dispatch_info.grid_size);

    ASSERT_NE(config, nullptr);
    config->handle = expected.id.handle;

    (void) user_data;
}

}  // namespace

namespace rocprofiler
{
namespace buffer
{
uint64_t
get_buffer_offset();
}
}  // namespace rocprofiler

TEST(spm_core, check_callbacks)
{
    int64_t count = 0;
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    context::context ctx;
    ctx.dispatch_spm =
        std::make_unique<rocprofiler::context::spm_dispatch_counter_collection_service>();
    ctx.dispatch_spm->enabled.wlock([](auto& data) { data = true; });

    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    hsa::get_queue_controller()->disable_serialization();

    bool any_spm_agent = false;
    for(const auto& [_, agent] : agents)
    {
        /**
         * Setup
         */
        auto rocp_agent = CHECK_NOTNULL(agent.get_rocp_agent());
        if(rocp_agent->runtime_visibility.hsa && rocp_agent->runtime_visibility.hip)
        {
            if(!is_spm_supported_arch(agent)) continue;
            any_spm_agent              = true;
            rocprofiler_queue_id_t qid = {.handle = static_cast<uint64_t>(count++)};
            hsa::FakeQueue         fq(agent, qid);
            auto                   metrics = getSPMMetrics(agent);
            ASSERT_TRUE(agent.get_rocp_agent());
            for(auto& metric : metrics)
            {
                if(!metric.expression().empty()) continue;

                /**
                 * Setup
                 */
                expected_dispatch                          expected = {};
                rocprofiler_counter_id_t                   id       = {.handle = metric.id()};
                std::vector<rocprofiler_spm_parameters_t*> input_params{};
                rocprofiler_spm_parameters_t               param{
                    .size = sizeof(rocprofiler_spm_parameters_t),
                    .type = ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
                    .value = 1200};
                input_params.push_back(&param);

                ROCPROFILER_CALL(rocprofiler_spm_create_counter_config(agent.get_rocp_agent()->id,
                                                                       &id,
                                                                       1,
                                                                       input_params.data(),
                                                                       input_params.size(),
                                                                       &expected.id),
                                 "Unable to create profile");
                auto profile = spm::get_spm_counter_config(expected.id);
                ASSERT_TRUE(profile);

                std::shared_ptr<spm::spm_counter_callback_info> cb_info =
                    std::make_shared<spm::spm_counter_callback_info>();
                cb_info->user_cb              = user_dispatch_cb;
                cb_info->callback_args        = static_cast<void*>(&expected);
                cb_info->record_callback      = null_record_callback;
                cb_info->record_callback_args = nullptr;
                context::correlation_id corr_id;
                corr_id.internal = count++;

                hsa::rocprofiler_packet pkt;
                pkt.ext_amd_aql_pm4.header = count++;

                expected.correlation_id = {.internal = corr_id.internal,
                                           .external = context::null_user_data};
                expected.workgroup_size = {pkt.kernel_dispatch.workgroup_size_x,
                                           pkt.kernel_dispatch.workgroup_size_y,
                                           pkt.kernel_dispatch.workgroup_size_z};
                expected.grid_size      = {pkt.kernel_dispatch.grid_size_x,
                                      pkt.kernel_dispatch.grid_size_y,
                                      pkt.kernel_dispatch.grid_size_z};
                expected.kernel_id      = count++;
                expected.dispatch_id    = count++;
                expected.queue_id       = qid;
                expected.agent_id       = fq.get_agent().get_rocp_agent()->id;

                hsa::queue_info_session_t::external_corr_id_map_t extern_ids = {};

                auto user_data       = rocprofiler_user_data_t{.value = corr_id.internal};
                auto ret_pkt         = spm::pre_kernel_call(&ctx,
                                                    cb_info,
                                                    fq,
                                                    pkt,
                                                    expected.kernel_id,
                                                    expected.dispatch_id,
                                                    &user_data,
                                                    extern_ids,
                                                    &corr_id);
                auto _sess           = hsa::queue_info_session_t{.queue = fq};
                _sess.correlation_id = &corr_id;

                auto sess = std::make_shared<hsa::queue_info_session_t>(std::move(_sess));
                ASSERT_TRUE(ret_pkt.packet)
                    << fmt::format("Expected a packet to be generated for - {}", metric.name());
                cb_info->packet_return_map.rlock([&](const auto& map) {
                    EXPECT_EQ(map.size(), 1) << "Expected packet_return_map to have one entry";
                    EXPECT_TRUE(common::get_val(map, ret_pkt.packet.get()))
                        << "Expected packet in packet_return_map";
                });
                auto  data    = std::vector<int>(10, 1);
                auto* spm_pkt = dynamic_cast<hsa::SPMPacket*>(ret_pkt.packet.get());
                rocprofiler::spm::aql_data_callback(0, &(data[0]), data.size(), 0, spm_pkt);
                spm::inst_pkt_t pkts;
                pkts.emplace_back(
                    std::make_pair(std::move(ret_pkt.packet), static_cast<spm::ClientID>(0)));
                post_kernel_call(&ctx, cb_info, sess, pkts, kernel_dispatch::profiling_time{});
                ROCPROFILER_CALL(rocprofiler_spm_destroy_counter_config(expected.id),
                                 "Could not delete profile id");
            }
        }
    }
    registration::set_init_status(1);

    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
    if(!any_spm_agent) ROCP_ERROR << "SPM unavailable";
}

TEST(spm_core, destroy_counter_profile)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    bool any_spm_agent = false;
    for(const auto& [_, agent] : agents)
    {
        auto rocp_agent = CHECK_NOTNULL(agent.get_rocp_agent());
        if(rocp_agent->runtime_visibility.hsa && rocp_agent->runtime_visibility.hip)
        {
            if(!is_spm_supported_arch(agent)) continue;
            any_spm_agent = true;
            auto metrics  = getSPMMetrics(agent);
            ASSERT_TRUE(agent.get_rocp_agent());
            for(auto& metric : metrics)
            {
                expected_dispatch        expected = {};
                rocprofiler_counter_id_t id       = {.handle = metric.id()};

                std::vector<rocprofiler_spm_parameters_t*> input_params{};
                rocprofiler_spm_parameters_t               param{
                    .size = sizeof(rocprofiler_spm_parameters_t),
                    .type = ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
                    .value = 1200};
                input_params.push_back(&param);
                ROCPROFILER_CALL(rocprofiler_spm_create_counter_config(agent.get_rocp_agent()->id,
                                                                       &id,
                                                                       1,
                                                                       input_params.data(),
                                                                       input_params.size(),
                                                                       &expected.id),
                                 "Unable to create profile");
                ROCPROFILER_CALL(rocprofiler_spm_destroy_counter_config(expected.id),
                                 "Could not delete profile id");
                /**
                 * Check the profile was actually destroyed
                 */
                auto profile = spm::get_spm_counter_config(expected.id);
                EXPECT_FALSE(profile);
            }
        }
    }

    registration::set_init_status(1);

    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
    if(!any_spm_agent) ROCP_ERROR << "SPM unavailable";
}

TEST(spm_core, start_stop_callback_ctx)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    registration::set_fini_status(0);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_spm_configure_callback_dispatch_service(get_client_ctx(),
                                                                         null_dispatch_callback,
                                                                         (void*) 0x12345,
                                                                         null_record_callback,
                                                                         (void*) 0x54321),
                     "Could not setup counting service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    /**
     * Check that the context was actually started
     */
    auto* ctx_p = context::get_mutable_registered_context(get_client_ctx());
    ASSERT_TRUE(ctx_p);
    auto& ctx = *ctx_p;

    ASSERT_TRUE(ctx.dispatch_spm);
    ASSERT_EQ(ctx.dispatch_spm->callbacks.size(), 1);
    EXPECT_EQ(ctx.dispatch_spm->callbacks.at(0)->user_cb, null_dispatch_callback);
    EXPECT_EQ(ctx.dispatch_spm->callbacks.at(0)->callback_args, (void*) 0x12345);
    EXPECT_EQ(ctx.dispatch_spm->callbacks.at(0)->record_callback, null_record_callback);
    EXPECT_EQ(ctx.dispatch_spm->callbacks.at(0)->record_callback_args, (void*) 0x54321);
    EXPECT_EQ(ctx.dispatch_spm->callbacks.at(0)->context.handle, get_client_ctx().handle);

    bool found = false;
    ctx.dispatch_spm->enabled.rlock([&](const auto& data) { found = data; });
    EXPECT_TRUE(found);

    found = false;
    hsa::get_queue_controller()->iterate_callbacks([&](auto cid, const auto&) {
        if(cid == ctx.dispatch_spm->callbacks.at(0)->queue_id)
        {
            found = true;
        }
    });
    EXPECT_TRUE(found);

    /**
     * Check if context can be disabled correctly
     */
    ROCPROFILER_CALL(rocprofiler_stop_context(get_client_ctx()), "stop context");

    found = false;
    ctx.dispatch_spm->enabled.rlock([&](const auto& data) { found = data; });
    EXPECT_FALSE(found);

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}

TEST(spm_core, start_stop_buffered_ctx)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    rocprofiler_buffer_id_t opt_buff_id = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               500 * sizeof(size_t),
                                               500 * sizeof(size_t),
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               null_buffered_callback,
                                               nullptr,
                                               &opt_buff_id),
                     "Could not create buffer");

    ROCPROFILER_CALL(rocprofiler_spm_configure_buffer_dispatch_service(
                         get_client_ctx(), opt_buff_id, null_dispatch_callback, (void*) 0x12345),
                     "Could not setup buffered service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    /**
     * Check that the context was actually started
     */
    auto* ctx_p = context::get_mutable_registered_context(get_client_ctx());
    ASSERT_TRUE(ctx_p);
    auto& ctx = *ctx_p;

    ASSERT_TRUE(ctx.dispatch_spm);
    ASSERT_EQ(ctx.dispatch_spm->callbacks.size(), 1);
    EXPECT_EQ(ctx.dispatch_spm->callbacks.at(0)->user_cb, null_dispatch_callback);
    EXPECT_EQ(ctx.dispatch_spm->callbacks.at(0)->callback_args, (void*) 0x12345);
    EXPECT_EQ(ctx.dispatch_spm->callbacks.at(0)->context.handle, get_client_ctx().handle);
    ASSERT_TRUE(ctx.dispatch_spm->callbacks.at(0)->buffer);
    EXPECT_EQ(ctx.dispatch_spm->callbacks.at(0)->buffer->handle, opt_buff_id.handle);

    bool found = false;
    ctx.dispatch_spm->enabled.rlock([&](const auto& data) { found = data; });
    EXPECT_TRUE(found);

    found = false;
    hsa::get_queue_controller()->iterate_callbacks([&](auto cid, const auto&) {
        if(cid == ctx.dispatch_spm->callbacks.at(0)->queue_id)
        {
            found = true;
        }
    });
    EXPECT_TRUE(found);

    /**
     * Check if context can be disabled correctly
     */
    ROCPROFILER_CALL(rocprofiler_stop_context(get_client_ctx()), "stop context");

    found = false;
    ctx.dispatch_spm->enabled.rlock([&](const auto& data) { found = data; });
    EXPECT_FALSE(found);

    rocprofiler_flush_buffer(opt_buff_id);
    rocprofiler_destroy_buffer(opt_buff_id);

    registration::set_init_status(1);

    registration::finalize();
}

TEST(spm_core, test_profile_incremental)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    bool any_spm_agent = false;
    for(const auto& [_, agent] : agents)
    {
        auto rocp_agent = CHECK_NOTNULL(agent.get_rocp_agent());
        if(rocp_agent->runtime_visibility.hsa && rocp_agent->runtime_visibility.hip)
        {
            if(!is_spm_supported_arch(agent)) continue;
            any_spm_agent = true;
            auto metrics  = getSPMMetrics(agent);
            ASSERT_TRUE(agent.get_rocp_agent());

            std::map<std::string, std::vector<counters::Metric>> metric_blocks;
            for(const auto& metric : metrics)
            {
                if(!metric.block().empty())
                {
                    metric_blocks[metric.block()].push_back(metric);
                }
            }

            rocprofiler_counter_config_id_t cfg_id = {};

            // Add one counter from each block to incrementally to make sure we can
            // add them incrementally
            for(const auto& [block_name, block_metrics] : metric_blocks)
            {
                rocprofiler_counter_config_id_t old_id = cfg_id;
                rocprofiler_counter_id_t        id     = {.handle = block_metrics.front().id()};

                std::vector<rocprofiler_spm_parameters_t*> input_params{};
                rocprofiler_spm_parameters_t               param{
                    .size = sizeof(rocprofiler_spm_parameters_t),
                    .type = ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
                    .value = 1200};
                input_params.push_back(&param);
                ROCPROFILER_CALL(
                    rocprofiler_spm_create_counter_config(agent.get_rocp_agent()->id,
                                                          &id,
                                                          1,
                                                          input_params.data(),
                                                          input_params.size(),
                                                          &cfg_id),
                    "Unable to create profile incrementally when we should be able to");
                EXPECT_NE(old_id.handle, cfg_id.handle) << "We expect that the handle changes this "
                                                           "is due to the existing profile being "
                                                           "unmodifiable after creation: "
                                                        << block_name;
            }

            // Check that we encounter an error of exceeds hardware limits eventually
            auto status = ROCPROFILER_STATUS_SUCCESS;
            for(const auto& metric : metrics)
            {
                /**
                 * Check profile construction
                 */
                std::vector<rocprofiler_spm_parameters_t*> input_params{};
                rocprofiler_spm_parameters_t               param{
                    .size = sizeof(rocprofiler_spm_parameters_t),
                    .type = ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
                    .value = 1200};
                input_params.push_back(&param);
                rocprofiler_counter_id_t id = {.handle = metric.id()};
                if(status = rocprofiler_spm_create_counter_config(agent.get_rocp_agent()->id,
                                                                  &id,
                                                                  1,
                                                                  input_params.data(),
                                                                  input_params.size(),
                                                                  &cfg_id);
                   status != ROCPROFILER_STATUS_SUCCESS)
                {
                    break;
                }
            }
            EXPECT_EQ(status, ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT);
        }
    }

    set_client_ctx(get_client_ctx());
    if(!any_spm_agent) ROCP_ERROR << "SPM unavailable";
}

TEST(spm_core, public_api_iterate_agents)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    auto agents        = hsa::get_queue_controller()->get_supported_agents();
    bool any_spm_agent = false;
    for(const auto& [_, agent] : agents)
    {
        auto rocp_agent = CHECK_NOTNULL(agent.get_rocp_agent());
        if(rocp_agent->runtime_visibility.hsa && rocp_agent->runtime_visibility.hip)
        {
            if(!is_spm_supported_arch(agent)) continue;
            any_spm_agent = true;
            auto expected = getSPMMetrics(agent);

            std::set<uint64_t> from_api{};

            // Iterate through the agents and get the counters available on that agent
            ROCPROFILER_CALL(rocprofiler_spm_iterate_agent_supported_counters(
                                 agent.get_rocp_agent()->id,
                                 [](rocprofiler_agent_id_t,
                                    rocprofiler_counter_id_t* counters,
                                    size_t                    num_counters,
                                    void*                     user_data) {
                                     std::set<uint64_t>* vec =
                                         static_cast<std::set<uint64_t>*>(user_data);
                                     for(size_t i = 0; i < num_counters; i++)
                                     {
                                         vec->insert(counters[i].handle);
                                     }
                                     return ROCPROFILER_STATUS_SUCCESS;
                                 },
                                 static_cast<void*>(&from_api)),
                             "Could not fetch supported counters");
            for(const auto& x : expected)
            {
                bool found = false;
                for(auto it = from_api.begin(); it != from_api.end(); ++it)
                {
                    rocprofiler_counter_id_t counter_id = {.handle = *it};
                    if(counters::get_base_metric_from_counter_id(counter_id) == x.id())
                    {
                        from_api.erase(it);
                        found = true;
                        break;
                    }
                }
                ASSERT_TRUE(found)
                    << "Expected counter ID " << x.id() << " not found in API results";
            }

            EXPECT_TRUE(from_api.empty());
        }
    }

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    if(!any_spm_agent) ROCP_ERROR << "SPM unavailable";
}

TEST(spm_core, query_agent_configurations)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);

    bool any_spm_agent = false;
    for(const auto& [_, agent] : agents)
    {
        auto rocp_agent = CHECK_NOTNULL(agent.get_rocp_agent());
        if(rocp_agent->runtime_visibility.hsa && rocp_agent->runtime_visibility.hip)
        {
            if(!is_spm_supported_arch(agent)) continue;
            any_spm_agent = true;

            struct query_result
            {
                size_t                                                 num_configs = 0;
                std::vector<rocprofiler_spm_available_configuration_t> configs;
            };

            query_result result{};

            auto status = rocprofiler_spm_query_agent_configurations(
                rocp_agent->id,
                [](const rocprofiler_spm_available_configuration_t** config,
                   size_t                                            num_config,
                   void* user_data) -> rocprofiler_status_t {
                    auto* res        = static_cast<query_result*>(user_data);
                    res->num_configs = num_config;
                    for(size_t i = 0; i < num_config; i++)
                        res->configs.push_back(*config[i]);
                    return ROCPROFILER_STATUS_SUCCESS;
                },
                &result);

            if(status != ROCPROFILER_STATUS_SUCCESS)
            {
                continue;
            }
            ASSERT_GT(result.num_configs, 0) << "Expected at least one configuration";

            bool found_interval = false;
            for(const auto& cfg : result.configs)
            {
                EXPECT_EQ(cfg.size, sizeof(rocprofiler_spm_available_configuration_t));
                if(cfg.type == ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES)
                {
                    found_interval = true;
                    EXPECT_EQ(cfg.interval.min_interval, 32);
                    EXPECT_GT(cfg.interval.max_interval, cfg.interval.min_interval);
                }
                EXPECT_NE(cfg.type, ROCPROFILER_SPM_PARAMETER_TYPE_NONE);
                EXPECT_NE(cfg.type, ROCPROFILER_SPM_PARAMETER_TYPE_LAST);
            }
            EXPECT_TRUE(found_interval) << "Expected a sample interval configuration";
        }
    }

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
    if(!any_spm_agent) ROCP_ERROR << "SPM unavailable";
}

TEST(spm_core, stop_context_removes_callbacks)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    registration::set_fini_status(0);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_spm_configure_callback_dispatch_service(get_client_ctx(),
                                                                         null_dispatch_callback,
                                                                         (void*) 0x12345,
                                                                         null_record_callback,
                                                                         (void*) 0x54321),
                     "Could not setup counting service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    auto* ctx_p = context::get_mutable_registered_context(get_client_ctx());
    ASSERT_TRUE(ctx_p);
    auto& ctx = *ctx_p;
    ASSERT_TRUE(ctx.dispatch_spm);
    ASSERT_EQ(ctx.dispatch_spm->callbacks.size(), 1);

    auto pre_stop_queue_id = ctx.dispatch_spm->callbacks.at(0)->queue_id;
    bool found             = false;
    hsa::get_queue_controller()->iterate_callbacks([&](auto cid, const auto&) {
        if(cid == pre_stop_queue_id) found = true;
    });
    EXPECT_TRUE(found) << "Callback should be registered after start";

    // Stop exercises queue_controller_sync + remove_callback
    ROCPROFILER_CALL(rocprofiler_stop_context(get_client_ctx()), "stop context");

    bool enabled = true;
    ctx.dispatch_spm->enabled.rlock([&](const auto& data) { enabled = data; });
    EXPECT_FALSE(enabled);

    EXPECT_EQ(ctx.dispatch_spm->callbacks.at(0)->queue_id, hsa::ClientID{-1})
        << "queue_id should be reset to -1 after stop";

    found = false;
    hsa::get_queue_controller()->iterate_callbacks([&](auto cid, const auto&) {
        if(cid == pre_stop_queue_id) found = true;
    });
    EXPECT_FALSE(found) << "Callback should be removed from queue controller after stop";

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}

TEST(spm_core, stop_context_sync_and_restart)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    registration::set_fini_status(0);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_spm_configure_callback_dispatch_service(get_client_ctx(),
                                                                         null_dispatch_callback,
                                                                         (void*) 0x12345,
                                                                         null_record_callback,
                                                                         (void*) 0x54321),
                     "Could not setup counting service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    auto* ctx_p = context::get_mutable_registered_context(get_client_ctx());
    ASSERT_TRUE(ctx_p);
    auto& ctx = *ctx_p;
    ASSERT_TRUE(ctx.dispatch_spm);

    auto pre_stop_queue_id = ctx.dispatch_spm->callbacks.at(0)->queue_id;

    ROCPROFILER_CALL(rocprofiler_stop_context(get_client_ctx()), "stop context");

    // Restart to verify queue controller is in a clean state after sync + teardown
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "restart context");

    bool enabled = false;
    ctx.dispatch_spm->enabled.rlock([&](const auto& data) { enabled = data; });
    EXPECT_TRUE(enabled) << "Context should be enabled after restart";

    auto post_restart_queue_id = ctx.dispatch_spm->callbacks.at(0)->queue_id;
    EXPECT_NE(post_restart_queue_id, hsa::ClientID{-1})
        << "Restarted context should have a valid queue_id";
    EXPECT_NE(post_restart_queue_id, pre_stop_queue_id)
        << "Restarted context should get a fresh callback ID";

    bool found = false;
    hsa::get_queue_controller()->iterate_callbacks([&](auto cid, const auto&) {
        if(cid == post_restart_queue_id) found = true;
    });
    EXPECT_TRUE(found) << "New callback should be registered after restart";

    ROCPROFILER_CALL(rocprofiler_stop_context(get_client_ctx()), "stop context final");

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}
