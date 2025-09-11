// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/dispatch_handlers.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/counters.h>
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
#include <tuple>

using namespace rocprofiler::counters::test_constants;
using namespace rocprofiler::counters;
using namespace rocprofiler;

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
auto
findDeviceMetrics(const hsa::AgentCache& agent, const std::unordered_set<std::string>& metrics)
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
        if(metrics.count(counter.name()) > 0 || metrics.empty())
        {
            ret.push_back(counter);
        }
    }
    return ret;
}

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
buffered_callback(rocprofiler_context_id_t,
                  rocprofiler_buffer_id_t,
                  rocprofiler_record_header_t** headers,
                  size_t                        num_headers,
                  void* /* user_data */,
                  uint64_t)
{
    std::set<double>   seen_data;
    std::set<uint64_t> seen_dims;
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
           header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            // Print the returned counter data.
            auto* record = static_cast<rocprofiler_record_counter_t*>(header->payload);
            seen_dims.insert(record->id);
            seen_data.insert(record->counter_value);
        }
    }

    // /**
    //  * Specific counters default to a size of 128 even if they have less data (primarily
    //  * TCP). This is a known quirk on AQL profile's end where it will allocate for 128 entries
    //  * but return less (and the data may be duplicated across entries). Skip these entires for
    //  * testing purposes since we cannot determine what mock data will be in the return (and its
    //  * arch dependent).
    //  */
    // if(expected.expected_size == 128) return;

    // EXPECT_EQ(seen_dims.size(), expected.expected_size);
    // EXPECT_EQ(seen_data.size(), expected.expected_size);

    // ASSERT_FALSE(seen_data.empty());
    // if(expected.is_special)
    // {
    //     EXPECT_EQ(*seen_data.begin(), expected.special_val);
    // }
    // else
    // {
    //     EXPECT_EQ(*seen_data.begin(), 1.0);
    //     EXPECT_EQ(*seen_data.rbegin(), double(seen_data.size()));
    // }
}

void
null_dispatch_callback(rocprofiler_dispatch_counting_service_data_t,
                       rocprofiler_counter_config_id_t*,
                       rocprofiler_user_data_t*,
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

void
null_record_callback(rocprofiler_dispatch_counting_service_data_t,
                     rocprofiler_record_counter_t*,
                     size_t,
                     rocprofiler_user_data_t,
                     void*)
{}

}  // namespace

TEST(core, check_packet_generation)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    for(const auto& [_, agent] : agents)
    {
        auto metrics = findDeviceMetrics(agent, {});
        ASSERT_FALSE(metrics.empty());
        ASSERT_TRUE(agent.get_rocp_agent());
        for(auto& metric : metrics)
        {
            /**
             * Check profile construction
             */
            rocprofiler_counter_config_id_t cfg_id = {.handle = 0};
            rocprofiler_counter_id_t        id     = {.handle = metric.id()};
            ROCP_ERROR << fmt::format("Generating packet for {}", metric);
            ROCPROFILER_CALL(
                rocprofiler_create_counter_config(agent.get_rocp_agent()->id, &id, 1, &cfg_id),
                "Unable to create profile");
            auto profile = counters::get_counter_config(cfg_id);
            ASSERT_TRUE(profile);
            EXPECT_EQ(counters::counter_callback_info::setup_counter_config(profile),
                      ROCPROFILER_STATUS_SUCCESS)
                << fmt::format("Could not build profile for {}", metric.name());

            /**
             * Check that a packet generator was created and there is an AST with constructed
             * dimensions
             */
            EXPECT_TRUE(profile->pkt_generator) << "No packet generator created";
            EXPECT_EQ(profile->asts.size(), 1);
            EXPECT_FALSE(profile->asts.at(0).dimension_types().empty());

            /**
             * Check packet generation
             */
            counters::counter_callback_info cb_info;
            std::unique_ptr<hsa::AQLPacket> pkt;
            EXPECT_EQ(cb_info.get_packet(pkt, profile), ROCPROFILER_STATUS_SUCCESS)
                << "Unable to generate packet";
            EXPECT_TRUE(pkt) << "Expected a packet to be generated";
            cb_info.packet_return_map.wlock([&](const auto& data) {
                EXPECT_EQ(data.size(), 1) << "Incorrect packet size";
                const auto* ptr = common::get_val(data, pkt.get());
                EXPECT_TRUE(ptr) << "Could not find pkt";
            });

            /**
             * Check that required hardware counters match
             */
            ASSERT_TRUE(profile->agent);
            auto       name_str     = std::string(profile->agent->name);
            const auto asts         = counters::get_ast_map();
            auto       req_counters = counters::get_required_hardware_counters(
                asts->arch_to_counter_asts, name_str, metric);
            for(const auto& req_metric : *req_counters)
            {
                if(req_metric.constant().empty())
                {
                    EXPECT_GT(profile->reqired_hw_counters.count(req_metric), 0)
                        << "Could not find metric - " << req_metric.name();
                    profile->reqired_hw_counters.erase(req_metric);
                }
                else
                {
                    EXPECT_GT(profile->required_special_counters.count(req_metric), 0)
                        << "Could not find metric - " << req_metric.name();
                    profile->required_special_counters.erase(req_metric);
                }
            }
            EXPECT_TRUE(profile->required_special_counters.empty())
                << "Special counters list larger than expected";
            EXPECT_TRUE(profile->reqired_hw_counters.empty())
                << "HW Counter list larger than expected";
        }
    }
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
user_dispatch_cb(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                 rocprofiler_counter_config_id_t*             config,
                 rocprofiler_user_data_t*                     user_data,
                 void*                                        callback_data_args)
{
    expected_dispatch& expected = *static_cast<expected_dispatch*>(callback_data_args);

    auto agent_id       = dispatch_data.dispatch_info.agent_id;
    auto queue_id       = dispatch_data.dispatch_info.queue_id;
    auto correlation_id = dispatch_data.correlation_id;
    auto kernel_id      = dispatch_data.dispatch_info.kernel_id;
    auto dispatch_id    = dispatch_data.dispatch_info.dispatch_id;

    EXPECT_EQ(sizeof(rocprofiler_dispatch_counting_service_data_t), dispatch_data.size);
    EXPECT_EQ(expected.kernel_id, kernel_id);
    EXPECT_EQ(expected.dispatch_id, dispatch_id);
    EXPECT_EQ(expected.agent_id, agent_id);
    EXPECT_EQ(expected.queue_id.handle, queue_id.handle);
    EXPECT_EQ(expected.correlation_id.internal, correlation_id.internal);
    EXPECT_EQ(expected.correlation_id.external.ptr, correlation_id.external.ptr);
    EXPECT_EQ(expected.correlation_id.external.value, correlation_id.external.value);
    EXPECT_EQ(expected.workgroup_size, dispatch_data.dispatch_info.workgroup_size);
    EXPECT_EQ(expected.grid_size, dispatch_data.dispatch_info.grid_size);

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

TEST(core, check_callbacks)
{
    int64_t count = 0;
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    context::context ctx;
    ctx.counter_collection =
        std::make_unique<rocprofiler::context::dispatch_counter_collection_service>();
    ctx.counter_collection->enabled.wlock([](auto& data) { data = true; });

    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    hsa::get_queue_controller()->disable_serialization();

    for(const auto& [_, agent] : agents)
    {
        /**
         * Setup
         */
        rocprofiler_queue_id_t qid = {.handle = static_cast<uint64_t>(count++)};
        hsa::FakeQueue         fq(agent, qid);
        auto                   metrics = findDeviceMetrics(agent, {});
        ASSERT_FALSE(metrics.empty());
        ASSERT_TRUE(agent.get_rocp_agent());
        for(auto& metric : metrics)
        {
            /**
             * Do not check expression evaluation here. This is checked as part of evaluate_ast
             * tests in a more controlled manner (aka, not requiring construction of the AST here).
             */
            if(!metric.expression().empty()) continue;

            /**
             * Setup
             */
            expected_dispatch        expected = {};
            rocprofiler_counter_id_t id       = {.handle = metric.id()};
            ROCPROFILER_CALL(
                rocprofiler_create_counter_config(agent.get_rocp_agent()->id, &id, 1, &expected.id),
                "Unable to create profile");
            auto profile = counters::get_counter_config(expected.id);
            ASSERT_TRUE(profile);

            std::shared_ptr<counters::counter_callback_info> cb_info =
                std::make_shared<counters::counter_callback_info>();
            cb_info->user_cb       = user_dispatch_cb;
            cb_info->callback_args = static_cast<void*>(&expected);

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

            hsa::Queue::queue_info_session_t::external_corr_id_map_t extern_ids = {};

            auto user_data = rocprofiler_user_data_t{.value = corr_id.internal};
            auto ret_pkt   = counters::queue_cb(&ctx,
                                              cb_info,
                                              fq,
                                              pkt,
                                              expected.kernel_id,
                                              expected.dispatch_id,
                                              &user_data,
                                              extern_ids,
                                              &corr_id);

            ASSERT_TRUE(ret_pkt.pkt)
                << fmt::format("Expected a packet to be generated for - {}", metric.name());

            /**
             * Create the buffer and run test
             */
            rocprofiler_buffer_id_t opt_buff_id = {.handle = 0};

            ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                       500 * sizeof(size_t),
                                                       500 * sizeof(size_t),
                                                       ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                       buffered_callback,
                                                       nullptr,
                                                       &opt_buff_id),
                             "Could not create buffer");
            cb_info->buffer = opt_buff_id;

            auto _sess           = hsa::Queue::queue_info_session_t{.queue = fq};
            _sess.correlation_id = &corr_id;

            auto sess = std::make_shared<hsa::Queue::queue_info_session_t>(std::move(_sess));

            counters::inst_pkt_t pkts;
            pkts.emplace_back(
                std::make_pair(std::move(ret_pkt.pkt), static_cast<counters::ClientID>(0)));
            completed_cb(&ctx, cb_info, sess, pkts, kernel_dispatch::profiling_time{});
            rocprofiler_flush_buffer(opt_buff_id);
            rocprofiler_destroy_buffer(opt_buff_id);
        }
    }
    registration::set_init_status(1);

    registration::finalize();
}

TEST(core, destroy_counter_profile)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    for(const auto& [_, agent] : agents)
    {
        auto metrics = findDeviceMetrics(agent, {});
        ASSERT_FALSE(metrics.empty());
        ASSERT_TRUE(agent.get_rocp_agent());
        for(auto& metric : metrics)
        {
            expected_dispatch        expected = {};
            rocprofiler_counter_id_t id       = {.handle = metric.id()};
            ROCPROFILER_CALL(
                rocprofiler_create_counter_config(agent.get_rocp_agent()->id, &id, 1, &expected.id),
                "Unable to create profile");
            ROCPROFILER_CALL(rocprofiler_destroy_counter_config(expected.id),
                             "Could not delete profile id");
            /**
             * Check the profile was actually destroyed
             */
            auto profile = counters::get_counter_config(expected.id);
            EXPECT_FALSE(profile);
        }
    }
    registration::set_init_status(1);

    registration::finalize();
}

TEST(core, start_stop_buffered_ctx)
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

    ROCPROFILER_CALL(rocprofiler_configure_buffer_dispatch_counting_service(
                         get_client_ctx(), opt_buff_id, null_dispatch_callback, (void*) 0x12345),
                     "Could not setup buffered service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    /**
     * Check that the context was actually started
     */
    auto* ctx_p = context::get_mutable_registered_context(get_client_ctx());
    ASSERT_TRUE(ctx_p);
    auto& ctx = *ctx_p;

    ASSERT_TRUE(ctx.counter_collection);
    ASSERT_EQ(ctx.counter_collection->callbacks.size(), 1);
    EXPECT_EQ(ctx.counter_collection->callbacks.at(0)->user_cb, null_dispatch_callback);
    EXPECT_EQ(ctx.counter_collection->callbacks.at(0)->callback_args, (void*) 0x12345);
    EXPECT_EQ(ctx.counter_collection->callbacks.at(0)->context.handle, get_client_ctx().handle);
    ASSERT_TRUE(ctx.counter_collection->callbacks.at(0)->buffer);
    EXPECT_EQ(ctx.counter_collection->callbacks.at(0)->buffer->handle, opt_buff_id.handle);

    bool found = false;
    ctx.counter_collection->enabled.rlock([&](const auto& data) { found = data; });
    EXPECT_TRUE(found);

    found = false;
    hsa::get_queue_controller()->iterate_callbacks([&](auto cid, const auto&) {
        if(cid == ctx.counter_collection->callbacks.at(0)->queue_id)
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
    ctx.counter_collection->enabled.rlock([&](const auto& data) { found = data; });
    EXPECT_FALSE(found);

    rocprofiler_flush_buffer(opt_buff_id);
    rocprofiler_destroy_buffer(opt_buff_id);

    registration::set_init_status(1);

    registration::finalize();
}

TEST(core, start_stop_callback_ctx)
{
    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_dispatch_counting_service(get_client_ctx(),
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

    ASSERT_TRUE(ctx.counter_collection);
    ASSERT_EQ(ctx.counter_collection->callbacks.size(), 1);
    EXPECT_EQ(ctx.counter_collection->callbacks.at(0)->user_cb, null_dispatch_callback);
    EXPECT_EQ(ctx.counter_collection->callbacks.at(0)->callback_args, (void*) 0x12345);
    EXPECT_EQ(ctx.counter_collection->callbacks.at(0)->record_callback, null_record_callback);
    EXPECT_EQ(ctx.counter_collection->callbacks.at(0)->record_callback_args, (void*) 0x54321);
    EXPECT_EQ(ctx.counter_collection->callbacks.at(0)->context.handle, get_client_ctx().handle);

    bool found = false;
    ctx.counter_collection->enabled.rlock([&](const auto& data) { found = data; });
    EXPECT_TRUE(found);

    found = false;
    hsa::get_queue_controller()->iterate_callbacks([&](auto cid, const auto&) {
        if(cid == ctx.counter_collection->callbacks.at(0)->queue_id)
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
    ctx.counter_collection->enabled.rlock([&](const auto& data) { found = data; });
    EXPECT_FALSE(found);

    registration::set_init_status(1);
    context::pop_client(1);
}

TEST(core, test_profile_incremental)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    for(const auto& [_, agent] : agents)
    {
        auto metrics = findDeviceMetrics(agent, {});
        ASSERT_FALSE(metrics.empty());
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
            ROCPROFILER_CALL(
                rocprofiler_create_counter_config(agent.get_rocp_agent()->id, &id, 1, &cfg_id),
                "Unable to create profile incrementally when we should be able to");
            EXPECT_NE(old_id.handle, cfg_id.handle)
                << "We expect that the handle changes this is due to the existing profile being "
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
            rocprofiler_counter_id_t id = {.handle = metric.id()};
            if(status =
                   rocprofiler_create_counter_config(agent.get_rocp_agent()->id, &id, 1, &cfg_id);
               status != ROCPROFILER_STATUS_SUCCESS)
            {
                break;
            }
        }
        EXPECT_EQ(status, ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT);
    }

    registration::set_init_status(1);

    registration::finalize();
}

TEST(core, public_api_iterate_agents)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    for(const auto& [_, agent] : agents)
    {
        std::set<uint64_t> from_api;

        // Iterate through the agents and get the counters available on that agent
        ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
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

        auto expected = findDeviceMetrics(agent, {});
        for(const auto& x : expected)
        {
            ASSERT_GT(from_api.count(x.id()), 0);
            from_api.erase(x.id());
        }
        EXPECT_TRUE(from_api.empty());
    }
}

TEST(core, check_load_counter_def_append)
{
    const std::string test_yaml = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters:
    - name: TEST_YAML_LOAD
      description: cycles
      properties: []
      definitions:
        - architectures:  
          - gfx950
          - gfx942
          - gfx10
          - gfx1010
          - gfx1030
          - gfx1031
          - gfx11
          - gfx1032
          - gfx1102
          - gfx906
          - gfx1100
          - gfx1101
          - gfx908
          - gfx90a
          - gfx9
          - gfx12
          - gfx1200
          - gfx1201
          expression: reduce(GRBM_GUI_ACTIVE,max)*CU_NUM
    )";
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(
        rocprofiler_load_counter_definition(
            test_yaml.c_str(), test_yaml.size(), ROCPROFILER_COUNTER_FLAG_APPEND_DEFINITION),
        "Could not load counter definition");
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    for(const auto& [_, agent] : agents)
    {
        EXPECT_EQ(findDeviceMetrics(agent, {"TEST_YAML_LOAD"}).size(), 1);
    }
}

TEST(core, check_load_counter_def)
{
    const std::string test_yaml = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters:  
    - name: GRBM_GUI_ACTIVE
      description: The GUI is Active
      properties: []
      definitions:
        - architectures:  
          - gfx950
          - gfx942
          - gfx941
          - gfx10
          - gfx1010
          - gfx1030
          - gfx1031
          - gfx11
          - gfx1032
          - gfx1102
          - gfx906
          - gfx1100
          - gfx1101
          - gfx940
          - gfx908
          - gfx900
          - gfx90a
          - gfx9
          - gfx12
          - gfx1200
          - gfx1201
          block: GRBM
          event: 2
    - name: TEST_YAML_LOAD
      description: cycles
      properties: []
      definitions:
        - architectures:  
          - gfx950
          - gfx942
          - gfx10
          - gfx1010
          - gfx1030
          - gfx1031
          - gfx11
          - gfx1032
          - gfx1102
          - gfx906
          - gfx1100
          - gfx1101
          - gfx908
          - gfx90a
          - gfx9
          - gfx12
          - gfx1200
          - gfx1201
          expression: reduce(GRBM_GUI_ACTIVE,max)
    )";
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(rocprofiler_load_counter_definition(
                         test_yaml.c_str(), test_yaml.size(), ROCPROFILER_COUNTER_FLAG_NONE),
                     "Could not load counter definition");
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    for(const auto& [_, agent] : agents)
    {
        // MAX_WAVE_SIZE should not be present
        EXPECT_EQ(
            findDeviceMetrics(agent, {"TEST_YAML_LOAD", "GRBM_GUI_ACTIVE", "MAX_WAVE_SIZE"}).size(),
            2);
    }
}

TEST(core, create_counter)
{
    std::vector<Metric> to_add = {
        Metric("", "SQ_WAVES_REDUCE_T", "", "", "", "reduce(SQ_WAVES,sum)", "", 10000),
        Metric("", "SQ_WAVES_AVR_T", "", "", "", "reduce(SQ_WAVES,avr)", "", 10001),
        Metric("", "SQ_WAVES_INVALID", "", "", "", "reduce(ABC,avr)", "", 10002),
    };

    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    for(const auto& [_, agent] : agents)
    {
        for(auto& metric : to_add)
        {
            rocprofiler_counter_id_t id;
            auto                     status  = rocprofiler_create_counter(metric.name().c_str(),
                                                     metric.name().size(),
                                                     metric.expression().c_str(),
                                                     metric.expression().size(),
                                                     metric.description().c_str(),
                                                     metric.description().size(),
                                                     agent.get_rocp_agent()->id,
                                                     &id);
            auto                     metrics = findDeviceMetrics(agent, {metric.name()});
            if(metric.name() == "SQ_WAVES_INVALID")
            {
                EXPECT_EQ(status, ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED);
                EXPECT_TRUE(metrics.empty());
            }
            else
            {
                EXPECT_EQ(metrics.size(), 1);
                EXPECT_EQ(metric.name(), metrics[0].name());
                EXPECT_EQ(metric.expression(), metrics[0].expression());
            }
        }
    }
}
