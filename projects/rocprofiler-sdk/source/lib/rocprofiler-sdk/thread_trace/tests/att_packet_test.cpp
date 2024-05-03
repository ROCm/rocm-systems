// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <functional>
#include <map>
#include <unordered_set>
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#define ROCPROFILER_CALL(ARG, MSG)                                                                 \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, ROCPROFILER_STATUS_SUCCESS) << MSG << " :: " << #ARG;                   \
    }

namespace rocprofiler
{
AmdExtTable&
get_ext_table()
{
    static auto _v = []() {
        auto val                                  = AmdExtTable{};
        val.hsa_amd_memory_pool_get_info_fn       = hsa_amd_memory_pool_get_info;
        val.hsa_amd_agent_iterate_memory_pools_fn = hsa_amd_agent_iterate_memory_pools;
        val.hsa_amd_memory_pool_allocate_fn       = hsa_amd_memory_pool_allocate;
        val.hsa_amd_memory_pool_free_fn           = hsa_amd_memory_pool_free;
        val.hsa_amd_agent_memory_pool_get_info_fn = hsa_amd_agent_memory_pool_get_info;
        val.hsa_amd_agents_allow_access_fn        = hsa_amd_agents_allow_access;
        return val;
    }();
    return _v;
}

CoreApiTable&
get_api_table()
{
    static auto _v = []() {
        auto val                       = CoreApiTable{};
        val.hsa_iterate_agents_fn      = hsa_iterate_agents;
        val.hsa_agent_get_info_fn      = hsa_agent_get_info;
        val.hsa_queue_create_fn        = hsa_queue_create;
        val.hsa_queue_destroy_fn       = hsa_queue_destroy;
        val.hsa_signal_wait_relaxed_fn = hsa_signal_wait_relaxed;
        return val;
    }();
    return _v;
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

}  // namespace rocprofiler

using namespace rocprofiler::aql;

TEST(thread_trace, construct_default_packets)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    rocprofiler::test_init();
    auto agents = rocprofiler::hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    for(const auto& [_, agent] : agents)
    {
        auto params = std::make_shared<rocprofiler::thread_trace_parameter_pack>();

        ThreadTraceAQLPacketFactory factory(
            agent, params, rocprofiler::get_api_table(), rocprofiler::get_ext_table());

        auto packet = factory.construct_packet();

        size_t vendor_packet = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
        ASSERT_TRUE(packet->start.header == vendor_packet);
        ASSERT_TRUE(packet->stop.header == vendor_packet);
        ASSERT_TRUE(packet->before_krn_pkt.size() > 0);
        ASSERT_TRUE(packet->after_krn_pkt.size() > 0);
    }
    hsa_shut_down();
}

TEST(thread_trace, configure_test)
{
    rocprofiler::test_init();

    rocprofiler::registration::init_logging();
    rocprofiler::registration::set_init_status(-1);
    rocprofiler::context::push_client(1);
    rocprofiler_context_id_t ctx;
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

    std::vector<rocprofiler_att_parameter_t> params;
    params.push_back({ROCPROFILER_ATT_PARAMETER_TARGET_CU, 1});
    params.push_back({ROCPROFILER_ATT_PARAMETER_SHADER_ENGINE_MASK, 0xF});
    params.push_back({ROCPROFILER_ATT_PARAMETER_BUFFER_SIZE, 0x1000000});
    params.push_back({ROCPROFILER_ATT_PARAMETER_SIMD_SELECT, 0xF});

    rocprofiler_configure_thread_trace_service(
        ctx,
        params.data(),
        params.size(),
        [](rocprofiler_queue_id_t,
           const rocprofiler_agent_t*,
           rocprofiler_correlation_id_t,
           const hsa_kernel_dispatch_packet_t*,
           uint64_t,
           void*) { return ROCPROFILER_ATT_CONTROL_NONE; },
        [](int64_t, void*, size_t, void*) {},
        nullptr);

    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    ROCPROFILER_CALL(rocprofiler_start_context(ctx), "context start failed");
    ROCPROFILER_CALL(rocprofiler_stop_context(ctx), "context stop failed");
}
