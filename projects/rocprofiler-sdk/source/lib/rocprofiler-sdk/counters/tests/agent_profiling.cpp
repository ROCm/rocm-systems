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

#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/counters/tests/code_object_loader.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/dispatch_profile.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <tuple>

using namespace rocprofiler::counters::test_constants;
using namespace rocprofiler::counters::testing;
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
    const auto*                   all_counters = counters::getMetricMap();

    ROCP_ERROR << "Looking up counters for " << std::string(agent.name());
    const auto* gfx_metrics = common::get_val(*all_counters, std::string(agent.name()));
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
    rocprofiler::hsa::copy_table(table.core_, 0);
    rocprofiler::hsa::copy_table(table.amd_ext_, 0);
    agent::construct_agent_cache(&table);
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
}

std::vector<rocprofiler_record_counter_t>&
global_recs()
{
    static std::vector<rocprofiler_record_counter_t> recs;
    return recs;
}

void
check_output_created(rocprofiler_context_id_t,
                     rocprofiler_buffer_id_t,
                     rocprofiler_record_header_t** headers,
                     size_t                        num_headers,
                     void*                         user_data,
                     uint64_t)
{
    // verifies that we got a record containing some data for a counter
    // does NOT validate the counters values.
    if(user_data == nullptr) return;

    uint64_t found_value = 0;
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
           header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
        {}
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
                header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            // Print the returned counter data.
            auto* record = static_cast<rocprofiler_record_counter_t*>(header->payload);
            if(found_value != 0 && found_value != record->user_data.value)
            {
                ROCP_FATAL << "Have records with different user data values we didn't expect";
                break;
            }
            found_value = record->user_data.value;
            // ROCP_ERROR << fmt::format("Found counter value: {}", record->counter_value);
            global_recs().push_back(*record);
        }
    }

    auto* signal = reinterpret_cast<hsa_signal_t*>(user_data);
    hsa_signal_store_relaxed(*signal, static_cast<int64_t>(found_value));
}

struct test_kernels
{
    CodeObject obj;

    test_kernels(const rocprofiler::hsa::AgentCache& agent)
    {
        CHECK(agent.get_rocp_agent());
        // Getting hasco Path
        std::string hasco_file_path =
            std::string(agent.get_rocp_agent()->name) + std::string("_agent_kernels.hsaco");
        search_hasco(common::filesystem::current_path(), hasco_file_path);
        CHECK_EQ(load_code_object(hasco_file_path, agent.get_hsa_agent(), obj), HSA_STATUS_SUCCESS);
    }

    uint64_t load_kernel(const rocprofiler::hsa::AgentCache& agent,
                         const std::string&                  kernel_name) const
    {
        Kernel kern;
        CHECK_EQ(get_kernel(obj, kernel_name, agent.get_hsa_agent(), kern), HSA_STATUS_SUCCESS);
        return kern.handle;
    }
};

uint16_t
packet_header(hsa_packet_type_t type)
{
    uint16_t header = type << HSA_PACKET_HEADER_TYPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    return header;
}

rocprofiler::hsa::rocprofiler_packet
gen_kernel_pkt(uint64_t obj)
{
    rocprofiler::hsa::rocprofiler_packet packet{};
    memset(((uint8_t*) &packet.kernel_dispatch) + 4, 0, sizeof(hsa_kernel_dispatch_packet_t) - 4);
    packet.kernel_dispatch.setup            = 1;
    packet.kernel_dispatch.header           = packet_header(HSA_PACKET_TYPE_KERNEL_DISPATCH);
    packet.kernel_dispatch.workgroup_size_x = 1;
    packet.kernel_dispatch.workgroup_size_y = 1;
    packet.kernel_dispatch.workgroup_size_z = 1;
    packet.kernel_dispatch.grid_size_x      = 1;
    packet.kernel_dispatch.grid_size_y      = 1;
    packet.kernel_dispatch.grid_size_z      = 1;
    packet.kernel_dispatch.kernel_object    = obj;
    packet.kernel_dispatch.kernarg_address  = nullptr;
    packet.kernel_dispatch.completion_signal.handle = 0;
    ROCP_ERROR << fmt::format("{:x}", packet.kernel_dispatch.kernel_object);
    return packet;
}

uint64_t
submitPacket(hsa_queue_t* queue, const void* packet)
{
    const uint32_t slot_size_b = 0x40;

    // advance command queue
    const uint64_t write_idx = hsa_queue_add_write_index_scacq_screl(queue, 1);
    while((write_idx - hsa_queue_load_read_index_relaxed(queue)) >= queue->size)
    {
        sched_yield();
    }

    const uint32_t slot_idx = (uint32_t)(write_idx % queue->size);
    // NOLINTBEGIN(performance-no-int-to-ptr)
    uint32_t* queue_slot =
        reinterpret_cast<uint32_t*>((uintptr_t)(queue->base_address) + (slot_idx * slot_size_b));
    const uint32_t* slot_data = reinterpret_cast<const uint32_t*>(packet);

    // Copy buffered commands into the queue slot.
    // Overwrite the AQL invalid header (first dword) last.
    // This prevents the slot from being read until it's fully written.
    memcpy(&queue_slot[1], &slot_data[1], slot_size_b - sizeof(uint32_t));
    std::atomic<uint32_t>* header_atomic_ptr =
        reinterpret_cast<std::atomic<uint32_t>*>(&queue_slot[0]);
    // NOLINTEND(performance-no-int-to-ptr)
    header_atomic_ptr->store(slot_data[0], std::memory_order_release);

    // ringdoor bell
    hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);

    return write_idx;
}

}  // namespace

class agent_profile_test : public ::testing::Test
{
protected:
    agent_profile_test() {}

    static void test_run(rocprofiler_counter_flag_t flags = ROCPROFILER_COUNTER_FLAG_NONE,
                         const std::unordered_set<std::string>& test_metrics = {},
                         size_t                                 delay        = 1)
    {
        hsa_init();
        registration::init_logging();
        registration::set_init_status(-1);
        context::push_client(1);
        test_init();
        // rocprofiler_debugger_block();
        counters::agent_profile_hsa_registration();

        std::string kernel_name = "null_kernel";

        ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
        ASSERT_GT(hsa::get_queue_controller()->get_supported_agents().size(), 0);
        for(const auto& [_, agent] : hsa::get_queue_controller()->get_supported_agents())
        {
            auto metrics = findDeviceMetrics(agent, test_metrics);
            ASSERT_FALSE(metrics.empty());
            ASSERT_TRUE(agent.get_rocp_agent());
            test_kernels kernel_loader(agent);
            auto         kernel_handle = kernel_loader.load_kernel(agent, kernel_name);
            auto         kernel_pkt    = gen_kernel_pkt(kernel_handle);

            hsa_queue_t* queue;
            CHECK_EQ(hsa_queue_create(agent.get_hsa_agent(),
                                      64,
                                      HSA_QUEUE_TYPE_SINGLE,
                                      nullptr,
                                      nullptr,
                                      UINT32_MAX,
                                      UINT32_MAX,
                                      &queue),
                     HSA_STATUS_SUCCESS);

            // We don't use the queue interceptor, need to enabling profiling manually
            hsa_amd_profiling_set_profiler_enabled(queue, 1);

            hsa_signal_t completion_signal;
            hsa_signal_create(1, 0, nullptr, &completion_signal);

            CHECK(agent.cpu_pool().handle != 0);
            CHECK(agent.get_hsa_agent().handle != 0);
            // Set state of the queue to allow profiling (may not be needed since AQL
            // may do this in the future).
            aql::set_profiler_active_on_queue(
                agent.cpu_pool(), agent.get_hsa_agent(), [&](hsa::rocprofiler_packet pkt) {
                    pkt.ext_amd_aql_pm4.completion_signal = completion_signal;
                    submitPacket(queue, (void*) &pkt);

                    if(hsa_signal_wait_relaxed(completion_signal,
                                               HSA_SIGNAL_CONDITION_EQ,
                                               0,
                                               20000000,
                                               HSA_WAIT_STATE_BLOCKED) != 0)
                    {
                        ROCP_FATAL << "Failed to set profiling mode on queue";
                    }
                    hsa_signal_store_relaxed(completion_signal, 1);
                });

            rocprofiler::hsa::rocprofiler_packet barrier{};

            hsa_signal_create(1, 0, nullptr, &completion_signal);
            barrier.barrier_and.header            = packet_header(HSA_PACKET_TYPE_BARRIER_AND);
            barrier.barrier_and.completion_signal = completion_signal;

            hsa_signal_t found_data;
            hsa_signal_create(0, 0, nullptr, &found_data);
            size_t track_metric = 0;
            for(auto& metric : metrics)
            {
                // global_recs().clear();
                track_metric++;
                ROCP_ERROR << "Testing metric " << metric.name();
                rocprofiler_context_id_t ctx = {.handle = 0};
                ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");
                rocprofiler_buffer_id_t opt_buff_id = {.handle = 0};
                ROCPROFILER_CALL(rocprofiler_create_buffer(ctx,
                                                           500 * sizeof(size_t),
                                                           500 * sizeof(size_t),
                                                           ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                           check_output_created,
                                                           &found_data,
                                                           &opt_buff_id),
                                 "Could not create buffer");
                /**
                 * Check profile construction
                 */
                rocprofiler_profile_config_id_t cfg_id = {};
                rocprofiler_counter_id_t        id     = {.handle = metric.id()};
                ROCPROFILER_CALL(
                    rocprofiler_create_profile_config(agent.get_rocp_agent()->id, &id, 1, &cfg_id),
                    "Unable to create profile");

                ROCPROFILER_CALL(
                    rocprofiler_configure_agent_profile_counting_service(
                        ctx,
                        opt_buff_id,
                        agent.get_rocp_agent()->id,
                        [](rocprofiler_context_id_t context_id,
                           rocprofiler_agent_id_t,
                           rocprofiler_agent_set_profile_callback_t set_config,
                           void*                                    user_data) {
                            CHECK(user_data);
                            if(auto status = set_config(
                                   context_id,
                                   *static_cast<rocprofiler_profile_config_id_t*>(user_data));
                               status != ROCPROFILER_STATUS_SUCCESS)
                            {
                                ROCP_FATAL << rocprofiler_get_status_string(status);
                            }
                        },
                        static_cast<void*>(&cfg_id)),
                    "Could not create agent collection");

                // This queue will only be present if a context exists when AgentCache is
                // construction This is a workaround for the test environment since we create
                // contexts after AgentCache constructed.
                agent::get_agent_cache(agent.get_rocp_agent())
                    ->init_agent_profile_queue(get_api_table(), get_ext_table());

                hsa_signal_store_screlease(completion_signal, 1);
                hsa_signal_store_screlease(found_data, 0);
                auto status = rocprofiler_start_context(ctx);
                if(status == ROCPROFILER_STATUS_ERROR_NO_HARDWARE_COUNTERS)
                {
                    ROCP_ERROR << fmt::format("No hardware counters for {}, skipping",
                                              metric.name());
                    continue;
                }
                else if(status != ROCPROFILER_STATUS_SUCCESS)
                {
                    ROCP_FATAL << "Failed to start context - "
                               << rocprofiler_get_status_string(status);
                }

                ROCPROFILER_CALL(status, "Could not start context");

                // Execute kernel
                submitPacket(queue, &kernel_pkt);
                submitPacket(queue, &kernel_pkt);
                submitPacket(queue, &kernel_pkt);
                submitPacket(queue, &kernel_pkt);
                submitPacket(queue, &kernel_pkt);
                submitPacket(queue, &barrier);
                usleep(delay);
                // Wait for completion
                hsa_signal_wait_relaxed(completion_signal,
                                        HSA_SIGNAL_CONDITION_EQ,
                                        0,
                                        UINT64_MAX,
                                        HSA_WAIT_STATE_BLOCKED);

                // Sample the counting service.
                ROCPROFILER_CALL(rocprofiler_sample_agent_profile_counting_service(
                                     ctx, {.value = track_metric}, flags),
                                 "Could not sample");
                ROCPROFILER_CALL(rocprofiler_stop_context(ctx), "Could not stop context");
                rocprofiler_flush_buffer(opt_buff_id);

                if(hsa_signal_wait_relaxed(found_data,
                                           HSA_SIGNAL_CONDITION_EQ,
                                           track_metric,
                                           20000000,
                                           HSA_WAIT_STATE_BLOCKED) !=
                   static_cast<int64_t>(track_metric))
                {
                    ROCP_FATAL << "Failed to get data for " << metric.name();
                }
            }
            hsa_signal_destroy(completion_signal);
            hsa_signal_destroy(found_data);
            hsa_queue_destroy(queue);
        }
        registration::set_init_status(1);
        context::pop_client(1);
    }
};

TEST_F(agent_profile_test, sync_counters) { test_run(); }
TEST_F(agent_profile_test, async_counters) { test_run(ROCPROFILER_COUNTER_FLAG_ASYNC); }
TEST_F(agent_profile_test, sync_grbm_verify)
{
    test_run(ROCPROFILER_COUNTER_FLAG_NONE, {"GRBM_COUNT"}, 50000);
    ROCP_ERROR << global_recs().size();

    for(const auto& val : global_recs())
    {
        rocprofiler_counter_id_t id;
        rocprofiler_query_record_counter_id(val.id, &id);
        rocprofiler_counter_info_v0_t info;
        rocprofiler_query_counter_info(id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info);
        ROCP_ERROR << fmt::format("Name: {} Counter value: {}", info.name, val.counter_value);
        EXPECT_GT(val.counter_value, 0.0);
    }
}

TEST_F(agent_profile_test, sync_gpu_util_verify)
{
    test_run(ROCPROFILER_COUNTER_FLAG_NONE, {"GPU_UTIL"}, 50000);
    ROCP_ERROR << global_recs().size();

    for(const auto& val : global_recs())
    {
        rocprofiler_counter_id_t id;
        rocprofiler_query_record_counter_id(val.id, &id);
        rocprofiler_counter_info_v0_t info;
        rocprofiler_query_counter_info(id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info);
        ROCP_ERROR << fmt::format("Name: {} Counter value: {}", info.name, val.counter_value);
        EXPECT_GT(val.counter_value, 0.0);
    }
}

TEST_F(agent_profile_test, sync_sq_waves_verify)
{
    test_run(ROCPROFILER_COUNTER_FLAG_NONE, {"SQ_WAVES_sum"}, 50000);
    ROCP_ERROR << global_recs().size();

    for(const auto& val : global_recs())
    {
        rocprofiler_counter_id_t id;
        rocprofiler_query_record_counter_id(val.id, &id);
        rocprofiler_counter_info_v0_t info;
        rocprofiler_query_counter_info(id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info);
        ROCP_ERROR << fmt::format("Name: {} Counter value: {}", info.name, val.counter_value);
        EXPECT_GT(val.counter_value, 0.0);
    }
}
