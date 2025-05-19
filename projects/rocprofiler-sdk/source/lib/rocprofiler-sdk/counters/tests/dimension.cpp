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

// Allow testing of deprecated calls
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/counters/dimensions.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

using namespace rocprofiler::counters::test_constants;

namespace
{
void
check_dim_pos(rocprofiler_counter_instance_id_t                                 test_id,
              rocprofiler::counters::rocprofiler_profile_counter_instance_types dim,
              size_t                                                            expected)
{
    EXPECT_EQ(rec_to_dim_pos(test_id, dim), expected);
    size_t pos = 0;
    rocprofiler_query_record_dimension_position(
        test_id, static_cast<rocprofiler_counter_dimension_id_t>(dim), &pos);
    EXPECT_EQ(pos, expected);
}

void
check_counter_id(rocprofiler_counter_instance_id_t id, uint64_t expected_handle)
{
    rocprofiler_counter_id_t api_id = {.handle = 0};
    rocprofiler_query_record_counter_id(id, &api_id);
    EXPECT_EQ(rocprofiler::counters::rec_to_counter_id(id).handle, expected_handle);
    EXPECT_EQ(rocprofiler::counters::rec_to_counter_id(id).handle, api_id.handle);
}
}  // namespace

TEST(dimension, set_get)
{
    using namespace rocprofiler::counters;
    int64_t                           max_counter_val = (std::numeric_limits<uint64_t>::max() >>
                               (64 - (DIM_BIT_LENGTH / ROCPROFILER_DIMENSION_LAST)));
    rocprofiler_counter_instance_id_t test_id         = 0;
    rocprofiler_counter_id_t          test_counter{.handle = 123};

    set_counter_in_rec(test_id, test_counter);
    // 0x007B000000000000 = decimal counter id 123 << DIM_BIT_LENGTH
    EXPECT_EQ(test_id, 0x007B000000000000);

    test_counter.handle = 321;
    set_counter_in_rec(test_id, test_counter);
    // 0x0141000000000000 = decimal counter id 321 << DIM_BIT_LENGTH
    EXPECT_EQ(test_id, 0x0141000000000000);
    check_counter_id(test_id, 321);

    // Test multiples of i, setting/getting those values across all
    // dimensions
    for(size_t multi_factor = 1; multi_factor < 7; multi_factor++)
    {
        for(size_t i = 1; i < static_cast<size_t>(ROCPROFILER_DIMENSION_LAST); i++)
        {
            auto dim = static_cast<rocprofiler_profile_counter_instance_types>(i);
            set_dim_in_rec(test_id, dim, i);
            check_dim_pos(test_id, dim, i);
            set_dim_in_rec(test_id, dim, i * multi_factor);
            for(size_t j = 1; j < static_cast<size_t>(ROCPROFILER_DIMENSION_LAST); j++)
            {
                if(i == j) continue;
                set_dim_in_rec(test_id,
                               static_cast<rocprofiler_profile_counter_instance_types>(j),
                               max_counter_val);
                check_dim_pos(test_id,
                              static_cast<rocprofiler_profile_counter_instance_types>(j),
                              max_counter_val);
                check_dim_pos(test_id, dim, i * multi_factor);
            }

            for(size_t j = static_cast<size_t>(ROCPROFILER_DIMENSION_LAST - 1); j > 0; j--)
            {
                if(i == j) continue;
                set_dim_in_rec(test_id,
                               static_cast<rocprofiler_profile_counter_instance_types>(j),
                               max_counter_val);
                check_dim_pos(
                    test_id, (rocprofiler_profile_counter_instance_types) j, max_counter_val);
                check_dim_pos(test_id, dim, i * multi_factor);
            }

            // Check that name exists
            EXPECT_TRUE(rocprofiler::common::get_val(
                rocprofiler::counters::dimension_map(),
                static_cast<rocprofiler_profile_counter_instance_types>(i)));
        }
    }

    for(size_t i = static_cast<size_t>(ROCPROFILER_DIMENSION_LAST - 1); i > 0; i--)
    {
        auto dim = static_cast<rocprofiler_profile_counter_instance_types>(i);
        set_dim_in_rec(test_id, dim, i * 5);
        check_dim_pos(test_id, dim, i * 5);
        set_dim_in_rec(test_id, dim, i * 3);
        check_dim_pos(test_id, dim, i * 3);
        for(size_t j = 1; j < 64; j++)
        {
            test_id = 0;
            set_dim_in_rec(test_id, dim, j);
            check_dim_pos(test_id, dim, j);
        }
    }

    test_counter.handle = 123;
    set_counter_in_rec(test_id, test_counter);
    check_counter_id(test_id, 123);

    // Test that all bits can be set/fetched for dims, 0xFAFBFCFDFEFF is a random
    // collection of 48 bits.
    set_dim_in_rec(test_id, ROCPROFILER_DIMENSION_NONE, 0xFAFBFCFDFEFF);
    check_dim_pos(test_id, ROCPROFILER_DIMENSION_NONE, 0xFAFBFCFDFEFF);
    check_counter_id(test_id, 123);
}

using namespace rocprofiler;

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

TEST(dimension, block_dim_test)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    for(const auto& [_, agent] : agents)
    {
        auto metrics = findDeviceMetrics(agent, {});
        ASSERT_FALSE(metrics.empty());
        ASSERT_TRUE(agent.get_rocp_agent());
        // aql::AQLPacketConstruct pkt(agent, metrics);
        // auto                    test_pkt = pkt.construct_packet(get_ext_table());
        for(const auto& metric : metrics)
        {
            /**
             * Calculate expected dimensions from AQL Profiler
             */
            std::unordered_map<counters::rocprofiler_profile_counter_instance_types, uint64_t>
                rocp_dims;
            ROCP_INFO << metric.name() << " " << metric.constant();
            if(!metric.constant().empty())
            {
                rocp_dims[counters::rocprofiler_profile_counter_instance_types::
                              ROCPROFILER_DIMENSION_INSTANCE] = 1;
            }
            else if(!metric.expression().empty())
            {
                continue;
            }
            else
            {
                aql::CounterPacketConstruct pkt_gen(agent.get_rocp_agent()->id, {metric});
                const auto&                 events = pkt_gen.get_counter_events(metric);
                for(const auto& event : events)
                {
                    std::map<int, uint64_t> dims;
                    auto status = aql::get_dim_info(agent.get_rocp_agent()->id, event, 0, dims);
                    CHECK_EQ(status, ROCPROFILER_STATUS_SUCCESS)
                        << rocprofiler_get_status_string(status);
                    for(const auto& [id, extent] : dims)
                    {
                        if(const auto* inst_type = rocprofiler::common::get_val(
                               counters::aqlprofile_id_to_rocprof_instance(), id))
                        {
                            rocp_dims.emplace(*inst_type, 0).first->second = extent;
                        }
                    }
                }
            }

            /**
             * Compare with actual
             */
            auto dims = getBlockDimensions(agent.name(), metric);
            EXPECT_FALSE(dims.empty());
            EXPECT_EQ(dims.size(), rocp_dims.size());
            for(const auto& dim : dims)
            {
                const auto* ptr = rocprofiler::common::get_val(rocp_dims, dim.type());
                ASSERT_TRUE(ptr) << fmt::format("{}", dim);
                EXPECT_EQ(*ptr, dim.size()) << fmt::format("{}", dim);
                EXPECT_EQ(std::string(counters::dimension_map().at(dim.type())), dim.name())
                    << fmt::format("{}", dim);
            }

            /**
             * Check this value exists in the dimension cache
             */
            auto        dim_ptr   = counters::get_dimension_cache();
            const auto* dim_cache = rocprofiler::common::get_val(dim_ptr->id_to_dim, metric.id());
            ASSERT_TRUE(dim_cache);
            EXPECT_EQ(fmt::format("{}", fmt::join(dims, "|")),
                      fmt::format("{}", fmt::join(*dim_cache, "|")));

            /**
             * Check counter instance count public API
             */
            size_t instance_count            = 0;
            size_t calculated_instance_count = 0;
            rocprofiler_query_counter_instance_count(
                agent.get_rocp_agent()->id, {.handle = metric.id()}, &instance_count);
            for(const auto& dim : dims)
            {
                if(calculated_instance_count == 0)
                    calculated_instance_count = dim.size();
                else if(dim.size() > 0)
                    calculated_instance_count = dim.size() * calculated_instance_count;
            }
            EXPECT_EQ(instance_count, calculated_instance_count);

            /**
             * Check the public API returns this value
             */
            rocprofiler_iterate_counter_dimensions(
                {.handle = metric.id()},
                [](rocprofiler_counter_id_t,
                   const rocprofiler_counter_record_dimension_info_t* dim_info,
                   size_t                                             num_dims,
                   void* user_data) -> rocprofiler_status_t {
                    auto expected_dims = *static_cast<
                        std::unordered_map<counters::rocprofiler_profile_counter_instance_types,
                                           uint64_t>*>(user_data);
                    EXPECT_EQ(num_dims, expected_dims.size());
                    for(size_t i = 0; i < num_dims; i++)
                    {
                        const auto* lookup_ptr = rocprofiler::common::get_val(
                            expected_dims,
                            static_cast<counters::rocprofiler_profile_counter_instance_types>(
                                dim_info[i].id));
                        EXPECT_TRUE(lookup_ptr);
                        if(!lookup_ptr) return ROCPROFILER_STATUS_ERROR;
                        EXPECT_EQ(*lookup_ptr, dim_info[i].instance_size);
                        EXPECT_EQ(
                            counters::dimension_map().at(
                                static_cast<counters::rocprofiler_profile_counter_instance_types>(
                                    dim_info[i].id)),
                            std::string(dim_info[i].name));
                    }
                    return ROCPROFILER_STATUS_SUCCESS;
                },
                static_cast<void*>(&rocp_dims));
        }
    }

    hsa_shut_down();
}
#pragma GCC diagnostic pop
