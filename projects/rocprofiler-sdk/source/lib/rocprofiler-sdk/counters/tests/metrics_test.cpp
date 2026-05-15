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

#include "metrics_test.h"

#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/counters/dimensions.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace
{
namespace counters = ::rocprofiler::counters;

auto
loadTestData(const std::unordered_map<std::string, std::vector<std::vector<std::string>>>& map)
{
    std::unordered_map<std::string, std::vector<counters::Metric>> ret;
    for(const auto& [gfx, dataMap] : map)
    {
        auto& metric_vec = ret.emplace(gfx, std::vector<counters::Metric>{}).first->second;
        for(const auto& data_vec : dataMap)
        {
            metric_vec.emplace_back("gfx9",
                                    data_vec.at(0),
                                    data_vec.at(1),
                                    data_vec.at(2),
                                    data_vec.at(4),
                                    data_vec.at(3),
                                    "",
                                    0);
        }
    }
    return ret;
}
}  // namespace

TEST(metrics, base_load)
{
    auto loaded_metrics = counters::loadMetrics();
    auto rocp_data      = [&]() {
        // get only base metrics (those without expressions)
        std::unordered_map<std::string, std::vector<counters::Metric>> ret;
        for(const auto& [gfx, metrics] : loaded_metrics->arch_to_metric)
        {
            std::vector<counters::Metric> base_metrics;
            std::copy_if(metrics.begin(),
                         metrics.end(),
                         std::back_inserter(base_metrics),
                         [](const auto& m) { return m.expression().empty(); });
            if(!base_metrics.empty()) ret.emplace(gfx, std::move(base_metrics));
        }
        return ret;
    }();
    auto test_data = loadTestData(basic_gfx908);

    ASSERT_EQ(rocp_data.count("gfx908"), 1);
    ASSERT_EQ(test_data.count("gfx908"), 1);
    auto rocp_data_v = rocp_data.at("gfx908");
    auto test_data_v = test_data.at("gfx908");
    // get_agent_available_properties() is the metrics added for fields in agent.hpp
    EXPECT_EQ(rocp_data_v.size(),
              test_data_v.size() + rocprofiler::agent::get_agent_available_properties().size());
    auto find = [&rocp_data_v](const auto& v) -> std::optional<counters::Metric> {
        for(const auto& ditr : rocp_data_v)
        {
            ROCP_INFO << fmt::format("{}", ditr);
            if(ditr.name() == v.name()) return ditr;
        }
        return std::nullopt;
    };
    auto equal = [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.name(), lhs.block(), lhs.event(), lhs.description()) ==
               std::tie(rhs.name(), rhs.block(), rhs.event(), rhs.description());
    };
    for(const auto& itr : test_data_v)
    {
        auto val = find(itr);
        if(!val)
        {
            EXPECT_TRUE(val) << "failed to find " << fmt::format("{}", itr);
            continue;
        }
        EXPECT_TRUE(equal(itr, *val)) << fmt::format("\n\t{} \n\t\t!= \n\t{}", itr, *val);
    }
}

TEST(metrics, derived_load)
{
    auto loaded_metrics = counters::loadMetrics();
    auto rocp_data      = [&]() {
        // get only derrived metrics
        std::unordered_map<std::string, std::vector<counters::Metric>> ret;
        for(const auto& [gfx, metrics] : loaded_metrics->arch_to_metric)
        {
            std::vector<counters::Metric> derived_metrics;
            std::copy_if(metrics.begin(),
                         metrics.end(),
                         std::back_inserter(derived_metrics),
                         [](const auto& m) { return !m.expression().empty(); });
            if(!derived_metrics.empty()) ret.emplace(gfx, std::move(derived_metrics));
        }
        return ret;
    }();

    auto test_data = loadTestData(derived_gfx908);
    ASSERT_EQ(rocp_data.count("gfx908"), 1);
    ASSERT_EQ(test_data.count("gfx908"), 1);
    auto rocp_data_v = rocp_data.at("gfx908");
    auto test_data_v = test_data.at("gfx908");
    EXPECT_EQ(rocp_data_v.size(), test_data_v.size());
    auto find = [&rocp_data_v](const auto& v) -> std::optional<counters::Metric> {
        for(const auto& ditr : rocp_data_v)
            if(ditr.name() == v.name()) return ditr;
        return std::nullopt;
    };
    auto equal = [](const auto& lhs, const auto& rhs) {
        return std::tie(
                   lhs.name(), lhs.block(), lhs.event(), lhs.description(), lhs.expression()) ==
               std::tie(rhs.name(), rhs.block(), rhs.event(), rhs.description(), rhs.expression());
    };
    for(const auto& itr : test_data_v)
    {
        auto val = find(itr);
        if(!val)
        {
            EXPECT_TRUE(val) << "failed to find " << fmt::format("{}", itr);
            continue;
        }
        EXPECT_TRUE(equal(itr, *val)) << fmt::format("\n\t{} \n\t\t!= \n\t{}", itr, *val);
    }
}

TEST(metrics, check_agent_valid)
{
    auto        mets      = counters::loadMetrics();
    const auto& rocp_data = mets->arch_to_metric;

    auto common_metrics = [&]() -> std::set<uint64_t> {
        std::set<uint64_t> ret;
        for(const auto& [gfx, counters] : rocp_data)
        {
            std::set<uint64_t> counter_ids;
            for(const auto& metric : counters)
            {
                counter_ids.insert(metric.id());
            }

            if(ret.empty())
            {
                ret = counter_ids;
            }
            else
            {
                std::set<uint64_t> out_intersection;
                std::set_intersection(ret.begin(),
                                      ret.end(),
                                      counter_ids.begin(),
                                      counter_ids.end(),
                                      std::inserter(out_intersection, out_intersection.begin()));
            }

            if(ret.empty()) return ret;
        }
        return ret;
    }();

    for(const auto& [gfx, counters] : rocp_data)
    {
        for(const auto& metric : counters)
        {
            ASSERT_EQ(counters::checkValidMetric(gfx, metric), true)
                << gfx << " " << fmt::format("{}", metric);
        }

        for(const auto& [other_gfx, other_counters] : rocp_data)
        {
            if(other_gfx == gfx) continue;
            for(const auto& metric : other_counters)
            {
                if(common_metrics.count(metric.id()) > 0 || !metric.constant().empty()) continue;
                EXPECT_EQ(counters::checkValidMetric(gfx, metric), false)
                    << fmt::format("GFX {} has Metric {} but shouldn't", gfx, metric);
            }
        }
    }
}

TEST(metrics, check_public_api_query)
{
    auto        metrics_map = rocprofiler::counters::loadMetrics();
    const auto& id_map      = metrics_map->id_to_metric;
    for(const auto& [id, metric] : id_map)
    {
        rocprofiler_counter_info_v1_t info;

        // Note: Direct dimension cache access requires an agent_id, which is not available
        // in this unit test. The API call below handles agent lookup internally.
        // auto dim_ptr = rocprofiler::counters::get_dimension_cache(agent_id);
        // const auto* dims = rocprofiler::common::get_val(dim_ptr->id_to_dim, metric.id());

        auto status = rocprofiler_query_counter_info(
            {.handle = id}, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info));

        // Skip dimension checks if no agent is found (happens in unit tests without GPU)
        if(status == ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND) continue;

        ASSERT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(std::string(info.name ? info.name : ""), metric.name());
        EXPECT_EQ(std::string(info.block ? info.block : ""), metric.block());
        EXPECT_EQ(std::string(info.expression ? info.expression : ""), metric.expression());
        EXPECT_EQ(info.is_derived, !metric.expression().empty());
        EXPECT_EQ(std::string(info.description ? info.description : ""), metric.description());

        // Dimensions are now verified through the API call above
        for(size_t i = 0; i < info.dimensions_count; i++)
        {
            EXPECT_GT(info.dimensions[i]->instance_size, 0u);
            EXPECT_TRUE(info.dimensions[i]->name != nullptr);
        }

        size_t instance_count = 0;
        // Calculate expected instance count from API-returned dimensions
        for(size_t i = 0; i < info.dimensions_count; i++)
        {
            if(instance_count == 0)
                instance_count = info.dimensions[i]->instance_size;
            else if(info.dimensions[i]->instance_size > 0)
                instance_count = info.dimensions[i]->instance_size * instance_count;
        }

        EXPECT_EQ(info.dimensions_instances_count, instance_count);
        std::set<std::vector<size_t>> dim_permutations;

        for(size_t i = 0; i < info.dimensions_instances_count; i++)
        {
            std::vector<size_t> dim_ids;
            ASSERT_EQ(
                rocprofiler::counters::rec_to_counter_id(info.dimensions_instances[i]->instance_id)
                    .handle,
                metric.id());
            for(size_t j = 0; j < info.dimensions_count; j++)
            {
                dim_ids.push_back(rocprofiler::counters::rec_to_dim_pos(
                    info.dimensions_instances[i]->instance_id,
                    static_cast<rocprofiler::counters::rocprofiler_profile_counter_instance_types>(
                        info.dimensions[j]->id)));
            }
            // Ensure that the permutation is unique
            ASSERT_EQ(dim_permutations.insert(dim_ids).second, true);
        }
        ASSERT_EQ(instance_count, dim_permutations.size());

        // Iterate over all dimensions available for this metric. Ex: XCC, SE, SA, Instances.
        for(size_t i = 0; i < info.dimensions_count; i++)
        {
            // Current dimension size, like SE[0:6] has size of 7.
            size_t current_dimension_size = info.dimensions[i]->instance_size;
            ASSERT_NE(info.dimensions[i]->name, nullptr);
            std::string current_dimension_name(info.dimensions[i]->name);

            // Used to store index wise count (like number of instances with SE[0], SE[1], etc.) to
            // validate permutations included in dimensions_instances
            auto index_to_count = std::map<int, int>{};

            // Iterate over all instances with unique dimensions indexes.
            for(size_t j = 0; j < info.dimensions_instances_count; j++)
            {
                size_t dimension_size = info.dimensions_instances[j]->dimensions_count;

                for(size_t k = 0; k < dimension_size; k++)
                {
                    // Let's say for a metric has XCC[0:7], SE[0:6], Instances[0:10]
                    // Total Instances count: 8*7*11
                    if(std::string_view(
                           info.dimensions_instances[j]->dimensions[k]->dimension_name) ==
                       current_dimension_name)
                        index_to_count[info.dimensions_instances[j]->dimensions[k]->index]++;
                }
            }
            for(auto pr : index_to_count)
            {
                // Ex: Check there are (8*7*11)/8 counter samples with XCC=0.
                ASSERT_EQ(pr.second, info.dimensions_instances_count / current_dimension_size);
            }
            // Ex: Maximum index of XCC doesn't exceed or be equal to 8.
            ASSERT_EQ(index_to_count.rbegin()->first + 1, current_dimension_size);
        }
    }
}
