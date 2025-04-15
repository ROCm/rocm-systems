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

        auto dim_ptr = rocprofiler::counters::get_dimension_cache();

        const auto* dims = rocprofiler::common::get_val(dim_ptr->id_to_dim, metric.id());
        ASSERT_TRUE(dims);

        auto status = rocprofiler_query_counter_info(
            {.handle = id}, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info));
        ASSERT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(std::string(info.name ? info.name : ""), metric.name());
        EXPECT_EQ(std::string(info.block ? info.block : ""), metric.block());
        EXPECT_EQ(std::string(info.expression ? info.expression : ""), metric.expression());
        EXPECT_EQ(info.is_derived, !metric.expression().empty());
        EXPECT_EQ(std::string(info.description ? info.description : ""), metric.description());

        EXPECT_EQ(info.dimensions_count, dims->size());
        for(size_t i = 0; i < info.dimensions_count; i++)
        {
            const auto& dim = dims->at(i);
            EXPECT_EQ(dim.size(), info.dimensions[i].instance_size);
            EXPECT_EQ(dim.type(), info.dimensions[i].id);
            EXPECT_EQ(std::string(info.dimensions[i].name), dim.name());
        }

        size_t instance_count = 0;
        // Checks the equality with the old rocprofiler_query_counter_instance_count
        for(const auto& metric_dim : *dims)
        {
            if(instance_count == 0)
                instance_count = metric_dim.size();
            else if(metric_dim.size() > 0)
                instance_count = metric_dim.size() * instance_count;
        }

        EXPECT_EQ(info.instance_ids_count, instance_count);
        std::set<std::vector<size_t>> dim_permutations;

        for(size_t i = 0; i < info.instance_ids_count; i++)
        {
            std::vector<size_t> dim_ids;
            ASSERT_EQ(rocprofiler::counters::rec_to_counter_id(info.instance_ids[i]).handle,
                      metric.id());
            for(const auto& metric_dim : *dims)
            {
                dim_ids.push_back(
                    rocprofiler::counters::rec_to_dim_pos(info.instance_ids[i], metric_dim.type()));
            }
            // Ensure that the premutation is unique
            ASSERT_EQ(dim_permutations.insert(dim_ids).second, true);
        }
        ASSERT_EQ(instance_count, dim_permutations.size());
    }
}
