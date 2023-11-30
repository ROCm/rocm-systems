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

#include "metrics_test.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"

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
    auto rocp_data = counters::getBaseHardwareMetrics();
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
            LOG(ERROR) << fmt::format("{}", ditr);
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
    auto rocp_data = counters::getDerivedHardwareMetrics();
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
