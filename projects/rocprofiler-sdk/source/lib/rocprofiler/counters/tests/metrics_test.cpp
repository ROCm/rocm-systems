#include "metrics_test.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "lib/rocprofiler/counters/metrics.hpp"

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
    EXPECT_EQ(rocp_data_v.size(), test_data_v.size());
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
