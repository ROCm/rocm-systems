#include "metrics_test.h"

#include <gtest/gtest.h>

#include "lib/rocprofiler/counters/metrics.hpp"

namespace
{
auto
loadTestData(std::unordered_map<std::string, std::vector<std::vector<std::string>>> map)
{
    std::unordered_map<std::string, std::vector<counters::Metric>> ret;
    for(auto& [gfx, dataMap] : map)
    {
        auto& metric_vec = ret.emplace(gfx, std::vector<counters::Metric>{}).first->second;
        for(auto& data_vec : dataMap)
        {
            metric_vec.emplace_back(
                data_vec.at(0), data_vec.at(1), data_vec.at(2), data_vec.at(4), data_vec.at(3));
        }
    }
    return ret;
}
}  // namespace

TEST(MetricsTest, BaseMetricLoad)
{
    auto x         = counters::getBaseHardwareMetrics();
    auto test_data = loadTestData(basic_gfx908);
    ASSERT_EQ(x.count("gfx908"), 1);
    ASSERT_EQ(test_data.count("gfx908"), 1);
    EXPECT_EQ(fmt::format("{}", x["gfx908"]), fmt::format("{}", test_data["gfx908"]));
}

TEST(MetricsTest, DerrivedMetricLoad)
{
    auto x         = counters::getDerrivedHardwareMetrics();
    auto test_data = loadTestData(derrived_gfx908);
    ASSERT_EQ(x.count("gfx908"), 1);
    ASSERT_EQ(test_data.count("gfx908"), 1);
    EXPECT_EQ(fmt::format("{}", x["gfx908"]), fmt::format("{}", test_data["gfx908"]));
}