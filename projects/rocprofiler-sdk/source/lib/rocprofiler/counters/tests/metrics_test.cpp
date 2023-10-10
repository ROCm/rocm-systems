#include "metrics_test.h"

#include <gtest/gtest.h>

#include "lib/rocprofiler/counters/metrics.hpp"

namespace
{
auto
loadTestData(const std::unordered_map<std::string, std::vector<std::vector<std::string>>>& map)
{
    std::unordered_map<std::string, std::vector<counters::Metric>> ret;
    for(const auto& [gfx, dataMap] : map)
    {
        auto& metric_vec = ret.emplace(gfx, std::vector<counters::Metric>{}).first->second;
        for(const auto& data_vec : dataMap)
        {
            metric_vec.emplace_back(
                data_vec.at(0), data_vec.at(1), data_vec.at(2), data_vec.at(4), data_vec.at(3));
        }
    }
    return ret;
}
}  // namespace

TEST(metrics, base_load)
{
    auto x         = counters::getBaseHardwareMetrics();
    auto test_data = loadTestData(basic_gfx908);
    ASSERT_EQ(x.count("gfx908"), 1);
    ASSERT_EQ(test_data.count("gfx908"), 1);
    EXPECT_EQ(fmt::format("{}", x["gfx908"]), fmt::format("{}", test_data["gfx908"]));
}

TEST(metrics, derived_load)
{
    auto x         = counters::getDerivedHardwareMetrics();
    auto test_data = loadTestData(derrived_gfx908);
    ASSERT_EQ(x.count("gfx908"), 1);
    ASSERT_EQ(test_data.count("gfx908"), 1);
    EXPECT_EQ(fmt::format("{}", x["gfx908"]), fmt::format("{}", test_data["gfx908"]));
}
