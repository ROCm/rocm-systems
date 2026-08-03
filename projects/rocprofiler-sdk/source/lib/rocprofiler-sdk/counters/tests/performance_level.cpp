// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../performance_level.hpp"

#include <gtest/gtest.h>

#include <unistd.h>
#include <fstream>
#include <string>

namespace rocprofiler
{
namespace counters
{
namespace
{
class PerformanceLevelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        root                  = common::filesystem::temp_directory_path() /
               ("rocprofiler-performance-level-" + std::to_string(::getpid()) + "-" +
                test_info->name());
        common::filesystem::remove_all(root);
    }

    void TearDown() override { common::filesystem::remove_all(root); }

    void write_level(uint32_t render_minor, const std::string& level)
    {
        auto path = get_counter_performance_level_path(render_minor, root);
        common::filesystem::create_directories(path.parent_path());
        auto output = std::ofstream{path.string()};
        ASSERT_TRUE(output.is_open());
        output << level << '\n';
    }

    common::filesystem::path root = {};
};

TEST(PerformanceLevel, ArchitectureRequirement)
{
    EXPECT_TRUE(requires_fixed_counter_performance_level("gfx1100"));
    EXPECT_TRUE(requires_fixed_counter_performance_level("gfx1151"));
    EXPECT_TRUE(requires_fixed_counter_performance_level("gfx1201"));
    EXPECT_FALSE(requires_fixed_counter_performance_level("gfx1030"));
    EXPECT_FALSE(requires_fixed_counter_performance_level("gfx942"));
    EXPECT_FALSE(requires_fixed_counter_performance_level(""));
}

TEST_F(PerformanceLevelTest, ReadsLevelThroughRenderNode)
{
    write_level(128, "profile_peak");

    auto result = read_counter_performance_level(128, root);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "profile_peak");
    EXPECT_FALSE(read_counter_performance_level(129, root).has_value());
}

TEST_F(PerformanceLevelTest, AutoIsRejectedForAffectedArchitecture)
{
    write_level(128, "auto");
    auto agent             = rocprofiler_agent_t{};
    agent.type             = ROCPROFILER_AGENT_TYPE_GPU;
    agent.name             = "gfx1100";
    agent.node_id          = 1;
    agent.drm_render_minor = 128;

    EXPECT_FALSE(check_agent_counter_performance_level(agent, root));
}

TEST_F(PerformanceLevelTest, FixedAndUnaffectedLevelsAreAccepted)
{
    write_level(128, "profile_standard");
    auto agent             = rocprofiler_agent_t{};
    agent.type             = ROCPROFILER_AGENT_TYPE_GPU;
    agent.name             = "gfx1201";
    agent.node_id          = 1;
    agent.drm_render_minor = 128;

    EXPECT_TRUE(check_agent_counter_performance_level(agent, root));

    write_level(128, "auto");
    agent.name = "gfx942";
    EXPECT_TRUE(check_agent_counter_performance_level(agent, root));
}

TEST_F(PerformanceLevelTest, MissingSysfsEntryDoesNotRejectCollection)
{
    auto agent             = rocprofiler_agent_t{};
    agent.type             = ROCPROFILER_AGENT_TYPE_GPU;
    agent.name             = "gfx1100";
    agent.node_id          = 1;
    agent.drm_render_minor = 128;

    EXPECT_TRUE(check_agent_counter_performance_level(agent, root));
}

}  // namespace
}  // namespace counters
}  // namespace rocprofiler
