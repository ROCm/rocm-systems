// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "gtest/gtest.h"

#include "library/components/cpu_load.hpp"
#include "library/cpu_load.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

namespace
{
void
set_env(const char* name, const char* value)
{
    if(value)
    {
        setenv(name, value, 1);
    }
    else
    {
        unsetenv(name);
    }
}

size_t
get_system_cpu_count()
{
    std::ifstream ifs("/proc/stat");
    if(!ifs.is_open()) return 0;

    size_t      count = 0;
    std::string line;
    while(std::getline(ifs, line))
    {
        if(line.size() >= 4 && line.substr(0, 3) == "cpu" && std::isdigit(line[3]))
        {
            count++;
        }
    }
    return count;
}

}  // anonymous namespace

class CpuLoadTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const char* orig = std::getenv("ROCPROFSYS_SAMPLING_CPUS");
        m_original_env     = orig ? std::string(orig) : std::string();
        m_had_env          = (orig != nullptr);
        m_system_cpu_count = get_system_cpu_count();
    }

    void TearDown() override
    {
        if(!m_had_env)
        {
            unsetenv("ROCPROFSYS_SAMPLING_CPUS");
        }
        else
        {
            setenv("ROCPROFSYS_SAMPLING_CPUS", m_original_env.c_str(), 1);
        }
    }

    std::string m_original_env;
    bool        m_had_env          = false;
    size_t      m_system_cpu_count = 0;
};

// ============================================================================
// Component Tests: CPU Enabling
// ============================================================================

TEST_F(CpuLoadTest, EnableAllCPUs)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "all");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), m_system_cpu_count);

    for(size_t i = 0; i < m_system_cpu_count; ++i)
    {
        EXPECT_TRUE(enabled.count(i) > 0) << "CPU " << i << " should be enabled";
    }
}

TEST_F(CpuLoadTest, EnableAllCPUsWithOnAlias)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "on");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), m_system_cpu_count);
}

TEST_F(CpuLoadTest, EnableNoCPUs)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "none");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 0);
}

TEST_F(CpuLoadTest, EnableNoCPUsWithOffAlias)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "off");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 0);
}

TEST_F(CpuLoadTest, EnableSpecificCPUs)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0,2,4");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 3);
    EXPECT_TRUE(enabled.count(0) > 0);
    EXPECT_TRUE(enabled.count(2) > 0);
    EXPECT_TRUE(enabled.count(4) > 0);
    EXPECT_FALSE(enabled.count(1) > 0);
    EXPECT_FALSE(enabled.count(3) > 0);
}

TEST_F(CpuLoadTest, EnableCPURange)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0-3");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 4);
    EXPECT_TRUE(enabled.count(0) > 0);
    EXPECT_TRUE(enabled.count(1) > 0);
    EXPECT_TRUE(enabled.count(2) > 0);
    EXPECT_TRUE(enabled.count(3) > 0);
    EXPECT_FALSE(enabled.count(4) > 0);
}

TEST_F(CpuLoadTest, EnableMixedCPUsAndRanges)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0,2-4,7");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 5);
    EXPECT_TRUE(enabled.count(0) > 0);
    EXPECT_TRUE(enabled.count(2) > 0);
    EXPECT_TRUE(enabled.count(3) > 0);
    EXPECT_TRUE(enabled.count(4) > 0);
    EXPECT_TRUE(enabled.count(7) > 0);
    EXPECT_FALSE(enabled.count(1) > 0);
    EXPECT_FALSE(enabled.count(5) > 0);
    EXPECT_FALSE(enabled.count(6) > 0);
}

TEST_F(CpuLoadTest, EnableCPUsWithSpaceDelimiter)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0 1 2");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 3);
    EXPECT_TRUE(enabled.count(0) > 0);
    EXPECT_TRUE(enabled.count(1) > 0);
    EXPECT_TRUE(enabled.count(2) > 0);
}

TEST_F(CpuLoadTest, EnableCPUsWithSemicolonDelimiter)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0;1;2");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 3);
    EXPECT_TRUE(enabled.count(0) > 0);
    EXPECT_TRUE(enabled.count(1) > 0);
    EXPECT_TRUE(enabled.count(2) > 0);
}

TEST_F(CpuLoadTest, EnableCPUsWithMixedDelimiters)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0,1 2;3");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 4);
    EXPECT_TRUE(enabled.count(0) > 0);
    EXPECT_TRUE(enabled.count(1) > 0);
    EXPECT_TRUE(enabled.count(2) > 0);
    EXPECT_TRUE(enabled.count(3) > 0);
}

// ============================================================================
// Component Tests: Edge Cases
// ============================================================================

TEST_F(CpuLoadTest, EnableInvalidCPU)
{
    std::string invalid_cpu = std::to_string(m_system_cpu_count + 10);
    set_env("ROCPROFSYS_SAMPLING_CPUS", invalid_cpu.c_str());
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 0);
}

TEST_F(CpuLoadTest, EnableValidAndInvalidCPUs)
{
    std::string mixed = "0," + std::to_string(m_system_cpu_count + 10) + ",1";
    set_env("ROCPROFSYS_SAMPLING_CPUS", mixed.c_str());
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 2);
    EXPECT_TRUE(enabled.count(0) > 0);
    EXPECT_TRUE(enabled.count(1) > 0);
}

TEST_F(CpuLoadTest, EnableRangePartiallyOutOfBounds)
{
    if(m_system_cpu_count < 2) GTEST_SKIP() << "Need at least 2 CPUs for this test";

    std::string range =
        std::to_string(m_system_cpu_count - 2) + "-" + std::to_string(m_system_cpu_count + 5);
    set_env("ROCPROFSYS_SAMPLING_CPUS", range.c_str());
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_TRUE(enabled.count(m_system_cpu_count - 2) > 0);
    EXPECT_TRUE(enabled.count(m_system_cpu_count - 1) > 0);
    EXPECT_FALSE(enabled.count(m_system_cpu_count) > 0);
}

TEST_F(CpuLoadTest, EmptyConfigurationDefaultsToAll)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", nullptr);
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), m_system_cpu_count);
}

TEST_F(CpuLoadTest, CaseInsensitiveKeywords)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "ALL");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), m_system_cpu_count);

    set_env("ROCPROFSYS_SAMPLING_CPUS", "NoNe");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled2 = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled2.size(), 0);
}

// ============================================================================
// Component Tests: Boundary Values
// ============================================================================

TEST_F(CpuLoadTest, EnableCPUZero)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 1);
    EXPECT_TRUE(enabled.count(0) > 0);
}

TEST_F(CpuLoadTest, EnableLastCPU)
{
    if(m_system_cpu_count == 0) GTEST_SKIP() << "No CPUs detected";

    std::string last_cpu = std::to_string(m_system_cpu_count - 1);
    set_env("ROCPROFSYS_SAMPLING_CPUS", last_cpu.c_str());
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 1);
    EXPECT_TRUE(enabled.count(m_system_cpu_count - 1) > 0);
}

TEST_F(CpuLoadTest, EnableSingleCPURange)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0-0");
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), 1);
    EXPECT_TRUE(enabled.count(0) > 0);
}

TEST_F(CpuLoadTest, EnableFullSystemRange)
{
    if(m_system_cpu_count == 0) GTEST_SKIP() << "No CPUs detected";

    std::string range = "0-" + std::to_string(m_system_cpu_count - 1);
    set_env("ROCPROFSYS_SAMPLING_CPUS", range.c_str());
    rocprofsys::component::cpu_load::configure();

    const auto& enabled = rocprofsys::component::cpu_load::get_enabled_cpus();

    EXPECT_EQ(enabled.size(), m_system_cpu_count);
}

// ============================================================================
// Component Tests: Load Calculation
// ============================================================================

TEST_F(CpuLoadTest, FirstSampleReturnsEmpty)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0,1,2");
    rocprofsys::component::cpu_load::configure();

    rocprofsys::component::cpu_load loader;
    loader.sample();

    const auto& loads = loader.get_loads();
    EXPECT_EQ(loads.size(), 0);
}

TEST_F(CpuLoadTest, SecondSampleReturnsLoadData)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0,1,2");
    rocprofsys::component::cpu_load::configure();

    rocprofsys::component::cpu_load loader;

    loader.sample();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    loader.sample();

    const auto& loads = loader.get_loads();

    EXPECT_EQ(loads.size(), 3);
    EXPECT_TRUE(loads.count(0) > 0);
    EXPECT_TRUE(loads.count(1) > 0);
    EXPECT_TRUE(loads.count(2) > 0);
}

TEST_F(CpuLoadTest, LoadPercentageInValidRange)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "all");
    rocprofsys::component::cpu_load::configure();

    rocprofsys::component::cpu_load loader;

    loader.sample();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    loader.sample();

    const auto& loads = loader.get_loads();

    for(const auto& [cpu_id, load_pct] : loads)
    {
        EXPECT_GE(load_pct, 0.0) << "CPU " << cpu_id << " load should be >= 0%";
        EXPECT_LE(load_pct, 100.0) << "CPU " << cpu_id << " load should be <= 100%";
    }
}

TEST_F(CpuLoadTest, GetLoadForSpecificCPU)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0,1");
    rocprofsys::component::cpu_load::configure();

    rocprofsys::component::cpu_load loader;

    loader.sample();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    loader.sample();

    double load0 = loader.get_load(0);
    double load1 = loader.get_load(1);

    EXPECT_GE(load0, 0.0);
    EXPECT_LE(load0, 100.0);
    EXPECT_GE(load1, 0.0);
    EXPECT_LE(load1, 100.0);
}

TEST_F(CpuLoadTest, GetLoadForNonEnabledCPUReturnsZero)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0");
    rocprofsys::component::cpu_load::configure();

    rocprofsys::component::cpu_load loader;

    loader.sample();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    loader.sample();

    double load1 = loader.get_load(1);

    EXPECT_EQ(load1, 0.0);
}

TEST_F(CpuLoadTest, MultipleSamplesUpdateLoad)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0");
    rocprofsys::component::cpu_load::configure();

    rocprofsys::component::cpu_load loader;

    loader.sample();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    loader.sample();
    double load1 = loader.get_load(0);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    loader.sample();
    double load2 = loader.get_load(0);

    EXPECT_GE(load1, 0.0);
    EXPECT_LE(load1, 100.0);
    EXPECT_GE(load2, 0.0);
    EXPECT_LE(load2, 100.0);
}

// ============================================================================
// Namespace Tests: Lifecycle Functions
// ============================================================================

TEST_F(CpuLoadTest, LifecycleFunctionsExecute)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0,1");

    rocprofsys::cpu_load::setup();
    rocprofsys::cpu_load::config();
    rocprofsys::cpu_load::sample();
    rocprofsys::cpu_load::sample();
    rocprofsys::cpu_load::post_process();
    rocprofsys::cpu_load::shutdown();
}

TEST_F(CpuLoadTest, MultipleSamplesViaNamespace)
{
    set_env("ROCPROFSYS_SAMPLING_CPUS", "0");

    rocprofsys::cpu_load::setup();
    rocprofsys::cpu_load::config();

    for(int i = 0; i < 3; ++i)
    {
        rocprofsys::cpu_load::sample();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    rocprofsys::cpu_load::post_process();
    rocprofsys::cpu_load::shutdown();

    SUCCEED();
}

// ============================================================================
// Component Tests: Static Metadata
// ============================================================================

TEST_F(CpuLoadTest, StaticMetadata)
{
    EXPECT_EQ(rocprofsys::component::cpu_load::label(), "cpu_load");
    EXPECT_EQ(rocprofsys::component::cpu_load::description(), "CPU load percentage");
    EXPECT_EQ(rocprofsys::component::cpu_load::unit(), "%");
    EXPECT_EQ(rocprofsys::component::cpu_load::display_unit(), "%");
}
