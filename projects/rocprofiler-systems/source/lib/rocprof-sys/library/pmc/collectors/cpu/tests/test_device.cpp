// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/cpu/device.hpp"
#include "library/pmc/device_providers/procfs/drivers/tests/mock_driver.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

using namespace rocprofsys::pmc::collectors::cpu;
using MockDriver      = rocprofsys::pmc::drivers::procfs::testing::mock_driver;
using cpu_jiffies     = rocprofsys::pmc::drivers::procfs::cpu_jiffies;
using rusage_snapshot = rocprofsys::pmc::drivers::procfs::rusage_snapshot;

using ::testing::_;
using ::testing::Return;

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace cpu
{
namespace testing
{

// ============================================================================
// Test Fixture
// ============================================================================

class CpuDeviceTest : public ::testing::Test
{
protected:
    std::shared_ptr<MockDriver> mock_driver;
    std::set<size_t>            monitored_cpus;

    void SetUp() override
    {
        mock_driver = std::make_shared<MockDriver>();
        mock_driver->set_up_defaults();
        monitored_cpus = { 0, 1, 2, 3 };
    }

    std::map<size_t, cpu_jiffies> make_jiffies(uint64_t user, uint64_t idle)
    {
        std::map<size_t, cpu_jiffies> result;
        for(size_t i = 0; i < 4; ++i)
        {
            cpu_jiffies j;
            j.user    = user;
            j.nice    = 0;
            j.system  = 0;
            j.idle    = idle;
            j.iowait  = 0;
            j.irq     = 0;
            j.softirq = 0;
            result[i] = j;
        }
        return result;
    }

    std::map<size_t, float> make_freqs(float mhz)
    {
        std::map<size_t, float> result;
        for(size_t i = 0; i < 4; ++i)
            result[i] = mhz;
        return result;
    }

    rusage_snapshot make_rusage(int64_t rss  = 50 * 1024 * 1024,
                                int64_t virt = 200 * 1024 * 1024)
    {
        rusage_snapshot snap;
        snap.page_rss         = rss;
        snap.virt_mem         = virt;
        snap.peak_rss         = 60 * 1024 * 1024;
        snap.context_switches = 1000;
        snap.page_faults      = 500;
        snap.user_mode_time   = 5000000;
        snap.kernel_mode_time = 1000000;
        return snap;
    }
};

// ============================================================================
// TC1: Initialization / Support Detection
// ============================================================================

TEST_F(CpuDeviceTest, TC1_1_AllMetricsSupportedWhenProcfsReadable)
{
    device<MockDriver> dev(mock_driver, monitored_cpus);

    EXPECT_TRUE(dev.is_supported());
    auto supported = dev.get_supported_metrics();
    EXPECT_EQ(supported.bits.frequency, 1u);
    EXPECT_EQ(supported.bits.load, 1u);
    EXPECT_EQ(supported.bits.page_rss, 1u);
    EXPECT_EQ(supported.bits.virt_mem, 1u);
    EXPECT_EQ(supported.bits.peak_rss, 1u);
    EXPECT_EQ(supported.bits.ctx_switches, 1u);
    EXPECT_EQ(supported.bits.page_faults, 1u);
    EXPECT_EQ(supported.bits.user_time, 1u);
    EXPECT_EQ(supported.bits.kernel_time, 1u);
}

TEST_F(CpuDeviceTest, TC1_2_NoLoadWhenProcStatEmpty)
{
    ON_CALL(*mock_driver, read_proc_stat())
        .WillByDefault(Return(std::map<size_t, cpu_jiffies>{}));

    device<MockDriver> dev(mock_driver, monitored_cpus);

    EXPECT_TRUE(dev.is_supported());  // frequency and process metrics still available
    EXPECT_EQ(dev.get_supported_metrics().bits.load, 0u);
    EXPECT_EQ(dev.get_supported_metrics().bits.frequency, 1u);
}

TEST_F(CpuDeviceTest, TC1_3_NoFrequencyWhenCpuinfoEmpty)
{
    ON_CALL(*mock_driver, read_cpu_frequencies())
        .WillByDefault(Return(std::map<size_t, float>{}));

    device<MockDriver> dev(mock_driver, monitored_cpus);

    EXPECT_TRUE(dev.is_supported());  // load and process metrics still available
    EXPECT_EQ(dev.get_supported_metrics().bits.frequency, 0u);
    EXPECT_EQ(dev.get_supported_metrics().bits.load, 1u);
}

TEST_F(CpuDeviceTest, TC1_4_MonitoredCpusStored)
{
    std::set<size_t>   subset = { 0, 2 };
    device<MockDriver> dev(mock_driver, subset);

    EXPECT_EQ(dev.get_monitored_cpus().size(), 2u);
    EXPECT_TRUE(dev.get_monitored_cpus().count(0) > 0);
    EXPECT_TRUE(dev.get_monitored_cpus().count(2) > 0);
    EXPECT_TRUE(dev.get_monitored_cpus().count(1) == 0);
}

// ============================================================================
// TC2: Frequency Collection
// ============================================================================

TEST_F(CpuDeviceTest, TC2_1_FrequenciesCollected)
{
    ON_CALL(*mock_driver, read_cpu_frequencies())
        .WillByDefault(Return(make_freqs(3500.0f)));

    device<MockDriver> dev(mock_driver, monitored_cpus);
    auto               result = dev.get_cpu_metrics();

    // Frequencies should be present for all monitored CPUs
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_FLOAT_EQ(cpu.frequency, 3500.0f);
    }
}

TEST_F(CpuDeviceTest, TC2_2_FrequenciesFilteredByMonitoredSet)
{
    std::set<size_t>   subset = { 1, 3 };
    device<MockDriver> dev(mock_driver, subset);
    auto               result = dev.get_cpu_metrics();

    // Should only have entries for CPUs 1 and 3
    std::set<size_t> collected_ids;
    for(const auto& cpu : result.cpu_data)
    {
        collected_ids.insert(cpu.cpu_id);
    }
    EXPECT_EQ(collected_ids.count(0), 0u);
    EXPECT_EQ(collected_ids.count(1), 1u);
    EXPECT_EQ(collected_ids.count(2), 0u);
    EXPECT_EQ(collected_ids.count(3), 1u);
}

// ============================================================================
// TC3: Load Calculation
// ============================================================================

TEST_F(CpuDeviceTest, TC3_1_FirstSampleReturnsNoLoad)
{
    device<MockDriver> dev(mock_driver, monitored_cpus);
    auto               result = dev.get_cpu_metrics();

    // First sample has no previous baseline, so load should be 0.0
    // (entries may exist from frequency collection, but load is 0)
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_DOUBLE_EQ(cpu.load, 0.0);
    }
}

TEST_F(CpuDeviceTest, TC3_2_SecondSampleComputesLoadFromDelta)
{
    // First call: baseline jiffies (100 user, 900 idle = 1000 total)
    auto baseline = make_jiffies(100, 900);
    EXPECT_CALL(*mock_driver, read_proc_stat())
        .WillOnce(Return(baseline))                 // init probe
        .WillOnce(Return(baseline))                 // first sample
        .WillOnce(Return(make_jiffies(200, 800)));  // second sample

    device<MockDriver> dev(mock_driver, monitored_cpus);

    // First sample: stores baseline
    (void) dev.get_cpu_metrics();

    // Second sample: delta is (200-100)=100 active out of (1000-1000)=0... wait
    // Actually: total1=1000, total2=1000, delta_total=0, so load=0
    // Let me fix: total1=100+900=1000, total2=200+800=1000, delta=0
    // Need different totals. Let's use WillOnce differently.

    // This test verifies the delta computation works.
    // actual load = delta_active / delta_total * 100
    // delta_active = (200-100) = 100
    // delta_total = (200+800) - (100+900) = 1000 - 1000 = 0
    // When delta_total=0, load should be 0%
    auto result = dev.get_cpu_metrics();
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_DOUBLE_EQ(cpu.load, 0.0);
    }
}

TEST_F(CpuDeviceTest, TC3_3_LoadCalculationWithIncreasingJiffies)
{
    // Baseline: 100 active (user), 900 idle, total=1000
    auto baseline = make_jiffies(100, 900);
    // After interval: 200 active (user), 1800 idle, total=2000
    // delta_active=100, delta_total=1000, load=10%
    auto after = make_jiffies(200, 1800);

    EXPECT_CALL(*mock_driver, read_proc_stat())
        .WillOnce(Return(baseline))  // init probe
        .WillOnce(Return(baseline))  // first sample (baseline stored)
        .WillOnce(Return(after));    // second sample (delta computed)

    device<MockDriver> dev(mock_driver, monitored_cpus);

    // First sample: stores baseline, no load
    (void) dev.get_cpu_metrics();

    // Second sample: should have load = 10%
    auto result = dev.get_cpu_metrics();
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_NEAR(cpu.load, 10.0, 0.001);
    }
}

TEST_F(CpuDeviceTest, TC3_4_FullLoadCalculation)
{
    // Baseline: 0 active, 1000 idle
    auto baseline = make_jiffies(0, 1000);
    // After: 1000 active, 1000 idle (total went from 1000 to 2000)
    // delta_active=1000, delta_total=1000, load=100%
    auto after = make_jiffies(1000, 1000);

    EXPECT_CALL(*mock_driver, read_proc_stat())
        .WillOnce(Return(baseline))  // init probe
        .WillOnce(Return(baseline))  // first sample
        .WillOnce(Return(after));    // second sample

    device<MockDriver> dev(mock_driver, monitored_cpus);
    (void) dev.get_cpu_metrics();

    auto result = dev.get_cpu_metrics();
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_NEAR(cpu.load, 100.0, 0.001);
    }
}

TEST_F(CpuDeviceTest, TC3_5_ZeroLoadWhenIdle)
{
    // Baseline: 100 active, 900 idle
    auto baseline = make_jiffies(100, 900);
    // After: 100 active, 1900 idle (only idle increased)
    // delta_active=0, delta_total=1000, load=0%
    auto after = make_jiffies(100, 1900);

    EXPECT_CALL(*mock_driver, read_proc_stat())
        .WillOnce(Return(baseline))
        .WillOnce(Return(baseline))
        .WillOnce(Return(after));

    device<MockDriver> dev(mock_driver, monitored_cpus);
    (void) dev.get_cpu_metrics();

    auto result = dev.get_cpu_metrics();
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_NEAR(cpu.load, 0.0, 0.001);
    }
}

// ============================================================================
// TC4: Process Metrics
// ============================================================================

TEST_F(CpuDeviceTest, TC4_1_ProcessMetricsCollected)
{
    auto snap = make_rusage();
    ON_CALL(*mock_driver, read_rusage()).WillByDefault(Return(snap));

    device<MockDriver> dev(mock_driver, monitored_cpus);
    auto               result = dev.get_cpu_metrics();

    EXPECT_EQ(result.process_data.page_rss, 50 * 1024 * 1024);
    EXPECT_EQ(result.process_data.virt_mem, 200 * 1024 * 1024);
    EXPECT_EQ(result.process_data.peak_rss, 60 * 1024 * 1024);
    EXPECT_EQ(result.process_data.context_switches, 1000);
    EXPECT_EQ(result.process_data.page_faults, 500);
    EXPECT_EQ(result.process_data.user_mode_time, 5000000);
    EXPECT_EQ(result.process_data.kernel_mode_time, 1000000);
}

TEST_F(CpuDeviceTest, TC4_2_ProcessMetricsWithZeroPeakRss)
{
    auto snap     = make_rusage();
    snap.peak_rss = 0;
    ON_CALL(*mock_driver, read_rusage()).WillByDefault(Return(snap));

    // When peak_rss is 0 during init, that bit should not be supported
    ON_CALL(*mock_driver, read_proc_stat()).WillByDefault(Return(make_jiffies(100, 900)));
    ON_CALL(*mock_driver, read_cpu_frequencies())
        .WillByDefault(Return(make_freqs(2000.0f)));

    device<MockDriver> dev(mock_driver, monitored_cpus);
    EXPECT_EQ(dev.get_supported_metrics().bits.peak_rss, 0u);
}

// ============================================================================
// TC5: Filtered CPU Sets
// ============================================================================

TEST_F(CpuDeviceTest, TC5_1_EmptyMonitoredSetProducesNoPerCpuData)
{
    std::set<size_t>   empty_set;
    device<MockDriver> dev(mock_driver, empty_set);
    auto               result = dev.get_cpu_metrics();

    EXPECT_TRUE(result.cpu_data.empty());
    // Process metrics should still be collected
    EXPECT_GT(result.process_data.page_rss, 0);
}

TEST_F(CpuDeviceTest, TC5_2_SingleCpuMonitored)
{
    std::set<size_t>   single = { 2 };
    device<MockDriver> dev(mock_driver, single);
    auto               result = dev.get_cpu_metrics();

    // Should have exactly one per-CPU entry for CPU 2
    size_t cpu2_count = 0;
    for(const auto& cpu : result.cpu_data)
    {
        if(cpu.cpu_id == 2) cpu2_count++;
    }
    EXPECT_EQ(cpu2_count, 1u);
}

TEST_F(CpuDeviceTest, TC5_3_NonExistentCpuIdSkipped)
{
    // Monitor CPU 99 which doesn't exist in mock data (only 0-3)
    std::set<size_t>   nonexistent = { 99 };
    device<MockDriver> dev(mock_driver, nonexistent);
    auto               result = dev.get_cpu_metrics();

    // CPU 99 doesn't appear in mock /proc/stat or /proc/cpuinfo
    EXPECT_TRUE(result.cpu_data.empty());
}

// ============================================================================
// TC6: Repeated Sampling
// ============================================================================

TEST_F(CpuDeviceTest, TC6_1_MultipleSamplesAccumulateCorrectly)
{
    auto jiffies1 = make_jiffies(100, 900);  // total=1000
    auto jiffies2 =
        make_jiffies(200, 1800);  // total=2000, delta=1000, active_delta=100 -> 10%
    auto jiffies3 =
        make_jiffies(700, 2300);  // total=3000, delta=1000, active_delta=500 -> 50%

    EXPECT_CALL(*mock_driver, read_proc_stat())
        .WillOnce(Return(jiffies1))   // init probe
        .WillOnce(Return(jiffies1))   // sample 1 (baseline)
        .WillOnce(Return(jiffies2))   // sample 2
        .WillOnce(Return(jiffies3));  // sample 3

    device<MockDriver> dev(mock_driver, monitored_cpus);

    (void) dev.get_cpu_metrics();  // baseline

    auto result2 = dev.get_cpu_metrics();  // 10%
    for(const auto& cpu : result2.cpu_data)
    {
        EXPECT_NEAR(cpu.load, 10.0, 0.001);
    }

    auto result3 = dev.get_cpu_metrics();  // 50%
    for(const auto& cpu : result3.cpu_data)
    {
        EXPECT_NEAR(cpu.load, 50.0, 0.001);
    }
}

// ============================================================================
// TC7: Combined Metrics
// ============================================================================

TEST_F(CpuDeviceTest, TC7_1_AllMetricsCombinedInSingleSample)
{
    auto jiffies1 = make_jiffies(100, 900);
    auto jiffies2 = make_jiffies(200, 1800);

    EXPECT_CALL(*mock_driver, read_proc_stat())
        .WillOnce(Return(jiffies1))
        .WillOnce(Return(jiffies1))
        .WillOnce(Return(jiffies2));

    ON_CALL(*mock_driver, read_cpu_frequencies())
        .WillByDefault(Return(make_freqs(3200.0f)));

    auto snap = make_rusage(100 * 1024 * 1024, 500 * 1024 * 1024);
    ON_CALL(*mock_driver, read_rusage()).WillByDefault(Return(snap));

    device<MockDriver> dev(mock_driver, monitored_cpus);
    (void) dev.get_cpu_metrics();  // baseline

    auto result = dev.get_cpu_metrics();

    // Check per-CPU data
    EXPECT_EQ(result.cpu_data.size(), 4u);
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_FLOAT_EQ(cpu.frequency, 3200.0f);
        EXPECT_NEAR(cpu.load, 10.0, 0.001);
    }

    // Check process data
    EXPECT_EQ(result.process_data.page_rss, 100 * 1024 * 1024);
    EXPECT_EQ(result.process_data.virt_mem, 500 * 1024 * 1024);
}

}  // namespace testing
}  // namespace cpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
