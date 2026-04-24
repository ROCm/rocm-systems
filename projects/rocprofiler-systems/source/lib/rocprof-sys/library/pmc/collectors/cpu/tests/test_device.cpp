// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "library/pmc/collectors/cpu/device.hpp"
#include "library/pmc/device_providers/procfs/drivers/tests/mock_driver.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

using namespace rocprofsys::pmc::collectors::cpu;
using MockDriver      = rocprofsys::pmc::drivers::procfs::testing::strict_mock_driver;
using cpu_jiffies     = rocprofsys::pmc::drivers::procfs::cpu_jiffies;
using rusage_snapshot = rocprofsys::pmc::drivers::procfs::rusage_snapshot;

using ::testing::_;
using ::testing::Return;

namespace rocprofsys::pmc::collectors::cpu::testing
{
class cpu_device_test : public ::testing::Test
{
protected:
    std::shared_ptr<MockDriver> mock_driver;
    std::set<size_t>            monitored_cpus;
    enabled_metrics             all_enabled{};

    void SetUp() override
    {
        mock_driver = std::make_shared<MockDriver>();
        mock_driver->set_up_defaults();
        monitored_cpus    = { 0, 1, 2, 3 };
        all_enabled.value = ALL_CPU_METRICS;
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

TEST_F(cpu_device_test, all_metrics_supported_when_procfs_readable)
{
    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(1);        // init probe
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(1);  // init probe
    EXPECT_CALL(*mock_driver, read_rusage()).Times(1);           // init probe

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);

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

TEST_F(cpu_device_test, no_load_when_proc_stat_empty)
{
    ON_CALL(*mock_driver, read_proc_stat())
        .WillByDefault(Return(std::map<size_t, cpu_jiffies>{}));

    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(1);        // init probe
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(1);  // init probe
    EXPECT_CALL(*mock_driver, read_rusage()).Times(1);           // init probe

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);

    EXPECT_TRUE(dev.is_supported());
    EXPECT_EQ(dev.get_supported_metrics().bits.load, 0u);
    EXPECT_EQ(dev.get_supported_metrics().bits.frequency, 1u);
}

TEST_F(cpu_device_test, no_frequency_when_cpuinfo_empty)
{
    ON_CALL(*mock_driver, read_cpu_frequencies())
        .WillByDefault(Return(std::map<size_t, float>{}));

    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(1);        // init probe
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(1);  // init probe
    EXPECT_CALL(*mock_driver, read_rusage()).Times(1);           // init probe

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);

    EXPECT_TRUE(dev.is_supported());
    EXPECT_EQ(dev.get_supported_metrics().bits.frequency, 0u);
    EXPECT_EQ(dev.get_supported_metrics().bits.load, 1u);
}

TEST_F(cpu_device_test, device_interface_methods)
{
    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(1);        // init probe
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(1);  // init probe
    EXPECT_CALL(*mock_driver, read_rusage()).Times(1);           // init probe

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);

    EXPECT_EQ(dev.get_index(), 0u);
    EXPECT_EQ(dev.get_name(), "CPU 0");
    EXPECT_FALSE(dev.get_product_name().empty());
    EXPECT_FALSE(dev.get_vendor_name().empty());
    EXPECT_EQ(dev.get_monitored_cpus().size(), 4u);
}

TEST_F(cpu_device_test, frequencies_collected)
{
    ON_CALL(*mock_driver, read_cpu_frequencies())
        .WillByDefault(Return(make_freqs(3500.0f)));

    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(2);        // init + 1 sample
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(2);  // init + 1 sample
    EXPECT_CALL(*mock_driver, read_rusage()).Times(2);           // init + 1 sample

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    auto               result = dev.get_cpu_metrics(all_enabled);

    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_FLOAT_EQ(cpu.frequency, 3500.0f);
    }
}

TEST_F(cpu_device_test, frequencies_filtered_by_monitored_set)
{
    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(2);        // init + 1 sample
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(2);  // init + 1 sample
    EXPECT_CALL(*mock_driver, read_rusage()).Times(2);           // init + 1 sample

    std::set<size_t>   subset = { 1, 3 };
    device<MockDriver> dev(mock_driver, 0, subset);
    auto               result = dev.get_cpu_metrics(all_enabled);

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

TEST_F(cpu_device_test, first_sample_returns_zero_load)
{
    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(2);        // init + 1 sample
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(2);  // init + 1 sample
    EXPECT_CALL(*mock_driver, read_rusage()).Times(2);           // init + 1 sample

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    auto               result = dev.get_cpu_metrics(all_enabled);

    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_DOUBLE_EQ(cpu.load, 0.0);
    }
}

TEST_F(cpu_device_test, load_calculation_with_increasing_jiffies)
{
    // Baseline: 100 active (user), 900 idle, total=1000
    auto baseline = make_jiffies(100, 900);
    // After: 200 active (user), 1800 idle, total=2000
    // delta_active=100, delta_total=1000, load=10%
    auto after = make_jiffies(200, 1800);

    EXPECT_CALL(*mock_driver, read_proc_stat())
        .WillOnce(Return(baseline))  // init probe
        .WillOnce(Return(baseline))  // first sample (baseline stored)
        .WillOnce(Return(after));    // second sample (delta computed)
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(3);  // init + 2 samples
    EXPECT_CALL(*mock_driver, read_rusage()).Times(3);           // init + 2 samples

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    (void) dev.get_cpu_metrics(all_enabled);  // baseline

    auto result = dev.get_cpu_metrics(all_enabled);
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_NEAR(cpu.load, 10.0, 0.001);
    }
}

TEST_F(cpu_device_test, full_load_calculation)
{
    auto baseline = make_jiffies(0, 1000);
    // delta_active=1000, delta_total=1000, load=100%
    auto after = make_jiffies(1000, 1000);

    EXPECT_CALL(*mock_driver, read_proc_stat())
        .WillOnce(Return(baseline))
        .WillOnce(Return(baseline))
        .WillOnce(Return(after));
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(3);  // init + 2 samples
    EXPECT_CALL(*mock_driver, read_rusage()).Times(3);           // init + 2 samples

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    (void) dev.get_cpu_metrics(all_enabled);

    auto result = dev.get_cpu_metrics(all_enabled);
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_NEAR(cpu.load, 100.0, 0.001);
    }
}

TEST_F(cpu_device_test, zero_load_when_idle)
{
    auto baseline = make_jiffies(100, 900);
    // Only idle increased: delta_active=0, delta_total=1000, load=0%
    auto after = make_jiffies(100, 1900);

    EXPECT_CALL(*mock_driver, read_proc_stat())
        .WillOnce(Return(baseline))
        .WillOnce(Return(baseline))
        .WillOnce(Return(after));
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(3);  // init + 2 samples
    EXPECT_CALL(*mock_driver, read_rusage()).Times(3);           // init + 2 samples

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    (void) dev.get_cpu_metrics(all_enabled);

    auto result = dev.get_cpu_metrics(all_enabled);
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_NEAR(cpu.load, 0.0, 0.001);
    }
}

TEST_F(cpu_device_test, process_metrics_collected)
{
    auto snap = make_rusage();
    ON_CALL(*mock_driver, read_rusage()).WillByDefault(Return(snap));

    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(2);        // init + 1 sample
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(2);  // init + 1 sample
    EXPECT_CALL(*mock_driver, read_rusage()).Times(2);           // init + 1 sample

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    auto               result = dev.get_cpu_metrics(all_enabled);

    EXPECT_EQ(result.process_data.page_rss, 50 * 1024 * 1024);
    EXPECT_EQ(result.process_data.virt_mem, 200 * 1024 * 1024);
    EXPECT_EQ(result.process_data.peak_rss, 60 * 1024 * 1024);
    EXPECT_EQ(result.process_data.context_switches, 1000);
    EXPECT_EQ(result.process_data.page_faults, 500);
    EXPECT_EQ(result.process_data.user_mode_time, 5000000);
    EXPECT_EQ(result.process_data.kernel_mode_time, 1000000);
}

TEST_F(cpu_device_test, zero_peak_rss_marks_unsupported)
{
    auto snap     = make_rusage();
    snap.peak_rss = 0;
    ON_CALL(*mock_driver, read_rusage()).WillByDefault(Return(snap));

    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(1);        // init probe
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(1);  // init probe
    EXPECT_CALL(*mock_driver, read_rusage()).Times(1);           // init probe

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    EXPECT_EQ(dev.get_supported_metrics().bits.peak_rss, 0u);
}

TEST_F(cpu_device_test, empty_monitored_set_produces_no_per_cpu_data)
{
    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(2);        // init + 1 sample
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(2);  // init + 1 sample
    EXPECT_CALL(*mock_driver, read_rusage()).Times(2);           // init + 1 sample

    std::set<size_t>   empty_set;
    device<MockDriver> dev(mock_driver, 0, empty_set);
    auto               result = dev.get_cpu_metrics(all_enabled);

    EXPECT_TRUE(result.cpu_data.empty());
    EXPECT_GT(result.process_data.page_rss, 0);
}

TEST_F(cpu_device_test, single_cpu_monitored)
{
    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(2);        // init + 1 sample
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(2);  // init + 1 sample
    EXPECT_CALL(*mock_driver, read_rusage()).Times(2);           // init + 1 sample

    std::set<size_t>   single = { 2 };
    device<MockDriver> dev(mock_driver, 0, single);
    auto               result = dev.get_cpu_metrics(all_enabled);

    size_t cpu2_count = 0;
    for(const auto& cpu : result.cpu_data)
    {
        if(cpu.cpu_id == 2) cpu2_count++;
    }
    EXPECT_EQ(cpu2_count, 1u);
}

TEST_F(cpu_device_test, nonexistent_cpu_id_has_zero_metrics)
{
    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(2);        // init + 1 sample
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(2);  // init + 1 sample
    EXPECT_CALL(*mock_driver, read_rusage()).Times(2);           // init + 1 sample

    std::set<size_t>   nonexistent = { 99 };
    device<MockDriver> dev(mock_driver, 0, nonexistent);
    auto               result = dev.get_cpu_metrics(all_enabled);

    // make_empty_metrics pre-populates zero entries for all monitored CPUs,
    // even if they don't appear in /proc/stat. This ensures Perfetto tracks
    // show zero rather than retaining stale values.
    ASSERT_EQ(result.cpu_data.size(), 1u);
    EXPECT_EQ(result.cpu_data[0].cpu_id, 99u);
    EXPECT_DOUBLE_EQ(result.cpu_data[0].load, 0.0);
    EXPECT_FLOAT_EQ(result.cpu_data[0].frequency, 0.0f);
}

TEST_F(cpu_device_test, multiple_samples_accumulate_correctly)
{
    auto jiffies1 = make_jiffies(100, 900);  // total=1000
    auto jiffies2 =
        make_jiffies(200, 1800);  // total=2000, delta=1000, active_delta=100 -> 10%
    auto jiffies3 =
        make_jiffies(700, 2300);  // total=3000, delta=1000, active_delta=500 -> 50%

    EXPECT_CALL(*mock_driver, read_proc_stat())
        .WillOnce(Return(jiffies1))                              // init probe
        .WillOnce(Return(jiffies1))                              // sample 1 (baseline)
        .WillOnce(Return(jiffies2))                              // sample 2
        .WillOnce(Return(jiffies3));                             // sample 3
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(4);  // init + 3 samples
    EXPECT_CALL(*mock_driver, read_rusage()).Times(4);           // init + 3 samples

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);

    (void) dev.get_cpu_metrics(all_enabled);  // baseline

    auto result2 = dev.get_cpu_metrics(all_enabled);  // 10%
    for(const auto& cpu : result2.cpu_data)
    {
        EXPECT_NEAR(cpu.load, 10.0, 0.001);
    }

    auto result3 = dev.get_cpu_metrics(all_enabled);  // 50%
    for(const auto& cpu : result3.cpu_data)
    {
        EXPECT_NEAR(cpu.load, 50.0, 0.001);
    }
}

TEST_F(cpu_device_test, all_metrics_combined_in_single_sample)
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

    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(3);  // init + 2 samples
    EXPECT_CALL(*mock_driver, read_rusage()).Times(3);           // init + 2 samples

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    (void) dev.get_cpu_metrics(all_enabled);  // baseline

    auto result = dev.get_cpu_metrics(all_enabled);

    EXPECT_EQ(result.cpu_data.size(), 4u);
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_FLOAT_EQ(cpu.frequency, 3200.0f);
        EXPECT_NEAR(cpu.load, 10.0, 0.001);
    }

    EXPECT_EQ(result.process_data.page_rss, 100 * 1024 * 1024);
    EXPECT_EQ(result.process_data.virt_mem, 500 * 1024 * 1024);
}

TEST_F(cpu_device_test, only_frequency_enabled_skips_load_and_process)
{
    enabled_metrics freq_only{};
    freq_only.bits.frequency = 1;

    ON_CALL(*mock_driver, read_cpu_frequencies())
        .WillByDefault(Return(make_freqs(2400.0f)));

    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(2);  // init + 1 sample

    // read_proc_stat should NOT be called when load is disabled
    EXPECT_CALL(*mock_driver, read_proc_stat())
        .Times(1);  // only the init probe in constructor

    // read_rusage should NOT be called when all process metrics are disabled
    EXPECT_CALL(*mock_driver, read_rusage())
        .Times(1);  // only the init probe in constructor

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    auto               result = dev.get_cpu_metrics(freq_only);

    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_FLOAT_EQ(cpu.frequency, 2400.0f);
        EXPECT_DOUBLE_EQ(cpu.load, 0.0);
    }
    EXPECT_EQ(result.process_data.page_rss, 0);
    EXPECT_EQ(result.process_data.virt_mem, 0);
}

TEST_F(cpu_device_test, only_load_enabled_skips_frequency_and_process)
{
    enabled_metrics load_only{};
    load_only.bits.load = 1;

    auto baseline = make_jiffies(100, 900);
    auto after    = make_jiffies(200, 1800);

    EXPECT_CALL(*mock_driver, read_proc_stat())
        .WillOnce(Return(baseline))  // init probe
        .WillOnce(Return(baseline))  // first sample (baseline)
        .WillOnce(Return(after));    // second sample

    EXPECT_CALL(*mock_driver, read_cpu_frequencies())
        .Times(1);  // only the init probe in constructor

    EXPECT_CALL(*mock_driver, read_rusage())
        .Times(1);  // only the init probe in constructor

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    (void) dev.get_cpu_metrics(load_only);  // baseline

    auto result = dev.get_cpu_metrics(load_only);

    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_NEAR(cpu.load, 10.0, 0.001);
        EXPECT_FLOAT_EQ(cpu.frequency, 0.0f);
    }
    EXPECT_EQ(result.process_data.page_rss, 0);
}

TEST_F(cpu_device_test, only_process_metrics_enabled)
{
    enabled_metrics process_only{};
    process_only.bits.page_rss  = 1;
    process_only.bits.peak_rss  = 1;
    process_only.bits.user_time = 1;

    auto snap = make_rusage();
    EXPECT_CALL(*mock_driver, read_rusage())
        .WillOnce(Return(snap))   // init probe
        .WillOnce(Return(snap));  // sample

    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(1);  // only the init probe

    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(1);  // only the init probe

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    auto               result = dev.get_cpu_metrics(process_only);

    EXPECT_EQ(result.process_data.page_rss, 50 * 1024 * 1024);
    EXPECT_EQ(result.process_data.peak_rss, 60 * 1024 * 1024);
    EXPECT_EQ(result.process_data.user_mode_time, 5000000);
    // Disabled process metrics should remain zero
    EXPECT_EQ(result.process_data.virt_mem, 0);
    EXPECT_EQ(result.process_data.context_switches, 0);
    EXPECT_EQ(result.process_data.kernel_mode_time, 0);
}

TEST_F(cpu_device_test, no_metrics_enabled_skips_all_reads)
{
    enabled_metrics none{};

    EXPECT_CALL(*mock_driver, read_proc_stat()).Times(1);        // only the init probe
    EXPECT_CALL(*mock_driver, read_cpu_frequencies()).Times(1);  // only the init probe
    EXPECT_CALL(*mock_driver, read_rusage()).Times(1);           // only the init probe

    device<MockDriver> dev(mock_driver, 0, monitored_cpus);
    auto               result = dev.get_cpu_metrics(none);

    // Per-CPU entries exist (from make_empty_metrics) but all values are zero
    EXPECT_EQ(result.cpu_data.size(), 4u);
    for(const auto& cpu : result.cpu_data)
    {
        EXPECT_FLOAT_EQ(cpu.frequency, 0.0f);
        EXPECT_DOUBLE_EQ(cpu.load, 0.0);
    }
    EXPECT_EQ(result.process_data.page_rss, 0);
    EXPECT_EQ(result.process_data.virt_mem, 0);
}

}  // namespace rocprofsys::pmc::collectors::cpu::testing
