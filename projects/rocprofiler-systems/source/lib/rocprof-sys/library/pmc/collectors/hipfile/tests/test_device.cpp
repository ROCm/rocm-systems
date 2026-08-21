// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mock_backend.hpp"

#include "library/pmc/collectors/hipfile/device.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

namespace rocprofsys::pmc::collectors::hipfile::testing
{
namespace
{
using device_t = device<mock_backend>;

constexpr std::uint64_t NS_PER_SEC = 1'000'000'000;
constexpr std::uint64_t TS_1       = 1 * NS_PER_SEC;
constexpr std::uint64_t TS_2       = 2 * NS_PER_SEC;
constexpr std::uint64_t TS_3       = 3 * NS_PER_SEC;

class HipFileDeviceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_backend       = std::make_shared<mock_backend>();
        m_device        = std::make_shared<device_t>(m_backend, 0);
        m_enabled.value = ALL_HIPFILE_METRICS;
    }

    [[nodiscard]] metrics sample(std::uint64_t timestamp)
    {
        return m_device->get_metrics(m_enabled, timestamp);
    }

    std::shared_ptr<mock_backend> m_backend;
    std::shared_ptr<device_t>     m_device;
    enabled_metrics               m_enabled{};
};

// ── Identity and metric set ─────────────────────────────────────────────────

TEST_F(HipFileDeviceTest, index_and_name_track_the_gpu_ordinal)
{
    device_t third{ m_backend, 3 };

    EXPECT_EQ(third.get_index(), 3U);
    EXPECT_EQ(third.get_name(), "GPU 3");
}

TEST_F(HipFileDeviceTest, all_metrics_supported_for_valid_ordinal)
{
    EXPECT_EQ(m_device->get_supported_metrics().value, ALL_HIPFILE_METRICS);
}

TEST_F(HipFileDeviceTest, ordinal_beyond_snapshot_supports_nothing)
{
    // Guards the read that would otherwise run off the end of per_gpu.
    device_t out_of_range{ m_backend, MAX_GPUS };

    EXPECT_FALSE(out_of_range.is_supported());
    EXPECT_EQ(out_of_range.get_supported_metrics().value, 0U);
    EXPECT_TRUE(out_of_range.get_metrics(m_enabled, TS_1).query_failed);
}

// ── Cumulative counter semantics ────────────────────────────────────────────

TEST_F(HipFileDeviceTest, counters_are_cumulative_not_deltas)
{
    m_backend->gpu(0).read_bytes = 1000;
    EXPECT_EQ(sample(TS_1).read_bytes, 1000U);

    m_backend->gpu(0).read_bytes = 2500;
    const auto second            = sample(TS_2);

    // The delta over this interval is 1500. Publishing that instead of the running
    // total is what this test exists to prevent: every other byte counter the profiler
    // ships (PCIe accumulator, XGMI, NIC) reports cumulative under ABSOLUTE.
    EXPECT_EQ(second.read_bytes, 2500U);
    EXPECT_NE(second.read_bytes, 1500U);
}

TEST_F(HipFileDeviceTest, all_counters_pass_through_unmodified)
{
    auto& stats            = m_backend->gpu(0);
    stats.read_bytes       = 11;
    stats.write_bytes      = 22;
    stats.read_ops         = 33;
    stats.write_ops        = 44;
    stats.fastpath_reads   = 55;
    stats.fastpath_writes  = 66;
    stats.fallback_reads   = 77;
    stats.fallback_writes  = 88;
    stats.unaligned_reads  = 99;
    stats.unaligned_writes = 110;
    stats.read_errors      = 121;
    stats.write_errors     = 132;

    const auto out = sample(TS_1);

    EXPECT_EQ(out.read_bytes, 11U);
    EXPECT_EQ(out.write_bytes, 22U);
    EXPECT_EQ(out.read_ops, 33U);
    EXPECT_EQ(out.write_ops, 44U);
    EXPECT_EQ(out.fastpath_reads, 55U);
    EXPECT_EQ(out.fastpath_writes, 66U);
    EXPECT_EQ(out.fallback_reads, 77U);
    EXPECT_EQ(out.fallback_writes, 88U);
    EXPECT_EQ(out.unaligned_reads, 99U);
    EXPECT_EQ(out.unaligned_writes, 110U);
    EXPECT_EQ(out.read_errors, 121U);
    EXPECT_EQ(out.write_errors, 132U);
}

TEST_F(HipFileDeviceTest, counter_reset_is_not_clamped)
{
    m_backend->gpu(0).read_bytes = 5000;
    sample(TS_1);

    // hipFile reset its stats. The counter reports what hipFile reports; only the
    // derived bandwidth needs to defend against the backwards step.
    m_backend->gpu(0).read_bytes = 100;
    EXPECT_EQ(sample(TS_2).read_bytes, 100U);
}

// ── Fastpath / fallback attribution ─────────────────────────────────────────

TEST_F(HipFileDeviceTest, fastpath_and_fallback_reads_are_distinct)
{
    m_backend->gpu(0).read_ops       = 100;
    m_backend->gpu(0).fastpath_reads = 70;
    m_backend->gpu(0).fallback_reads = 30;

    const auto out = sample(TS_1);

    EXPECT_EQ(out.fastpath_reads, 70U);
    EXPECT_EQ(out.fallback_reads, 30U);
    EXPECT_EQ(out.fastpath_reads + out.fallback_reads, out.read_ops);
}

TEST_F(HipFileDeviceTest, compat_mode_reports_only_fallback)
{
    // What HIPFILE_FORCE_COMPAT_MODE looks like from the collector's side: every
    // operation goes through POSIX, so the fastpath tracks must stay flat at zero.
    m_backend->gpu(0).read_ops        = 40;
    m_backend->gpu(0).fallback_reads  = 40;
    m_backend->gpu(0).write_ops       = 20;
    m_backend->gpu(0).fallback_writes = 20;

    const auto out = sample(TS_1);

    EXPECT_EQ(out.fallback_reads, 40U);
    EXPECT_EQ(out.fallback_writes, 20U);
    EXPECT_EQ(out.fastpath_reads, 0U);
    EXPECT_EQ(out.fastpath_writes, 0U);
}

// ── Bandwidth: wall-clock normalisation ─────────────────────────────────────

TEST_F(HipFileDeviceTest, bandwidth_first_sample_is_zero)
{
    m_backend->gpu(0).read_bytes = 1'000'000;

    // No interval has elapsed yet, so there is no rate to report. Dividing the lifetime
    // total by nothing would open every trace with a spike that never happened.
    EXPECT_DOUBLE_EQ(sample(TS_1).read_bandwidth, 0.0);
}

TEST_F(HipFileDeviceTest, bandwidth_normalised_to_wall_clock)
{
    m_backend->gpu(0).read_bytes = 1000;
    sample(TS_1);

    // 1000 more bytes across a one-second interval.
    m_backend->gpu(0).read_bytes = 2000;

    EXPECT_DOUBLE_EQ(sample(TS_2).read_bandwidth, 1000.0);
}

TEST_F(HipFileDeviceTest, bandwidth_halves_when_interval_doubles)
{
    m_backend->gpu(0).read_bytes = 0;
    sample(TS_1);

    m_backend->gpu(0).read_bytes = 1000;
    const auto one_second        = sample(TS_2).read_bandwidth;

    m_backend->gpu(0).read_bytes = 2000;
    const auto also_one_second   = sample(TS_3).read_bandwidth;

    EXPECT_DOUBLE_EQ(one_second, 1000.0);
    EXPECT_DOUBLE_EQ(also_one_second, 1000.0);

    // Same bytes over a two-second interval must read half the rate.
    device_t slow{ m_backend, 0 };
    m_backend->gpu(0).read_bytes = 0;
    slow.get_metrics(m_enabled, TS_1);
    m_backend->gpu(0).read_bytes = 1000;
    EXPECT_DOUBLE_EQ(slow.get_metrics(m_enabled, TS_3).read_bandwidth, 500.0);
}

TEST_F(HipFileDeviceTest, write_bandwidth_uses_write_bytes)
{
    m_backend->gpu(0).read_bytes  = 0;
    m_backend->gpu(0).write_bytes = 0;
    sample(TS_1);

    m_backend->gpu(0).read_bytes  = 9'000;
    m_backend->gpu(0).write_bytes = 4'000;

    const auto out = sample(TS_2);

    // Distinct values so a read/write transposition in the bandwidth path cannot pass.
    EXPECT_DOUBLE_EQ(out.read_bandwidth, 9'000.0);
    EXPECT_DOUBLE_EQ(out.write_bandwidth, 4'000.0);
}

TEST_F(HipFileDeviceTest, bandwidth_is_zero_when_no_io_occurred)
{
    m_backend->gpu(0).read_bytes = 4096;
    sample(TS_1);

    // Bytes unchanged: an idle interval reads as zero bandwidth, not as a repeat of
    // the previous rate.
    EXPECT_DOUBLE_EQ(sample(TS_2).read_bandwidth, 0.0);
}

TEST_F(HipFileDeviceTest, bandwidth_zero_elapsed_returns_zero)
{
    m_backend->gpu(0).read_bytes = 1000;
    sample(TS_1);

    m_backend->gpu(0).read_bytes = 2000;

    // Two samples at the same timestamp would divide by zero.
    EXPECT_DOUBLE_EQ(sample(TS_1).read_bandwidth, 0.0);
}

TEST_F(HipFileDeviceTest, bandwidth_survives_counter_reset)
{
    m_backend->gpu(0).read_bytes = 10'000;
    sample(TS_1);

    m_backend->gpu(0).read_bytes = 500;

    // A backwards counter means an unmeasurable interval, not a negative rate.
    EXPECT_DOUBLE_EQ(sample(TS_2).read_bandwidth, 0.0);
}

TEST_F(HipFileDeviceTest, bandwidth_ignores_io_duration)
{
    // The backend snapshot deliberately carries no duration fields, so bandwidth cannot
    // silently regress to normalising by time-spent-in-I/O. If a duration is ever
    // reintroduced to gpu_stats, this pairing of a 1-second wall interval with a large
    // byte count keeps the expected value pinned to the wall-clock answer.
    m_backend->gpu(0).read_bytes = 0;
    sample(TS_1);

    m_backend->gpu(0).read_bytes = 2'000'000;

    EXPECT_DOUBLE_EQ(sample(TS_2).read_bandwidth, 2'000'000.0);
}

// ── Availability ────────────────────────────────────────────────────────────

TEST_F(HipFileDeviceTest, unavailable_backend_marks_query_failed)
{
    m_backend->gpu(0).read_bytes = 4096;
    m_backend->available         = false;

    const auto out = sample(TS_1);

    EXPECT_TRUE(out.query_failed);
    EXPECT_EQ(out.read_bytes, 0U);
}

TEST_F(HipFileDeviceTest, available_backend_does_not_mark_query_failed)
{
    EXPECT_FALSE(sample(TS_1).query_failed);
}

TEST_F(HipFileDeviceTest, default_constructed_metrics_are_not_a_failed_query)
{
    // The paused collector emits a value-initialized metrics to drop tracks to zero.
    // If that looked like a failed query it would be suppressed and the tracks would
    // flatline at their last value instead.
    const metrics paused{};

    EXPECT_FALSE(paused.query_failed);
}

// ── Per-GPU isolation ───────────────────────────────────────────────────────

TEST_F(HipFileDeviceTest, devices_read_their_own_ordinal)
{
    m_backend->gpu(0).read_bytes = 100;
    m_backend->gpu(1).read_bytes = 200;
    m_backend->gpu(2).read_bytes = 300;

    device_t gpu1{ m_backend, 1 };
    device_t gpu2{ m_backend, 2 };

    EXPECT_EQ(sample(TS_1).read_bytes, 100U);
    EXPECT_EQ(gpu1.get_metrics(m_enabled, TS_1).read_bytes, 200U);
    EXPECT_EQ(gpu2.get_metrics(m_enabled, TS_1).read_bytes, 300U);
}

TEST_F(HipFileDeviceTest, bandwidth_state_is_per_device)
{
    device_t gpu1{ m_backend, 1 };

    m_backend->gpu(0).read_bytes = 0;
    m_backend->gpu(1).read_bytes = 0;
    sample(TS_1);
    gpu1.get_metrics(m_enabled, TS_1);

    m_backend->gpu(0).read_bytes = 1000;
    m_backend->gpu(1).read_bytes = 5000;

    EXPECT_DOUBLE_EQ(sample(TS_2).read_bandwidth, 1000.0);
    EXPECT_DOUBLE_EQ(gpu1.get_metrics(m_enabled, TS_2).read_bandwidth, 5000.0);
}

TEST_F(HipFileDeviceTest, inactive_gpu_reports_zeros)
{
    m_backend->gpu(0).read_bytes = 4096;

    device_t   idle{ m_backend, 5 };
    const auto out = idle.get_metrics(m_enabled, TS_1);

    EXPECT_FALSE(out.query_failed);
    EXPECT_EQ(out.read_bytes, 0U);
    EXPECT_DOUBLE_EQ(out.read_bandwidth, 0.0);
}

TEST_F(HipFileDeviceTest, one_query_per_timestamp_across_devices)
{
    device_t gpu1{ m_backend, 1 };

    sample(TS_1);
    gpu1.get_metrics(m_enabled, TS_1);

    // The devices share a backend; memoization there is what keeps an interval to one
    // hipFile call. The device layer must not defeat it by querying out of band.
    ASSERT_EQ(m_backend->queried_timestamps.size(), 2U);
    EXPECT_EQ(m_backend->queried_timestamps[0], TS_1);
    EXPECT_EQ(m_backend->queried_timestamps[1], TS_1);
}

}  // namespace
}  // namespace rocprofsys::pmc::collectors::hipfile::testing
