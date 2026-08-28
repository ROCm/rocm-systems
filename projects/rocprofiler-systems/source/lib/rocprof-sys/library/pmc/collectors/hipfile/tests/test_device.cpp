// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mock_backend.hpp"

#include "library/pmc/collectors/hipfile/device.hpp"
#include "library/pmc/collectors/hipfile/types.hpp"

#include <gtest/gtest.h>

#include <cstddef>
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

namespace slot_bytes
{
constexpr std::uint64_t slot0 = 111;
constexpr std::uint64_t slot4 = 999;
constexpr std::uint64_t kb1   = 1000;
constexpr std::uint64_t kb2p5 = 2500;
constexpr std::uint64_t kb1p5 = 1500;
constexpr std::uint64_t kb5   = 5000;
constexpr std::uint64_t b100  = 100;
constexpr std::uint64_t b200  = 200;
constexpr std::uint64_t b300  = 300;
constexpr std::uint64_t kb4   = 4096;
constexpr std::uint64_t mb1   = 1'000'000;
constexpr std::uint64_t kb2   = 2000;
constexpr std::uint64_t mb2   = 2'000'000;
constexpr std::uint64_t kb9   = 9'000;
constexpr std::uint64_t kb4w  = 4'000;
constexpr std::uint64_t b500  = 500;
constexpr std::uint64_t kb10  = 10'000;
}  // namespace slot_bytes

namespace counter_values
{
constexpr std::uint64_t read_bytes       = 11;
constexpr std::uint64_t write_bytes      = 22;
constexpr std::uint64_t read_ops         = 33;
constexpr std::uint64_t write_ops        = 44;
constexpr std::uint64_t fastpath_reads   = 55;
constexpr std::uint64_t fastpath_writes  = 66;
constexpr std::uint64_t fallback_reads   = 77;
constexpr std::uint64_t fallback_writes  = 88;
constexpr std::uint64_t unaligned_reads  = 99;
constexpr std::uint64_t unaligned_writes = 110;
constexpr std::uint64_t read_errors      = 121;
constexpr std::uint64_t write_errors     = 132;
}  // namespace counter_values

namespace attribution
{
constexpr std::uint64_t read_ops         = 100;
constexpr std::uint64_t fastpath_reads   = 70;
constexpr std::uint64_t fallback_reads   = 30;
constexpr std::uint64_t read_ops_compat  = 40;
constexpr std::uint64_t write_ops_compat = 20;
}  // namespace attribution

constexpr std::uint32_t k_remapped_profiler_index = 4;
constexpr std::size_t   k_inactive_profiler_index = 5;
constexpr double        k_one_second_bandwidth    = 1000.0;
constexpr double        k_half_second_bandwidth   = 500.0;
constexpr double        k_two_megabyte_bandwidth  = 2'000'000.0;
constexpr double        k_nine_kilobyte_bandwidth = 9'000.0;
constexpr double        k_four_kilobyte_bandwidth = 4'000.0;
constexpr double        k_five_kilobyte_bandwidth = 5000.0;

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

TEST_F(HipFileDeviceTest, index_and_name_track_the_profiler_device_index)
{
    const device_t remapped{ m_backend, 0, 4 };

    EXPECT_EQ(remapped.get_hipfile_slot(), 0U);
    EXPECT_EQ(remapped.get_index(), 4U);
    EXPECT_EQ(remapped.get_name(), "GPU 4");
}

TEST_F(HipFileDeviceTest, is_supported_bounds_the_hipfile_slot_not_the_profiler_index)
{
    const device_t high_profiler_index{ m_backend, 0, MAX_GPUS + 4 };
    EXPECT_TRUE(high_profiler_index.is_supported());

    const device_t slot_past_capacity{ m_backend, MAX_GPUS, 0 };
    EXPECT_FALSE(slot_past_capacity.is_supported());
}

TEST_F(HipFileDeviceTest, all_metrics_supported_for_valid_ordinal)
{
    EXPECT_EQ(m_device->get_supported_metrics().value, ALL_HIPFILE_METRICS);
}

TEST_F(HipFileDeviceTest, metrics_are_read_from_the_hipfile_slot_not_the_profiler_index)
{
    m_backend->gpu(0).read_bytes = slot_bytes::slot0;
    m_backend->gpu(4).read_bytes = slot_bytes::slot4;

    device_t remapped{ m_backend, 0, k_remapped_profiler_index };

    EXPECT_EQ(remapped.get_metrics(m_enabled, TS_1).read_bytes, slot_bytes::slot0);
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
    m_backend->gpu(0).read_bytes = slot_bytes::kb1;
    EXPECT_EQ(sample(TS_1).read_bytes, slot_bytes::kb1);

    m_backend->gpu(0).read_bytes = slot_bytes::kb2p5;
    const auto second            = sample(TS_2);

    // The delta over this interval is 1500. Publishing that instead of the running
    // total is what this test exists to prevent: every other byte counter the profiler
    // ships (PCIe accumulator, XGMI, NIC) reports cumulative under ABSOLUTE.
    EXPECT_EQ(second.read_bytes, slot_bytes::kb2p5);
    EXPECT_NE(second.read_bytes, slot_bytes::kb1p5);
}

TEST_F(HipFileDeviceTest, all_counters_pass_through_unmodified)
{
    auto& stats            = m_backend->gpu(0);
    stats.read_bytes       = counter_values::read_bytes;
    stats.write_bytes      = counter_values::write_bytes;
    stats.read_ops         = counter_values::read_ops;
    stats.write_ops        = counter_values::write_ops;
    stats.fastpath_reads   = counter_values::fastpath_reads;
    stats.fastpath_writes  = counter_values::fastpath_writes;
    stats.fallback_reads   = counter_values::fallback_reads;
    stats.fallback_writes  = counter_values::fallback_writes;
    stats.unaligned_reads  = counter_values::unaligned_reads;
    stats.unaligned_writes = counter_values::unaligned_writes;
    stats.read_errors      = counter_values::read_errors;
    stats.write_errors     = counter_values::write_errors;

    const auto out = sample(TS_1);

    EXPECT_EQ(out.read_bytes, counter_values::read_bytes);
    EXPECT_EQ(out.write_bytes, counter_values::write_bytes);
    EXPECT_EQ(out.read_ops, counter_values::read_ops);
    EXPECT_EQ(out.write_ops, counter_values::write_ops);
    EXPECT_EQ(out.fastpath_reads, counter_values::fastpath_reads);
    EXPECT_EQ(out.fastpath_writes, counter_values::fastpath_writes);
    EXPECT_EQ(out.fallback_reads, counter_values::fallback_reads);
    EXPECT_EQ(out.fallback_writes, counter_values::fallback_writes);
    EXPECT_EQ(out.unaligned_reads, counter_values::unaligned_reads);
    EXPECT_EQ(out.unaligned_writes, counter_values::unaligned_writes);
    EXPECT_EQ(out.read_errors, counter_values::read_errors);
    EXPECT_EQ(out.write_errors, counter_values::write_errors);
}

TEST_F(HipFileDeviceTest, counter_reset_is_not_clamped)
{
    m_backend->gpu(0).read_bytes = slot_bytes::kb5;
    sample(TS_1);

    // hipFile reset its stats. The counter reports what hipFile reports; only the
    // derived bandwidth needs to defend against the backwards step.
    m_backend->gpu(0).read_bytes = slot_bytes::b100;
    EXPECT_EQ(sample(TS_2).read_bytes, slot_bytes::b100);
}

// ── Fastpath / fallback attribution ─────────────────────────────────────────

TEST_F(HipFileDeviceTest, fastpath_and_fallback_reads_are_distinct)
{
    m_backend->gpu(0).read_ops       = attribution::read_ops;
    m_backend->gpu(0).fastpath_reads = attribution::fastpath_reads;
    m_backend->gpu(0).fallback_reads = attribution::fallback_reads;

    const auto out = sample(TS_1);

    EXPECT_EQ(out.fastpath_reads, attribution::fastpath_reads);
    EXPECT_EQ(out.fallback_reads, attribution::fallback_reads);
    EXPECT_EQ(out.fastpath_reads + out.fallback_reads, out.read_ops);
}

TEST_F(HipFileDeviceTest, compat_mode_reports_only_fallback)
{
    // What HIPFILE_FORCE_COMPAT_MODE looks like from the collector's side: every
    // operation goes through POSIX, so the fastpath tracks must stay flat at zero.
    m_backend->gpu(0).read_ops        = attribution::read_ops_compat;
    m_backend->gpu(0).fallback_reads  = attribution::read_ops_compat;
    m_backend->gpu(0).write_ops       = attribution::write_ops_compat;
    m_backend->gpu(0).fallback_writes = attribution::write_ops_compat;

    const auto out = sample(TS_1);

    EXPECT_EQ(out.fallback_reads, attribution::read_ops_compat);
    EXPECT_EQ(out.fallback_writes, attribution::write_ops_compat);
    EXPECT_EQ(out.fastpath_reads, 0U);
    EXPECT_EQ(out.fastpath_writes, 0U);
}

// ── Bandwidth: wall-clock normalisation ─────────────────────────────────────

TEST_F(HipFileDeviceTest, bandwidth_first_sample_is_zero)
{
    m_backend->gpu(0).read_bytes = slot_bytes::mb1;

    // No interval has elapsed yet, so there is no rate to report. Dividing the lifetime
    // total by nothing would open every trace with a spike that never happened.
    EXPECT_DOUBLE_EQ(sample(TS_1).read_bandwidth, 0.0);
}

TEST_F(HipFileDeviceTest, bandwidth_normalised_to_wall_clock)
{
    m_backend->gpu(0).read_bytes = slot_bytes::kb1;
    sample(TS_1);

    // 1000 more bytes across a one-second interval.
    m_backend->gpu(0).read_bytes = slot_bytes::kb2;

    EXPECT_DOUBLE_EQ(sample(TS_2).read_bandwidth, k_one_second_bandwidth);
}

TEST_F(HipFileDeviceTest, bandwidth_halves_when_interval_doubles)
{
    m_backend->gpu(0).read_bytes = 0;
    sample(TS_1);

    m_backend->gpu(0).read_bytes = slot_bytes::kb1;
    const auto one_second        = sample(TS_2).read_bandwidth;

    m_backend->gpu(0).read_bytes = slot_bytes::kb2;
    const auto also_one_second   = sample(TS_3).read_bandwidth;

    EXPECT_DOUBLE_EQ(one_second, k_one_second_bandwidth);
    EXPECT_DOUBLE_EQ(also_one_second, k_one_second_bandwidth);

    // Same bytes over a two-second interval must read half the rate.
    device_t slow{ m_backend, 0 };
    m_backend->gpu(0).read_bytes = 0;
    slow.get_metrics(m_enabled, TS_1);
    m_backend->gpu(0).read_bytes = slot_bytes::kb1;
    EXPECT_DOUBLE_EQ(slow.get_metrics(m_enabled, TS_3).read_bandwidth,
                     k_half_second_bandwidth);
}

TEST_F(HipFileDeviceTest, write_bandwidth_uses_write_bytes)
{
    m_backend->gpu(0).read_bytes  = 0;
    m_backend->gpu(0).write_bytes = 0;
    sample(TS_1);

    m_backend->gpu(0).read_bytes  = slot_bytes::kb9;
    m_backend->gpu(0).write_bytes = slot_bytes::kb4w;

    const auto out = sample(TS_2);

    // Distinct values so a read/write transposition in the bandwidth path cannot pass.
    EXPECT_DOUBLE_EQ(out.read_bandwidth, k_nine_kilobyte_bandwidth);
    EXPECT_DOUBLE_EQ(out.write_bandwidth, k_four_kilobyte_bandwidth);
}

TEST_F(HipFileDeviceTest, bandwidth_is_zero_when_no_io_occurred)
{
    m_backend->gpu(0).read_bytes = slot_bytes::kb4;
    sample(TS_1);

    // Bytes unchanged: an idle interval reads as zero bandwidth, not as a repeat of
    // the previous rate.
    EXPECT_DOUBLE_EQ(sample(TS_2).read_bandwidth, 0.0);
}

TEST_F(HipFileDeviceTest, bandwidth_zero_elapsed_returns_zero)
{
    m_backend->gpu(0).read_bytes = slot_bytes::kb1;
    sample(TS_1);

    m_backend->gpu(0).read_bytes = slot_bytes::kb2;

    // Two samples at the same timestamp would divide by zero.
    EXPECT_DOUBLE_EQ(sample(TS_1).read_bandwidth, 0.0);
}

TEST_F(HipFileDeviceTest, bandwidth_survives_counter_reset)
{
    m_backend->gpu(0).read_bytes = slot_bytes::kb10;
    sample(TS_1);

    m_backend->gpu(0).read_bytes = slot_bytes::b500;

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

    m_backend->gpu(0).read_bytes = slot_bytes::mb2;

    EXPECT_DOUBLE_EQ(sample(TS_2).read_bandwidth, k_two_megabyte_bandwidth);
}

// ── Availability ────────────────────────────────────────────────────────────

TEST_F(HipFileDeviceTest, unavailable_backend_marks_query_failed)
{
    m_backend->gpu(0).read_bytes = slot_bytes::kb4;
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
    m_backend->gpu(0).read_bytes = slot_bytes::b100;
    m_backend->gpu(1).read_bytes = slot_bytes::b200;
    m_backend->gpu(2).read_bytes = slot_bytes::b300;

    device_t gpu1{ m_backend, 1 };
    device_t gpu2{ m_backend, 2 };

    EXPECT_EQ(sample(TS_1).read_bytes, slot_bytes::b100);
    EXPECT_EQ(gpu1.get_metrics(m_enabled, TS_1).read_bytes, slot_bytes::b200);
    EXPECT_EQ(gpu2.get_metrics(m_enabled, TS_1).read_bytes, slot_bytes::b300);
}

TEST_F(HipFileDeviceTest, bandwidth_state_is_per_device)
{
    device_t gpu1{ m_backend, 1 };

    m_backend->gpu(0).read_bytes = 0;
    m_backend->gpu(1).read_bytes = 0;
    sample(TS_1);
    gpu1.get_metrics(m_enabled, TS_1);

    m_backend->gpu(0).read_bytes = slot_bytes::kb1;
    m_backend->gpu(1).read_bytes = slot_bytes::kb5;

    EXPECT_DOUBLE_EQ(sample(TS_2).read_bandwidth, k_one_second_bandwidth);
    EXPECT_DOUBLE_EQ(gpu1.get_metrics(m_enabled, TS_2).read_bandwidth,
                     k_five_kilobyte_bandwidth);
}

TEST_F(HipFileDeviceTest, inactive_gpu_reports_zeros)
{
    m_backend->gpu(0).read_bytes = slot_bytes::kb4;

    device_t   idle{ m_backend, k_inactive_profiler_index };
    const auto out = idle.get_metrics(m_enabled, TS_1);

    EXPECT_FALSE(out.query_failed);
    EXPECT_EQ(out.read_bytes, 0U);
    EXPECT_DOUBLE_EQ(out.read_bandwidth, 0.0);
}

TEST_F(HipFileDeviceTest, both_devices_forward_the_same_timestamp)
{
    device_t gpu1{ m_backend, 1 };

    sample(TS_1);
    gpu1.get_metrics(m_enabled, TS_1);

    // This mock records every get_stats; it does not memoize. One hipFileGetStatsL3
    // per interval is the real backend's job (snapshot_is_memoized_per_timestamp).
    // What the device layer must not do is query with a private clock: that would
    // miss the backend memo and issue a second L3 call for the same sample tick.
    ASSERT_EQ(m_backend->queried_timestamps.size(), 2U);
    EXPECT_EQ(m_backend->queried_timestamps[0], TS_1);
    EXPECT_EQ(m_backend->queried_timestamps[1], TS_1);
}

}  // namespace
}  // namespace rocprofsys::pmc::collectors::hipfile::testing
