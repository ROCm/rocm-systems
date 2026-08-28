// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mock_backend.hpp"

#include "library/pmc/collectors/hipfile/collector.hpp"
#include "library/pmc/collectors/hipfile/device.hpp"
#include "library/pmc/collectors/hipfile/hipfile_traits.hpp"
#include "library/pmc/collectors/hipfile/types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace rocprofsys::pmc::collectors::hipfile::testing
{
namespace
{
using device_t = device<mock_backend>;

/// One value written to the trace cache, flattened for assertion.
struct recorded_sample
{
    std::size_t   device_id;
    std::string   metric;
    double        value;
    std::uint64_t timestamp;
};

/**
 * @brief Cache stand-in capturing what the collector would have written.
 *
 * Mirrors the real cache_policy's gating - a failed query writes nothing, and only
 * metrics both enabled and supported are emitted - so the workflow tests exercise the
 * same decisions without a trace cache behind them.
 *
 * The real policy stores the whole per-GPU snapshot as one hipfile_pmc_sample and lets
 * the Perfetto and RocPD processors expand it over METRIC_TABLE. This stub expands
 * eagerly instead, because these tests assert on individual metric values; the gating
 * decisions under test are the same either way.
 */
struct stub_cache
{
    inline static std::vector<recorded_sample> samples{};
    inline static std::vector<std::size_t>     metadata_gpus{};
    inline static std::size_t                  category_init_count = 0;

    static void reset()
    {
        samples.clear();
        metadata_gpus.clear();
        category_init_count = 0;
    }

    static void initialize_category_metadata() { ++category_init_count; }
    static void initialize_tracks_metadata() {}
    static void initialize_pmc_metadata(std::size_t gpu_id)
    {
        metadata_gpus.push_back(gpu_id);
    }

    // NOLINTNEXTLINE(readability-function-size) -- needs a refactor of every collector's
    // store_sample
    static void store_sample(std::size_t device_id, const std::string& /*device_name*/,
                             const enabled_metrics& enabled_cfg,
                             const enabled_metrics& supported, const metrics& values,
                             std::uint64_t timestamp)
    {
        if(values.query_failed)
        {
            return;
        }

        const std::uint32_t active = enabled_cfg.value & supported.value;
        for(const auto& metric : METRIC_TABLE)
        {
            if((active & (1U << metric.bit)) == 0U)
            {
                continue;
            }
            samples.push_back(recorded_sample{ .device_id = device_id,
                                               .metric    = metric.suffix,
                                               .value     = metric.value(values),
                                               .timestamp = timestamp });
        }
    }

    [[nodiscard]] static std::vector<recorded_sample> for_metric(const std::string& name)
    {
        std::vector<recorded_sample> out;
        std::ranges::copy_if(samples, std::back_inserter(out),
                             [&name](const recorded_sample& sample_rec) {
                                 return sample_rec.metric == name;
                             });
        return out;
    }
};

struct stub_settings
{
    inline static device_filter            gpu_filter{};
    inline static std::vector<std::size_t> visible_type_indices{};
    inline static enabled_metrics          hipfile_metrics{};
    inline static bool                     perfetto_legacy = false;

    static void set_visible_identity(std::size_t n)
    {
        visible_type_indices.resize(n);
        for(std::size_t idx = 0; idx < n; ++idx)
        {
            visible_type_indices[idx] = idx;
        }
    }

    static void reset()
    {
        gpu_filter      = device_filter{};
        gpu_filter.mode = device_selection_mode::ALL;
        set_visible_identity(2);
        hipfile_metrics.value = ALL_HIPFILE_METRICS;
        perfetto_legacy       = false;
    }

    static device_filter            get_gpu_device_filter() { return gpu_filter; }
    static std::vector<std::size_t> get_visible_gpu_type_indices()
    {
        return visible_type_indices;
    }
    static enabled_metrics get_hipfile_enabled_metrics() { return hipfile_metrics; }
    static bool            get_use_perfetto_legacy_metrics() { return perfetto_legacy; }
};

/// Counts the legacy Perfetto path, which must stay unused: hipFile tracks reach
/// Perfetto through the PMC records, so a second producer here would double every value.
struct stub_perfetto
{
    inline static std::size_t store_count = 0;

    static void reset() { store_count = 0; }

    static void store_sample(std::size_t /*device_id*/, const metrics& /*values*/,
                             std::uint64_t /*timestamp*/)
    {
        ++store_count;
    }
};

struct stub_config
{
    using SettingsApi = stub_settings;
    using PerfettoApi = stub_perfetto;
    using CacheApi    = stub_cache;
};

using collector_t = collector<mock_provider, device_t, stub_config>;

constexpr std::uint64_t NS_PER_SEC = 1'000'000'000;
constexpr std::uint64_t TS_1       = 1 * NS_PER_SEC;
constexpr std::uint64_t TS_2       = 2 * NS_PER_SEC;

namespace test_bytes
{
constexpr std::uint64_t kb512 = 512;
constexpr std::uint64_t kb4   = 4096;
constexpr std::uint64_t b1000 = 1000;
constexpr std::uint64_t b3000 = 3000;
}  // namespace test_bytes

class HipFileCollectorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        stub_cache::reset();
        stub_perfetto::reset();
        stub_settings::reset();
        m_provider  = std::make_shared<mock_provider>();
        m_collector = std::make_unique<collector_t>(m_provider);
    }

    void setup_and_config()
    {
        m_collector->setup();
        m_collector->config();
    }

    [[nodiscard]] mock_backend& backend() { return *m_provider->backend; }

    std::shared_ptr<mock_provider> m_provider;
    std::unique_ptr<collector_t>   m_collector;
};

// ── Lifecycle ───────────────────────────────────────────────────────────────

TEST_F(HipFileCollectorTest, setup_enumerates_visible_gpus)
{
    stub_settings::set_visible_identity(3);

    m_collector->setup();

    EXPECT_EQ(m_collector->get_device_count(), 3U);
}

TEST_F(HipFileCollectorTest, config_registers_metadata_for_every_gpu)
{
    stub_settings::set_visible_identity(2);

    setup_and_config();

    // Registration happens once for the whole known device set, rather than lazily on
    // first activity as the previous implementation did.
    EXPECT_EQ(stub_cache::category_init_count, 1U);
    EXPECT_EQ(stub_cache::metadata_gpus, (std::vector<std::size_t>{ 0, 1 }));
}

TEST_F(HipFileCollectorTest, sample_emits_every_metric_for_every_gpu)
{
    stub_settings::set_visible_identity(2);
    setup_and_config();

    m_collector->sample(static_cast<std::int64_t>(TS_1));

    EXPECT_EQ(stub_cache::samples.size(), 2U * HIPFILE_METRICS_COUNT);
}

TEST_F(HipFileCollectorTest, shutdown_propagates_to_provider)
{
    setup_and_config();

    m_collector->shutdown();

    EXPECT_TRUE(m_provider->shutdown_called);
}

// ── Track scoping ───────────────────────────────────────────────────────────

TEST_F(HipFileCollectorTest, every_track_is_gpu_indexed)
{
    stub_settings::set_visible_identity(2);
    setup_and_config();

    m_collector->sample(static_cast<std::int64_t>(TS_1));

    // No sample may arrive without a GPU it belongs to. The process-scoped counters that
    // used to ride along on GPU 0 are gone entirely.
    for(const auto& sample : stub_cache::samples)
    {
        EXPECT_LT(sample.device_id, 2U);
    }
}

TEST_F(HipFileCollectorTest, no_registration_tracks_are_emitted)
{
    setup_and_config();
    m_collector->sample(static_cast<std::int64_t>(TS_1));

    EXPECT_TRUE(stub_cache::for_metric("File Registrations").empty());
    EXPECT_TRUE(stub_cache::for_metric("Buffer Registrations").empty());
}

TEST_F(HipFileCollectorTest, gpu_zero_carries_no_extra_tracks)
{
    stub_settings::set_visible_identity(2);
    setup_and_config();

    m_collector->sample(static_cast<std::int64_t>(TS_1));

    const auto count_for = [](std::size_t gpu) {
        return std::count_if(
            stub_cache::samples.begin(), stub_cache::samples.end(),
            [gpu](const auto& sample) { return sample.device_id == gpu; });
    };

    EXPECT_EQ(count_for(0), count_for(1));
}

// ── Enablement ──────────────────────────────────────────────────────────────

TEST_F(HipFileCollectorTest, disabled_metrics_are_not_emitted)
{
    stub_settings::set_visible_identity(1);
    stub_settings::hipfile_metrics.value = metric_bit_mask("Read Bytes");

    setup_and_config();
    m_collector->sample(static_cast<std::int64_t>(TS_1));

    ASSERT_EQ(stub_cache::samples.size(), 1U);
    EXPECT_EQ(stub_cache::samples.front().metric, "Read Bytes");
}

TEST_F(HipFileCollectorTest, selecting_a_group_emits_both_directions)
{
    stub_settings::set_visible_identity(1);
    stub_settings::hipfile_metrics.value = metric_group_mask("fastpath");

    setup_and_config();
    m_collector->sample(static_cast<std::int64_t>(TS_1));

    ASSERT_EQ(stub_cache::samples.size(), 2U);
    EXPECT_EQ(stub_cache::for_metric("Fastpath Reads").size(), 1U);
    EXPECT_EQ(stub_cache::for_metric("Fastpath Writes").size(), 1U);
}

// ── Metric groups ───────────────────────────────────────────────────────────

constexpr std::array<const char*, 7> ALL_GROUPS{ "bytes",    "ops",    "fastpath",
                                                 "fallback", "errors", "unaligned",
                                                 "bandwidth" };

TEST(HipFileMetricGroups, each_group_covers_exactly_one_read_and_one_write)
{
    for(const auto* group : ALL_GROUPS)
    {
        EXPECT_EQ(std::popcount(metric_group_mask(group)), 2) << group;
    }
}

TEST(HipFileMetricGroups, groups_partition_the_metric_table)
{
    // Every track reachable through exactly one group: no metric is orphaned (and so
    // unselectable) and none is claimed twice.
    std::uint32_t combined = 0;
    int           bits     = 0;
    for(const auto* group : ALL_GROUPS)
    {
        combined |= metric_group_mask(group);
        bits += std::popcount(metric_group_mask(group));
    }

    EXPECT_EQ(combined, ALL_HIPFILE_METRICS);
    EXPECT_EQ(bits, static_cast<int>(HIPFILE_METRICS_COUNT));
}

TEST(HipFileMetricGroups, unknown_group_selects_nothing)
{
    EXPECT_EQ(metric_group_mask(""), 0U);
    EXPECT_EQ(metric_group_mask("nonsense"), 0U);
    // The per-track spellings were replaced by group names, so they must not resolve.
    EXPECT_EQ(metric_group_mask("read_bytes"), 0U);
}

TEST_F(HipFileCollectorTest, no_metrics_enabled_emits_nothing)
{
    stub_settings::hipfile_metrics.value = 0U;

    setup_and_config();
    m_collector->sample(static_cast<std::int64_t>(TS_1));

    EXPECT_TRUE(stub_cache::samples.empty());
    EXPECT_EQ(backend().call_count, 0U);
}

// ── Values through the full path ────────────────────────────────────────────

TEST_F(HipFileCollectorTest, cumulative_values_reach_the_cache)
{
    stub_settings::set_visible_identity(1);
    setup_and_config();

    backend().gpu(0).read_bytes = test_bytes::b1000;
    m_collector->sample(static_cast<std::int64_t>(TS_1));

    backend().gpu(0).read_bytes = test_bytes::b3000;
    m_collector->sample(static_cast<std::int64_t>(TS_2));

    const auto read_bytes = stub_cache::for_metric("Read Bytes");
    ASSERT_EQ(read_bytes.size(), 2U);
    EXPECT_DOUBLE_EQ(read_bytes[0].value, static_cast<double>(test_bytes::b1000));
    EXPECT_DOUBLE_EQ(read_bytes[1].value, static_cast<double>(test_bytes::b3000));
}

TEST_F(HipFileCollectorTest, bandwidth_reaches_the_cache_wall_clock_normalised)
{
    stub_settings::set_visible_identity(1);
    setup_and_config();

    backend().gpu(0).read_bytes = 0;
    m_collector->sample(static_cast<std::int64_t>(TS_1));

    backend().gpu(0).read_bytes = test_bytes::kb4;
    m_collector->sample(static_cast<std::int64_t>(TS_2));

    const auto bandwidth = stub_cache::for_metric("Read Bandwidth");
    ASSERT_EQ(bandwidth.size(), 2U);
    EXPECT_DOUBLE_EQ(bandwidth[0].value, 0.0);
    EXPECT_DOUBLE_EQ(bandwidth[1].value, static_cast<double>(test_bytes::kb4));
}

// ── Unavailable backend ─────────────────────────────────────────────────────

TEST_F(HipFileCollectorTest, unavailable_backend_emits_nothing)
{
    setup_and_config();
    backend().available = false;

    m_collector->sample(static_cast<std::int64_t>(TS_1));

    // hipFile is simply not in use in this process. Writing zeros would claim a
    // measurement that was never taken.
    EXPECT_TRUE(stub_cache::samples.empty());
}

TEST_F(HipFileCollectorTest, devices_survive_an_unavailable_interval)
{
    setup_and_config();
    const auto before = m_collector->get_device_count();

    backend().available = false;
    m_collector->sample(static_cast<std::int64_t>(TS_1));

    // hipFile stats commonly become readable only after the target's first I/O, so an
    // unavailable interval must not permanently disable the devices.
    EXPECT_EQ(m_collector->get_device_count(), before);

    backend().available         = true;
    backend().gpu(0).read_bytes = test_bytes::kb512;
    m_collector->sample(static_cast<std::int64_t>(TS_2));

    EXPECT_FALSE(stub_cache::samples.empty());
}

// ── Perfetto ────────────────────────────────────────────────────────────────

TEST_F(HipFileCollectorTest, enabling_perfetto_does_not_change_pmc_output)
{
    stub_settings::set_visible_identity(1);
    stub_settings::perfetto_legacy = true;
    setup_and_config();

    m_collector->sample(static_cast<std::int64_t>(TS_1));

    // base::collector routes to PerfettoApi once per device whenever the legacy path is
    // on. hipFile plugs a no-op in there (perfetto_policy), because its tracks already
    // reach Perfetto as PMC records; a real writer would put two producers on every
    // track. What must hold either way is that the PMC output is unchanged.
    EXPECT_EQ(stub_perfetto::store_count, 1U);
    EXPECT_EQ(stub_cache::samples.size(), HIPFILE_METRICS_COUNT);
}

// ── Pause ───────────────────────────────────────────────────────────────────

TEST_F(HipFileCollectorTest, pause_emits_zeros_for_every_track)
{
    stub_settings::set_visible_identity(1);
    setup_and_config();

    backend().gpu(0).read_bytes = test_bytes::kb4;
    m_collector->sample(static_cast<std::int64_t>(TS_1));
    stub_cache::reset();

    m_collector->pause(static_cast<std::int64_t>(TS_2));

    ASSERT_EQ(stub_cache::samples.size(), HIPFILE_METRICS_COUNT);
    for(const auto& sample : stub_cache::samples)
    {
        EXPECT_DOUBLE_EQ(sample.value, 0.0) << sample.metric;
    }
}

TEST_F(HipFileCollectorTest, pause_is_not_suppressed_as_a_failed_query)
{
    setup_and_config();
    backend().available = false;

    m_collector->pause(static_cast<std::int64_t>(TS_1));

    // Pause writes a value-initialized metrics, which must not be mistaken for the
    // unavailable path regardless of what the backend currently reports.
    EXPECT_FALSE(stub_cache::samples.empty());
}

}  // namespace
}  // namespace rocprofsys::pmc::collectors::hipfile::testing
