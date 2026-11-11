// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/collection/tests/log_capture.hpp"
#include "core/collection/tests/recording_sink.hpp"
#include "library/pmc/collectors/hipfile/device.hpp"
#include "library/pmc/collectors/hipfile/tests/mock_backend.hpp"
#include "library/pmc/collectors/hipfile/types.hpp"
#include "library/pmc/common/types.hpp"
#include "library/pmc/pollers/hipfile.hpp"
#include "rocprofiler-systems/categories.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace rocprofsys::pmc::pollers
{
namespace
{
namespace hipfile_metrics = pmc::collectors::hipfile;

using collection::testing::log_capture;
using collection::testing::recording_sink;
using hipfile_metrics::testing::mock_backend;
using hipfile_metrics::testing::mock_provider;

constexpr std::size_t   k_first_gpu      = 3;
constexpr std::size_t   k_second_gpu     = 5;
constexpr std::uint64_t k_read_bytes     = 8192;
constexpr std::uint64_t k_write_bytes    = 1024;
constexpr std::uint64_t k_timestamp      = 1000;
constexpr std::uint64_t k_next_timestamp = 2000;

struct stub_settings
{
    inline static hipfile_metrics::enabled_metrics enabled{};
    inline static std::vector<std::size_t>         visible{};

    static device_filter            get_gpu_device_filter() { return {}; }
    static std::vector<std::size_t> get_visible_gpu_type_indices() { return visible; }
    static hipfile_metrics::enabled_metrics get_hipfile_enabled_metrics()
    {
        return enabled;
    }
};

using device_t = hipfile_metrics::device<mock_backend>;
using poller_t = hipfile<mock_provider, device_t, stub_settings>;

/** A hipFile session whose stats query fails on demand. */
struct failing_backend
{
    hipfile_metrics::stats_snapshot snapshot{};
    bool                            fail = false;

    [[nodiscard]] const hipfile_metrics::stats_snapshot& get_stats(std::uint64_t)
    {
        if(fail)
        {
            throw std::runtime_error("hipFile session lost");
        }
        return snapshot;
    }

    [[nodiscard]] bool is_available() const noexcept { return true; }
};

/** Hands out one failing_backend per GPU so a single GPU can fail. */
struct failing_provider
{
    std::vector<std::shared_ptr<failing_backend>> backends;
    bool                                          shutdown_called = false;

    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_devices(
        const std::vector<std::size_t>& type_indices)
    {
        std::vector<std::shared_ptr<Device>> devices;
        for(std::size_t slot = 0; slot < type_indices.size(); ++slot)
        {
            backends.push_back(std::make_shared<failing_backend>());
            devices.push_back(
                std::make_shared<Device>(backends.back(), slot, type_indices[slot]));
        }
        return devices;
    }

    void shutdown() { shutdown_called = true; }
};

class hipfile_poller_test_interface : public ::testing::Test
{
protected:
    void SetUp() override
    {
        stub_settings::enabled.value            = 0;
        stub_settings::enabled.bits.read_bytes  = 1;
        stub_settings::enabled.bits.write_bytes = 1;
        stub_settings::visible                  = { k_first_gpu };
        m_provider                              = std::make_shared<mock_provider>();
        m_provider->backend->gpu(0).read_bytes  = k_read_bytes;
        m_provider->backend->gpu(0).write_bytes = k_write_bytes;
        m_poller                                = std::make_unique<poller_t>(m_provider);
    }

    recording_sink                 m_sink;
    std::shared_ptr<mock_provider> m_provider;
    std::unique_ptr<poller_t>      m_poller;
};

TEST_F(hipfile_poller_test_interface, columns_are_the_enabled_metrics_of_each_visible_gpu)
{
    ASSERT_TRUE(m_poller->describe(m_sink));

    const auto schemas = m_sink.schemas();
    ASSERT_EQ(schemas.size(), 1U);
    const auto& description = schemas.front();
    EXPECT_EQ(description.agent, agent_type::gpu);
    EXPECT_EQ(description.device, k_first_gpu);
    EXPECT_EQ(description.target_arch, "GPU");
    EXPECT_EQ(description.event_category, "hipfile");
    ASSERT_EQ(description.columns.size(), 2U);
    const auto& read_bytes = description.columns[0];
    EXPECT_EQ(read_bytes.pmc_name, "device_storage_read_bytes");
    EXPECT_EQ(read_bytes.symbol, "Read Bytes");
    EXPECT_EQ(read_bytes.unit, "bytes");
    EXPECT_EQ(read_bytes.perfetto_track_name, "GPU [3] Storage Read Bytes (S)");
    EXPECT_EQ(read_bytes.rocpd_track_name, "GPU [3] Storage Read Bytes (S)");
    EXPECT_EQ(read_bytes.category_id, ROCPROFSYS_CATEGORY_HIPFILE);
    EXPECT_EQ(description.columns[1].pmc_name, "device_storage_write_bytes");
}

TEST_F(hipfile_poller_test_interface, poll_writes_the_counters)
{
    ASSERT_TRUE(m_poller->describe(m_sink));

    m_poller->poll(k_timestamp, m_sink);

    const auto batches = m_sink.batches();
    ASSERT_EQ(batches.size(), 1U);
    EXPECT_EQ(batches.front().timestamp, k_timestamp);
    EXPECT_THAT(batches.front().values,
                ::testing::ElementsAre(k_read_bytes, k_write_bytes));
}

TEST_F(hipfile_poller_test_interface, failed_query_writes_no_row)
{
    ASSERT_TRUE(m_poller->describe(m_sink));
    m_provider->backend->available = false;

    m_poller->poll(k_timestamp, m_sink);
    m_provider->backend->available = true;
    m_poller->poll(k_next_timestamp, m_sink);

    const auto batches = m_sink.batches();
    ASSERT_EQ(batches.size(), 1U);
    EXPECT_EQ(batches.front().timestamp, k_next_timestamp);
}

TEST_F(hipfile_poller_test_interface,
       describe_is_false_without_visible_gpus_and_shuts_down)
{
    stub_settings::visible.clear();

    EXPECT_FALSE(m_poller->describe(m_sink));
    EXPECT_TRUE(m_sink.schemas().empty());
    EXPECT_TRUE(m_provider->shutdown_called);
}

TEST_F(hipfile_poller_test_interface, describe_is_false_when_no_metric_is_enabled)
{
    stub_settings::enabled.value = 0;

    EXPECT_FALSE(m_poller->describe(m_sink));
    EXPECT_TRUE(m_sink.schemas().empty());
}

TEST_F(hipfile_poller_test_interface, release_shuts_the_provider_down_and_drops_the_gpus)
{
    ASSERT_TRUE(m_poller->describe(m_sink));

    m_poller->release();
    m_poller->poll(k_timestamp, m_sink);

    EXPECT_TRUE(m_provider->shutdown_called);
    EXPECT_TRUE(m_sink.batches().empty());
    EXPECT_EQ(m_provider->backend->call_count, 0U);
}

TEST_F(hipfile_poller_test_interface, failed_gpu_is_disabled_and_the_others_keep_writing)
{
    stub_settings::visible = { k_first_gpu, k_second_gpu };
    const auto provider    = std::make_shared<failing_provider>();
    hipfile<failing_provider, hipfile_metrics::device<failing_backend>, stub_settings>
        poller{ provider };
    ASSERT_TRUE(poller.describe(m_sink));
    ASSERT_EQ(provider->backends.size(), 2U);
    provider->backends[0]->fail = true;

    const log_capture logs;
    poller.poll(k_timestamp, m_sink);
    provider->backends[0]->fail = false;
    poller.poll(k_next_timestamp, m_sink);

    EXPECT_TRUE(m_sink.batches_of(0).empty());
    EXPECT_EQ(m_sink.batches_of(1).size(), 2U);
    EXPECT_EQ(logs.count("error", "hipfile device"), 1U);
}
TEST_F(hipfile_poller_test_interface, selecting_a_group_gives_both_directions)
{
    stub_settings::enabled.value = hipfile_metrics::metric_group_mask("fastpath");

    ASSERT_TRUE(m_poller->describe(m_sink));

    const auto schemas = m_sink.schemas();
    ASSERT_EQ(schemas.size(), 1U);
    ASSERT_EQ(schemas.front().columns.size(), 2U);
    EXPECT_EQ(schemas.front().columns[0].symbol, "Fastpath Reads");
    EXPECT_EQ(schemas.front().columns[1].symbol, "Fastpath Writes");
}

TEST_F(hipfile_poller_test_interface, disabled_metrics_have_no_column)
{
    stub_settings::enabled.value = hipfile_metrics::metric_bit_mask("Read Bytes");
    ASSERT_TRUE(m_poller->describe(m_sink));

    m_poller->poll(k_timestamp, m_sink);

    const auto schemas = m_sink.schemas();
    ASSERT_EQ(schemas.size(), 1U);
    ASSERT_EQ(schemas.front().columns.size(), 1U);
    EXPECT_EQ(schemas.front().columns.front().symbol, "Read Bytes");
    const auto rows = m_sink.batches();
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_THAT(rows.front().values, ::testing::ElementsAre(k_read_bytes));
}

TEST_F(hipfile_poller_test_interface, each_gpu_has_its_own_schema_and_track_names)
{
    stub_settings::visible = { k_first_gpu, k_second_gpu };

    ASSERT_TRUE(m_poller->describe(m_sink));

    const auto schemas = m_sink.schemas();
    ASSERT_EQ(schemas.size(), 2U);
    EXPECT_EQ(schemas[0].device, k_first_gpu);
    EXPECT_EQ(schemas[1].device, k_second_gpu);
    EXPECT_EQ(schemas[0].columns.front().perfetto_track_name,
              "GPU [3] Storage Read Bytes (S)");
    EXPECT_EQ(schemas[1].columns.front().perfetto_track_name,
              "GPU [5] Storage Read Bytes (S)");
}

TEST_F(hipfile_poller_test_interface, bandwidth_is_bytes_per_second_of_the_poll_interval)
{
    constexpr std::uint64_t k_first_second  = 1'000'000'000;
    constexpr std::uint64_t k_second_second = 2'000'000'000;
    stub_settings::enabled.value = hipfile_metrics::metric_group_mask("bandwidth");
    ASSERT_TRUE(m_poller->describe(m_sink));

    m_provider->backend->gpu(0).read_bytes = 0;
    m_poller->poll(k_first_second, m_sink);
    m_provider->backend->gpu(0).read_bytes = k_read_bytes;
    m_poller->poll(k_second_second, m_sink);

    const auto schemas = m_sink.schemas();
    ASSERT_EQ(schemas.size(), 1U);
    ASSERT_EQ(schemas.front().columns.front().symbol, "Read Bandwidth");
    const auto batches = m_sink.batches();
    ASSERT_EQ(batches.size(), 2U);
    EXPECT_DOUBLE_EQ(batches[0].values.front(), 0.0);
    EXPECT_DOUBLE_EQ(batches[1].values.front(), static_cast<double>(k_read_bytes));
}
}  // namespace
}  // namespace rocprofsys::pmc::pollers
