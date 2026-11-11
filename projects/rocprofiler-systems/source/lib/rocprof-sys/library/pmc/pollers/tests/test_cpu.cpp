// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/procfs/backend.hpp"
#include "core/agent.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/tests/log_capture.hpp"
#include "core/collection/tests/recording_sink.hpp"
#include "library/pmc/collectors/cpu/device.hpp"
#include "library/pmc/collectors/cpu/tests/mock_backend.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "library/pmc/common/types.hpp"
#include "library/pmc/pollers/cpu.hpp"
#include "rocprofiler-systems/categories.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::pollers
{
namespace
{
namespace cpu_metrics = pmc::collectors::cpu;
namespace procfs      = backends::procfs;

using mock_backend = procfs::testing::strict_mock_backend;
using device_t     = cpu_metrics::device<mock_backend>;
using collection::testing::log_capture;
using collection::testing::recording_sink;
using ::testing::AnyNumber;
using ::testing::Return;
using ::testing::Throw;

constexpr std::size_t  k_second_socket  = 1;
constexpr std::size_t  k_third_cpu      = 2;
constexpr std::size_t  k_fourth_cpu     = 3;
constexpr float        k_frequency_mhz  = 2400.0F;
constexpr std::int64_t k_page_rss_bytes = 64 * procfs::testing::MB_TO_BYTES;
// decimal megabytes, as develop wrote them
constexpr double k_bytes_per_mb = 1'000'000.0;
constexpr double k_page_rss_mb  = static_cast<double>(k_page_rss_bytes) / k_bytes_per_mb;
constexpr std::uint64_t k_timestamp      = 1000;
constexpr std::uint64_t k_next_timestamp = 2000;

struct stub_settings
{
    inline static cpu_metrics::enabled_metrics enabled{};

    static device_filter                get_cpu_device_filter() { return {}; }
    static cpu_metrics::enabled_metrics get_cpu_enabled_metrics() { return enabled; }
};

struct stub_provider
{
    using backend_t = mock_backend;

    std::vector<std::shared_ptr<device_t>> devices;

    template <typename Device>
    std::vector<std::shared_ptr<Device>> get_devices()
    {
        static_assert(std::is_same_v<Device, device_t>);
        return devices;
    }
};

using poller_t = cpu<stub_provider, stub_settings>;

std::map<std::size_t, float>
frequencies(const std::set<std::size_t>& cpu_ids)
{
    std::map<std::size_t, float> result;
    for(const auto cpu_id : cpu_ids)
    {
        result[cpu_id] = k_frequency_mhz;
    }
    return result;
}

procfs::rusage_snapshot
rusage()
{
    auto snapshot     = procfs::testing::DEFAULT_RUSAGE;
    snapshot.page_rss = k_page_rss_bytes;
    return snapshot;
}

/** A socket backend that reports frequencies and rusage; load is not supported. Each
 * read is allowed any number of times unless the test sets its own expectation. */
std::shared_ptr<mock_backend>
make_backend(const std::set<std::size_t>& cpu_ids)
{
    const auto backend = std::make_shared<mock_backend>();
    EXPECT_CALL(*backend, read_proc_stat())
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::map<std::size_t, procfs::cpu_jiffies>{}));
    EXPECT_CALL(*backend, read_cpu_frequencies())
        .Times(AnyNumber())
        .WillRepeatedly(Return(frequencies(cpu_ids)));
    EXPECT_CALL(*backend, read_rusage())
        .Times(AnyNumber())
        .WillRepeatedly(Return(rusage()));
    return backend;
}

std::shared_ptr<device_t>
make_socket(std::size_t socket_id, std::set<std::size_t> cpu_ids)
{
    const auto backend = make_backend(cpu_ids);
    return std::make_shared<device_t>(backend, socket_id, std::move(cpu_ids));
}

std::vector<std::string>
pmc_names(const collection::schema& description)
{
    std::vector<std::string> names;
    names.reserve(description.columns.size());
    for(const auto& column : description.columns)
    {
        names.push_back(column.pmc_name);
    }
    return names;
}

class cpu_poller_test_interface : public ::testing::Test
{
protected:
    void SetUp() override
    {
        stub_settings::enabled.value          = 0;
        stub_settings::enabled.bits.frequency = 1;
        stub_settings::enabled.bits.page_rss  = 1;
    }

    [[nodiscard]] bool describe(std::vector<std::shared_ptr<device_t>> devices)
    {
        const auto provider = std::make_shared<stub_provider>();
        provider->devices   = std::move(devices);
        m_poller            = std::make_unique<poller_t>(provider);
        return m_poller->describe(m_sink);
    }

    recording_sink            m_sink;
    std::unique_ptr<poller_t> m_poller;
};

TEST_F(cpu_poller_test_interface,
       process_columns_only_on_the_first_socket_and_a_column_per_core)
{
    ASSERT_TRUE(
        describe({ make_socket(0, { 0, 1 }),
                   make_socket(k_second_socket, { k_third_cpu, k_fourth_cpu }) }));

    const auto schemas = m_sink.schemas();
    ASSERT_EQ(schemas.size(), 2U);
    EXPECT_THAT(pmc_names(schemas[0]),
                ::testing::ElementsAre("process_physical_memory",
                                       "cpu_frequency [0] Core [0]",
                                       "cpu_frequency [0] Core [1]"));
    EXPECT_THAT(pmc_names(schemas[1]),
                ::testing::ElementsAre("cpu_frequency [1] Core [2]",
                                       "cpu_frequency [1] Core [3]"));
    EXPECT_EQ(schemas[1].device, k_second_socket);
    EXPECT_EQ(schemas[0].agent, agent_type::cpu);
    EXPECT_EQ(schemas[0].target_arch, "CPU");
    EXPECT_EQ(schemas[0].event_category, "cpu_frequency");
    const auto& process = schemas[0].columns[0];
    EXPECT_EQ(process.symbol, "Page RSS");
    EXPECT_EQ(process.unit, "MB");
    EXPECT_EQ(process.perfetto_track_name, "CPU Memory Usage (S)");
    EXPECT_EQ(process.category_id, ROCPROFSYS_CATEGORY_PROCESS_PAGE);
    const auto& core = schemas[1].columns[0];
    EXPECT_EQ(core.perfetto_track_name, "CPU [1] Core [2] Frequency (S)");
    EXPECT_EQ(core.unit, "MHz");
    EXPECT_EQ(core.category_id, ROCPROFSYS_CATEGORY_CPU_FREQ);
}

TEST_F(cpu_poller_test_interface, poll_writes_process_and_core_values)
{
    ASSERT_TRUE(describe({ make_socket(0, { 0, 1 }) }));

    m_poller->poll(k_timestamp, m_sink);

    const auto batches = m_sink.batches();
    ASSERT_EQ(batches.size(), 1U);
    EXPECT_EQ(batches.front().timestamp, k_timestamp);
    EXPECT_THAT(batches.front().values,
                ::testing::ElementsAre(::testing::DoubleEq(k_page_rss_mb),
                                       k_frequency_mhz, k_frequency_mhz));
}

TEST_F(cpu_poller_test_interface, core_the_device_did_not_report_has_no_value)
{
    const auto backend = make_backend({ 0, 1 });
    EXPECT_CALL(*backend, read_cpu_frequencies())
        .WillOnce(Return(frequencies({ 0, 1 })))
        .WillOnce(Return(frequencies({ 0 })));
    ASSERT_TRUE(describe(
        { std::make_shared<device_t>(backend, 0, std::set<std::size_t>{ 0, 1 }) }));

    m_poller->poll(k_timestamp, m_sink);

    const auto batches = m_sink.batches();
    ASSERT_EQ(batches.size(), 1U);
    const auto& values = batches.front().values;
    ASSERT_EQ(values.size(), 3U);
    EXPECT_EQ(values[1], k_frequency_mhz);
    EXPECT_TRUE(std::isnan(values[2]));
}

TEST_F(cpu_poller_test_interface, failed_socket_is_disabled_and_the_others_keep_writing)
{
    const auto failing = make_backend({ 0 });
    EXPECT_CALL(*failing, read_cpu_frequencies())
        .WillOnce(Return(frequencies({ 0 })))
        .WillOnce(Throw(std::runtime_error("cpuinfo unreadable")));
    ASSERT_TRUE(
        describe({ std::make_shared<device_t>(failing, 0, std::set<std::size_t>{ 0 }),
                   make_socket(k_second_socket, { 1 }) }));

    const log_capture logs;
    m_poller->poll(k_timestamp, m_sink);
    m_poller->poll(k_next_timestamp, m_sink);

    EXPECT_TRUE(m_sink.batches_of(0).empty());
    EXPECT_EQ(m_sink.batches_of(1).size(), 2U);
    EXPECT_EQ(logs.count("error", "cpu device"), 1U);
}

TEST_F(cpu_poller_test_interface, describe_is_false_without_sockets)
{
    EXPECT_FALSE(describe({}));
    EXPECT_TRUE(m_sink.schemas().empty());
}

TEST_F(cpu_poller_test_interface, describe_is_false_when_no_metric_is_enabled)
{
    stub_settings::enabled.value = 0;

    EXPECT_FALSE(describe({ make_socket(0, { 0 }) }));
    EXPECT_TRUE(m_sink.schemas().empty());
}

TEST_F(cpu_poller_test_interface, release_drops_the_sockets)
{
    ASSERT_TRUE(describe({ make_socket(0, { 0 }) }));

    m_poller->release();
    m_poller->poll(k_timestamp, m_sink);

    EXPECT_TRUE(m_sink.batches().empty());
}
TEST_F(cpu_poller_test_interface, describe_is_false_without_provider)
{
    poller_t poller{ nullptr };

    EXPECT_FALSE(poller.describe(m_sink));
}
}  // namespace
}  // namespace rocprofsys::pmc::pollers
