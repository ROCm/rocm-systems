// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/amd_smi/nic_types.hpp"
#include "core/agent.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/tests/log_capture.hpp"
#include "core/collection/tests/recording_sink.hpp"
#include "library/pmc/collectors/nic/device.hpp"
#include "library/pmc/collectors/nic/tests/mock_nic_backend.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "library/pmc/common/types.hpp"
#include "library/pmc/pollers/nic_smi.hpp"
#include "rocprofiler-systems/categories.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::pollers
{
namespace
{
namespace nic     = pmc::collectors::nic;
namespace nic_api = backends::amd_smi::nic;

using mock_backend = ::testing::StrictMock<backends::amd_smi::testing::mock_nic_backend>;
using device_t     = nic::device<mock_backend>;
using collection::testing::log_capture;
using collection::testing::recording_sink;
using ::testing::_;
using ::testing::AnyNumber;
using ::testing::Return;
using ::testing::Throw;

constexpr std::size_t   k_first_smi_index   = 5;
constexpr std::size_t   k_dropped_smi_index = 7;
constexpr std::size_t   k_second_smi_index  = 9;
constexpr std::uint64_t k_rx_bytes          = 4096;
constexpr std::uint64_t k_tx_cnp            = 12;
constexpr std::uint64_t k_timestamp         = 1000;
constexpr std::uint64_t k_next_timestamp    = 2000;

struct stub_settings
{
    inline static nic::enabled_metrics enabled{};

    static nic_device_filter    get_nic_device_filter() { return {}; }
    static nic::enabled_metrics get_nic_enabled_metrics() { return enabled; }
};

struct stub_provider
{
    std::vector<std::shared_ptr<device_t>> devices;

    template <typename Device>
    std::vector<std::shared_ptr<Device>> get_nic_devices()
    {
        static_assert(std::is_same_v<Device, device_t>);
        return devices;
    }
};

using poller_t = nic_smi<stub_provider, device_t, stub_settings>;

std::vector<nic_api::stat_entry>
rdma_stats()
{
    return { { .name = "rx_rdma_ucast_bytes", .value = k_rx_bytes },
             { .name = "tx_rdma_cnp_pkts", .value = k_tx_cnp } };
}

/** A NIC with one RDMA port; `port_count` 0 makes it unsupported. */
std::shared_ptr<mock_backend>
make_backend(std::uint8_t port_count = 1)
{
    const auto backend = std::make_shared<mock_backend>();
    EXPECT_CALL(*backend, get_nic_asic_info())
        .WillOnce(
            Return(nic_api::asic_info{ .product_name = "NIC", .vendor_name = "AMD" }));
    EXPECT_CALL(*backend, get_nic_port_info())
        .WillOnce(Return(nic_api::port_info{ .device_name = "rdma" }));
    EXPECT_CALL(*backend, get_nic_rdma_info())
        .WillOnce(Return(nic_api::rdma_info{ .port_count = port_count }));
    return backend;
}

std::shared_ptr<device_t>
make_device(std::size_t smi_index)
{
    const auto backend = make_backend();
    EXPECT_CALL(*backend, get_nic_rdma_port_statistics(0))
        .Times(AnyNumber())
        .WillRepeatedly(Return(rdma_stats()));
    return std::make_shared<device_t>(backend, smi_index);
}

class nic_smi_poller_test_interface : public ::testing::Test
{
protected:
    void SetUp() override
    {
        stub_settings::enabled.value                    = 0;
        stub_settings::enabled.bits.rx_rdma_ucast_bytes = 1;
        stub_settings::enabled.bits.tx_rdma_cnp_pkts    = 1;
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

TEST_F(nic_smi_poller_test_interface,
       columns_are_the_enabled_metrics_with_their_track_names)
{
    ASSERT_TRUE(describe({ make_device(k_first_smi_index) }));

    const auto schemas = m_sink.schemas();
    ASSERT_EQ(schemas.size(), 1U);
    const auto& description = schemas.front();
    EXPECT_EQ(description.agent, agent_type::nic);
    EXPECT_EQ(description.target_arch, "NIC");
    EXPECT_EQ(description.event_category, "amd_smi_nic");
    ASSERT_EQ(description.columns.size(), 2U);
    const auto& rx_bytes = description.columns[0];
    EXPECT_EQ(rx_bytes.pmc_name, "nic_rx_ucast_bytes");
    EXPECT_EQ(rx_bytes.symbol, "NIC RX UCast Bytes");
    EXPECT_EQ(rx_bytes.unit, "bytes");
    EXPECT_EQ(rx_bytes.rocpd_track_name, "ainic_rx_rdma_ucast_bytes");
    EXPECT_EQ(rx_bytes.perfetto_track_name, "NIC [5] RX RDMA BYTES (S)");
    EXPECT_EQ(rx_bytes.category_id, ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_RX_UCAST_BYTES);
    EXPECT_EQ(description.columns[1].pmc_name, "nic_tx_cnp_pkts");
    EXPECT_EQ(description.columns[1].rocpd_track_name, "ainic_tx_rdma_cnp_pkts");
}

TEST_F(nic_smi_poller_test_interface, schema_device_is_the_position_among_sampled_nics)
{
    const auto dropped = make_backend(0);
    ASSERT_TRUE(describe({ make_device(k_first_smi_index),
                           std::make_shared<device_t>(dropped, k_dropped_smi_index),
                           make_device(k_second_smi_index) }));

    const auto schemas = m_sink.schemas();
    ASSERT_EQ(schemas.size(), 2U);
    EXPECT_EQ(schemas[0].device, 0U);
    EXPECT_EQ(schemas[1].device, 1U);
    // the Perfetto tracks keep the AMD SMI index
    EXPECT_EQ(schemas[1].columns.front().perfetto_track_name,
              "NIC [9] RX RDMA BYTES (S)");
}

TEST_F(nic_smi_poller_test_interface, poll_writes_the_counters_of_each_nic)
{
    ASSERT_TRUE(describe({ make_device(k_first_smi_index) }));

    m_poller->poll(k_timestamp, m_sink);

    const auto batches = m_sink.batches();
    ASSERT_EQ(batches.size(), 1U);
    EXPECT_EQ(batches.front().timestamp, k_timestamp);
    EXPECT_THAT(batches.front().values, ::testing::ElementsAre(k_rx_bytes, k_tx_cnp));
}

TEST_F(nic_smi_poller_test_interface, describe_is_false_without_a_supported_nic)
{
    const auto unsupported = make_backend(0);

    EXPECT_FALSE(describe({ std::make_shared<device_t>(unsupported, 0) }));
    EXPECT_TRUE(m_sink.schemas().empty());
}

TEST_F(nic_smi_poller_test_interface, describe_is_false_when_no_metric_is_enabled)
{
    stub_settings::enabled.value = 0;

    EXPECT_FALSE(describe({ make_device(k_first_smi_index) }));
    EXPECT_TRUE(m_sink.schemas().empty());
}

TEST_F(nic_smi_poller_test_interface, failed_nic_is_disabled_and_the_others_keep_writing)
{
    const auto failing = make_backend();
    EXPECT_CALL(*failing, get_nic_rdma_port_statistics(_))
        .WillOnce(Return(rdma_stats()))
        .WillOnce(Throw(std::logic_error("nic lost")));
    ASSERT_TRUE(describe({ std::make_shared<device_t>(failing, k_first_smi_index),
                           make_device(k_second_smi_index) }));

    const log_capture logs;
    m_poller->poll(k_timestamp, m_sink);
    m_poller->poll(k_next_timestamp, m_sink);

    EXPECT_TRUE(m_sink.batches_of(0).empty());
    EXPECT_EQ(m_sink.batches_of(1).size(), 2U);
    EXPECT_EQ(logs.count("error", "nic_smi device"), 1U);
}

TEST_F(nic_smi_poller_test_interface, release_drops_the_nics)
{
    const auto backend = make_backend();
    EXPECT_CALL(*backend, get_nic_rdma_port_statistics(_)).WillOnce(Return(rdma_stats()));
    ASSERT_TRUE(describe({ std::make_shared<device_t>(backend, k_first_smi_index) }));

    m_poller->release();
    m_poller->poll(k_timestamp, m_sink);

    EXPECT_TRUE(m_sink.batches().empty());
}
}  // namespace
}  // namespace rocprofsys::pmc::pollers
