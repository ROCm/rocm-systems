// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/nic/device.hpp"
#include "library/pmc/collectors/nic/tests/mock_nic_driver.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>

using namespace rocprofsys::pmc::collectors::nic;
using ::testing::AtLeast;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Throw;

using MockDriver = StrictMock<rocprofsys::pmc::collectors::nic::testing::mock_nic_driver>;

namespace rocprofsys::pmc::collectors::nic::testing
{

/**
 * @brief Test fixture for NIC device tests.
 */
class NicDeviceTest : public ::testing::Test
{
protected:
    std::shared_ptr<MockDriver> mock_driver;
    size_t                      test_index = 0;

    void SetUp() override { mock_driver = std::make_shared<MockDriver>(); }

    void SetupBaseNicInfo()
    {
        EXPECT_CALL(*mock_driver, get_nic_asic_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(asic_info{ "AMD AINIC Test", "AMD" }));

        EXPECT_CALL(*mock_driver, get_nic_port_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(port_info{ "enp226s0" }));

        EXPECT_CALL(*mock_driver, get_nic_rdma_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(rdma_info{ 1 }));
    }

    void SetupFullRdmaSupport()
    {
        SetupBaseNicInfo();

        EXPECT_CALL(*mock_driver, get_nic_rdma_port_statistics(0))
            .Times(AtLeast(1))
            .WillRepeatedly(Return(std::vector<stat_entry>{
                { "rx_rdma_ucast_bytes", 0 },
                { "tx_rdma_ucast_bytes", 0 },
                { "rx_rdma_ucast_pkts", 0 },
                { "tx_rdma_ucast_pkts", 0 },
                { "rx_rdma_cnp_pkts", 0 },
                { "tx_rdma_cnp_pkts", 0 },
                { "tx_rdma_ack_timeout", 0 },
                { "resp_tx_pkt_seq_err", 0 },
                { "req_rx_pkt_seq_err", 0 },
                { "req_rx_impl_nak_seq_err", 0 },
            }));
    }

    void SetupStatisticsData()
    {
        SetupBaseNicInfo();

        EXPECT_CALL(*mock_driver, get_nic_rdma_port_statistics(0))
            .Times(AtLeast(1))
            .WillRepeatedly(Return(std::vector<stat_entry>{
                { "rx_rdma_ucast_bytes", 1000000 },
                { "tx_rdma_ucast_bytes", 2000000 },
                { "rx_rdma_ucast_pkts", 5000 },
                { "tx_rdma_ucast_pkts", 6000 },
                { "rx_rdma_cnp_pkts", 100 },
                { "tx_rdma_cnp_pkts", 200 },
                { "tx_rdma_ack_timeout", 50 },
                { "resp_tx_pkt_seq_err", 150 },
                { "req_rx_pkt_seq_err", 250 },
                { "req_rx_impl_nak_seq_err", 350 },
            }));
    }

    /**
     * @brief Configure mock to simulate no RDMA support.
     */
    void SetupNoRdmaSupport()
    {
        EXPECT_CALL(*mock_driver, get_nic_asic_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(asic_info{ "Generic NIC", "Unknown" }));

        EXPECT_CALL(*mock_driver, get_nic_port_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(port_info{ "eth0" }));

        EXPECT_CALL(*mock_driver, get_nic_rdma_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Throw(std::runtime_error("get_nic_rdma_dev_info failed: 2")));
    }
};

TEST_F(NicDeviceTest, DeviceIsSupported_WhenRdmaAvailable)
{
    SetupFullRdmaSupport();

    device<MockDriver> dev(mock_driver, test_index);

    EXPECT_TRUE(dev.is_supported());
    EXPECT_EQ(dev.get_index(), test_index);
    EXPECT_EQ(dev.get_name(), "enp226s0");
    EXPECT_EQ(dev.get_product_name(), "AMD AINIC Test");
    EXPECT_EQ(dev.get_vendor_name(), "AMD");
}

TEST_F(NicDeviceTest, DeviceIsNotSupported_WhenNoRdma)
{
    SetupNoRdmaSupport();

    device<MockDriver> dev(mock_driver, test_index);

    EXPECT_FALSE(dev.is_supported());
}

TEST_F(NicDeviceTest, GetSupportedMetrics_AllEnabled)
{
    SetupFullRdmaSupport();

    device<MockDriver> dev(mock_driver, test_index);
    auto               supported = dev.get_supported_metrics();

    EXPECT_TRUE(supported.bits.rx_rdma_ucast_bytes);
    EXPECT_TRUE(supported.bits.tx_rdma_ucast_bytes);
    EXPECT_TRUE(supported.bits.rx_rdma_ucast_pkts);
    EXPECT_TRUE(supported.bits.tx_rdma_ucast_pkts);
    EXPECT_TRUE(supported.bits.rx_rdma_cnp_pkts);
    EXPECT_TRUE(supported.bits.tx_rdma_cnp_pkts);
    EXPECT_TRUE(supported.bits.tx_rdma_ack_timeout);
    EXPECT_TRUE(supported.bits.resp_tx_pkt_seq_err);
    EXPECT_TRUE(supported.bits.req_rx_pkt_seq_err);
    EXPECT_TRUE(supported.bits.req_rx_impl_nak_seq_err);
}

TEST_F(NicDeviceTest, GetNicMetrics_ReturnsCorrectValues)
{
    SetupStatisticsData();

    device<MockDriver> dev(mock_driver, test_index);
    auto               m = dev.get_nic_metrics();

    EXPECT_EQ(m.rx_rdma_ucast_bytes, 1000000ULL);
    EXPECT_EQ(m.tx_rdma_ucast_bytes, 2000000ULL);
    EXPECT_EQ(m.rx_rdma_ucast_pkts, 5000ULL);
    EXPECT_EQ(m.tx_rdma_ucast_pkts, 6000ULL);
    EXPECT_EQ(m.rx_rdma_cnp_pkts, 100ULL);
    EXPECT_EQ(m.tx_rdma_cnp_pkts, 200ULL);
    EXPECT_EQ(m.tx_rdma_ack_timeout, 50ULL);
    EXPECT_EQ(m.resp_tx_pkt_seq_err, 150ULL);
    EXPECT_EQ(m.req_rx_pkt_seq_err, 250ULL);
    EXPECT_EQ(m.req_rx_impl_nak_seq_err, 350ULL);
}

TEST_F(NicDeviceTest, GetNicMetrics_ReturnsZeros_WhenNoRdmaPorts)
{
    EXPECT_CALL(*mock_driver, get_nic_asic_info())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(asic_info{ "Test NIC", "Test Vendor" }));

    EXPECT_CALL(*mock_driver, get_nic_port_info())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(port_info{ "enp226s0" }));

    EXPECT_CALL(*mock_driver, get_nic_rdma_info())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(rdma_info{ 0 }));

    device<MockDriver> dev(mock_driver, test_index);

    EXPECT_FALSE(dev.is_supported());

    auto m = dev.get_nic_metrics();
    EXPECT_EQ(m.rx_rdma_ucast_bytes, 0ULL);
    EXPECT_EQ(m.tx_rdma_ucast_bytes, 0ULL);
}

TEST_F(NicDeviceTest, GetNicMetrics_ReturnsZeros_WhenStatisticsQueryThrows)
{
    EXPECT_CALL(*mock_driver, get_nic_asic_info())
        .WillOnce(Return(asic_info{ "AMD AINIC Test", "AMD" }));

    EXPECT_CALL(*mock_driver, get_nic_port_info())
        .WillOnce(Return(port_info{ "enp226s0" }));

    EXPECT_CALL(*mock_driver, get_nic_rdma_info()).WillOnce(Return(rdma_info{ 1 }));

    EXPECT_CALL(*mock_driver, get_nic_rdma_port_statistics(0))
        .WillOnce(Return(std::vector<stat_entry>{
            { "rx_rdma_ucast_bytes", 0 },
            { "tx_rdma_ucast_bytes", 0 },
            { "rx_rdma_ucast_pkts", 0 },
            { "tx_rdma_ucast_pkts", 0 },
            { "rx_rdma_cnp_pkts", 0 },
            { "tx_rdma_cnp_pkts", 0 },
            { "tx_rdma_ack_timeout", 0 },
            { "resp_tx_pkt_seq_err", 0 },
            { "req_rx_pkt_seq_err", 0 },
            { "req_rx_impl_nak_seq_err", 0 },
        }))
        .WillOnce(Throw(std::runtime_error("stats query failed")));

    device<MockDriver> dev(mock_driver, test_index);
    EXPECT_TRUE(dev.is_supported());

    auto m = dev.get_nic_metrics();
    EXPECT_EQ(m.rx_rdma_ucast_bytes, 0ULL);
    EXPECT_EQ(m.tx_rdma_ucast_bytes, 0ULL);
    EXPECT_EQ(m.rx_rdma_ucast_pkts, 0ULL);
    EXPECT_EQ(m.tx_rdma_ucast_pkts, 0ULL);
    EXPECT_EQ(m.rx_rdma_cnp_pkts, 0ULL);
    EXPECT_EQ(m.tx_rdma_cnp_pkts, 0ULL);
    EXPECT_EQ(m.tx_rdma_ack_timeout, 0ULL);
    EXPECT_EQ(m.resp_tx_pkt_seq_err, 0ULL);
    EXPECT_EQ(m.req_rx_pkt_seq_err, 0ULL);
    EXPECT_EQ(m.req_rx_impl_nak_seq_err, 0ULL);
}

TEST_F(NicDeviceTest, GetNicMetrics_IgnoresUnknownStatNames)
{
    SetupBaseNicInfo();

    EXPECT_CALL(*mock_driver, get_nic_rdma_port_statistics(0))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<stat_entry>{
            { "rx_rdma_ucast_bytes", 1000 },
            { "unknown_stat_1", 9999 },
            { "tx_rdma_ucast_bytes", 2000 },
            { "some_other_counter", 8888 },
        }));

    device<MockDriver> dev(mock_driver, test_index);
    auto               m = dev.get_nic_metrics();

    EXPECT_EQ(m.rx_rdma_ucast_bytes, 1000ULL);
    EXPECT_EQ(m.tx_rdma_ucast_bytes, 2000ULL);
    EXPECT_EQ(m.rx_rdma_ucast_pkts, 0ULL);
    EXPECT_EQ(m.tx_rdma_ucast_pkts, 0ULL);
    EXPECT_EQ(m.rx_rdma_cnp_pkts, 0ULL);
    EXPECT_EQ(m.tx_rdma_cnp_pkts, 0ULL);
    EXPECT_EQ(m.tx_rdma_ack_timeout, 0ULL);
    EXPECT_EQ(m.resp_tx_pkt_seq_err, 0ULL);
    EXPECT_EQ(m.req_rx_pkt_seq_err, 0ULL);
    EXPECT_EQ(m.req_rx_impl_nak_seq_err, 0ULL);
}

TEST_F(NicDeviceTest, GetNicMetrics_HandlesPartialStats)
{
    SetupBaseNicInfo();

    EXPECT_CALL(*mock_driver, get_nic_rdma_port_statistics(0))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<stat_entry>{
            { "rx_rdma_ucast_bytes", 500 },
            { "tx_rdma_cnp_pkts", 10 },
        }));

    device<MockDriver> dev(mock_driver, test_index);
    auto               m = dev.get_nic_metrics();

    EXPECT_EQ(m.rx_rdma_ucast_bytes, 500ULL);
    EXPECT_EQ(m.tx_rdma_cnp_pkts, 10ULL);
    EXPECT_EQ(m.tx_rdma_ucast_bytes, 0ULL);
    EXPECT_EQ(m.rx_rdma_ucast_pkts, 0ULL);
    EXPECT_EQ(m.tx_rdma_ucast_pkts, 0ULL);
    EXPECT_EQ(m.rx_rdma_cnp_pkts, 0ULL);
    EXPECT_EQ(m.tx_rdma_ack_timeout, 0ULL);
    EXPECT_EQ(m.resp_tx_pkt_seq_err, 0ULL);
    EXPECT_EQ(m.req_rx_pkt_seq_err, 0ULL);
    EXPECT_EQ(m.req_rx_impl_nak_seq_err, 0ULL);
}

TEST_F(NicDeviceTest, DeviceInitializes_WhenAsicInfoThrows)
{
    EXPECT_CALL(*mock_driver, get_nic_asic_info())
        .WillOnce(Throw(std::runtime_error("get_nic_asic_info failed")));

    EXPECT_CALL(*mock_driver, get_nic_port_info())
        .WillOnce(Return(port_info{ "enp226s0" }));

    EXPECT_CALL(*mock_driver, get_nic_rdma_info()).WillOnce(Return(rdma_info{ 1 }));

    EXPECT_CALL(*mock_driver, get_nic_rdma_port_statistics(0))
        .WillOnce(Return(std::vector<stat_entry>{
            { "rx_rdma_ucast_bytes", 0 },
        }));

    device<MockDriver> dev(mock_driver, test_index);

    EXPECT_TRUE(dev.is_supported());
    EXPECT_TRUE(dev.get_product_name().empty());
    EXPECT_TRUE(dev.get_vendor_name().empty());
    EXPECT_EQ(dev.get_name(), "enp226s0");
}

TEST_F(NicDeviceTest, DeviceInitializes_WhenPortInfoThrows)
{
    EXPECT_CALL(*mock_driver, get_nic_asic_info())
        .WillOnce(Return(asic_info{ "AMD AINIC Test", "AMD" }));

    EXPECT_CALL(*mock_driver, get_nic_port_info())
        .WillOnce(Throw(std::runtime_error("get_nic_port_info failed")));

    EXPECT_CALL(*mock_driver, get_nic_rdma_info()).WillOnce(Return(rdma_info{ 1 }));

    EXPECT_CALL(*mock_driver, get_nic_rdma_port_statistics(0))
        .WillOnce(Return(std::vector<stat_entry>{
            { "rx_rdma_ucast_bytes", 0 },
        }));

    device<MockDriver> dev(mock_driver, test_index);

    EXPECT_TRUE(dev.is_supported());
    EXPECT_TRUE(dev.get_name().empty());
    EXPECT_EQ(dev.get_product_name(), "AMD AINIC Test");
    EXPECT_EQ(dev.get_vendor_name(), "AMD");
}

TEST_F(NicDeviceTest, DeviceNotSupported_WhenStatsEmpty)
{
    SetupBaseNicInfo();

    EXPECT_CALL(*mock_driver, get_nic_rdma_port_statistics(0))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<stat_entry>{}));

    device<MockDriver> dev(mock_driver, test_index);

    EXPECT_FALSE(dev.is_supported());
}

}  // namespace rocprofsys::pmc::collectors::nic::testing
