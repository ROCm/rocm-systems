// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "library/pmc/collectors/nic/device.hpp"
#include "library/pmc/device_providers/amd_smi/drivers/tests/mock_driver.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>

using namespace rocprofsys::pmc::collectors::nic;
using ::testing::_;
using ::testing::AtLeast;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

using MockDriver =
    ::testing::StrictMock<rocprofsys::pmc::drivers::amd_smi::testing::mock_driver>;

namespace rocprofsys::pmc::collectors::nic::testing
{

/**
 * @brief Test fixture for NIC device tests.
 *
 * Provides common setup for device tests including mock driver and
 * helper methods for configuring mock behavior.
 */
class NicDeviceTest : public ::testing::Test
{
protected:
    std::shared_ptr<MockDriver> mock_driver;
    amdsmi_processor_handle     test_handle;
    processor_type_t            test_processor_type;
    size_t                      test_index;

    void SetUp() override
    {
        mock_driver         = std::make_shared<MockDriver>();
        test_handle         = reinterpret_cast<amdsmi_processor_handle>(0x5678);
        test_processor_type = AMDSMI_PROCESSOR_TYPE_AMD_NIC;
        test_index          = 0;
    }

    /**
     * @brief Configure mock to return NIC with full RDMA support.
     */
    void SetupFullRdmaSupport()
    {
        // Setup ASIC info for vendor and product names
        amdsmi_nic_asic_info_t asic_info{};
        std::strncpy(asic_info.product_name, "AMD AINIC Test",
                     sizeof(asic_info.product_name) - 1);
        std::strncpy(asic_info.vendor_name, "AMD", sizeof(asic_info.vendor_name) - 1);

        EXPECT_CALL(*mock_driver, get_nic_asic_info(test_handle, _))
            .Times(AtLeast(1))
            .WillRepeatedly(
                DoAll(SetArgPointee<1>(asic_info), Return(AMDSMI_STATUS_SUCCESS)));

        // Setup port info
        amdsmi_nic_port_info_t port_info{};
        port_info.num_ports = 1;
        std::strncpy(port_info.ports[0].netdev, "enp226s0",
                     sizeof(port_info.ports[0].netdev) - 1);

        EXPECT_CALL(*mock_driver, get_nic_port_info(test_handle, _))
            .Times(AtLeast(1))
            .WillRepeatedly(
                DoAll(SetArgPointee<1>(port_info), Return(AMDSMI_STATUS_SUCCESS)));

        // Setup RDMA device info
        amdsmi_nic_rdma_devices_info_t rdma_info{};
        rdma_info.num_rdma_dev                    = 1;
        rdma_info.rdma_dev_info[0].num_rdma_ports = 1;
        std::strncpy(rdma_info.rdma_dev_info[0].rdma_dev, "rdma0",
                     sizeof(rdma_info.rdma_dev_info[0].rdma_dev) - 1);

        EXPECT_CALL(*mock_driver, get_nic_rdma_dev_info(test_handle, _))
            .Times(AtLeast(1))
            .WillRepeatedly(
                DoAll(SetArgPointee<1>(rdma_info), Return(AMDSMI_STATUS_SUCCESS)));

        // Setup statistics count query
        EXPECT_CALL(*mock_driver,
                    get_nic_rdma_port_statistics(test_handle, 0, _, nullptr))
            .Times(AtLeast(1))
            .WillRepeatedly([](amdsmi_processor_handle, uint8_t, uint32_t* count,
                               amdsmi_nic_stat_t*) {
                *count = 6;
                return AMDSMI_STATUS_SUCCESS;
            });
    }

    /**
     * @brief Configure mock to return full statistics.
     */
    void SetupStatisticsData()
    {
        SetupFullRdmaSupport();

        EXPECT_CALL(*mock_driver,
                    get_nic_rdma_port_statistics(test_handle, 0, _, ::testing::NotNull()))
            .Times(AtLeast(1))
            .WillRepeatedly([](amdsmi_processor_handle, uint8_t, uint32_t* count,
                               amdsmi_nic_stat_t* stats) {
                *count = 6;

                std::strncpy(stats[0].name, "rx_rdma_ucast_bytes",
                             sizeof(stats[0].name) - 1);
                stats[0].value = 1000000;

                std::strncpy(stats[1].name, "tx_rdma_ucast_bytes",
                             sizeof(stats[1].name) - 1);
                stats[1].value = 2000000;

                std::strncpy(stats[2].name, "rx_rdma_ucast_pkts",
                             sizeof(stats[2].name) - 1);
                stats[2].value = 5000;

                std::strncpy(stats[3].name, "tx_rdma_ucast_pkts",
                             sizeof(stats[3].name) - 1);
                stats[3].value = 6000;

                std::strncpy(stats[4].name, "rx_rdma_cnp_pkts",
                             sizeof(stats[4].name) - 1);
                stats[4].value = 100;

                std::strncpy(stats[5].name, "tx_rdma_cnp_pkts",
                             sizeof(stats[5].name) - 1);
                stats[5].value = 200;

                return AMDSMI_STATUS_SUCCESS;
            });
    }

    /**
     * @brief Configure mock to return no RDMA support.
     */
    void SetupNoRdmaSupport()
    {
        // Setup ASIC info
        amdsmi_nic_asic_info_t asic_info{};
        std::strncpy(asic_info.product_name, "Generic NIC",
                     sizeof(asic_info.product_name) - 1);
        std::strncpy(asic_info.vendor_name, "Unknown", sizeof(asic_info.vendor_name) - 1);

        EXPECT_CALL(*mock_driver, get_nic_asic_info(test_handle, _))
            .Times(AtLeast(1))
            .WillRepeatedly(
                DoAll(SetArgPointee<1>(asic_info), Return(AMDSMI_STATUS_SUCCESS)));

        // Setup port info (NIC exists but no RDMA)
        amdsmi_nic_port_info_t port_info{};
        port_info.num_ports = 1;
        std::strncpy(port_info.ports[0].netdev, "eth0",
                     sizeof(port_info.ports[0].netdev) - 1);

        EXPECT_CALL(*mock_driver, get_nic_port_info(test_handle, _))
            .Times(AtLeast(1))
            .WillRepeatedly(
                DoAll(SetArgPointee<1>(port_info), Return(AMDSMI_STATUS_SUCCESS)));

        // RDMA query fails
        EXPECT_CALL(*mock_driver, get_nic_rdma_dev_info(test_handle, _))
            .Times(AtLeast(1))
            .WillRepeatedly(Return(AMDSMI_STATUS_NOT_SUPPORTED));
    }
};

TEST_F(NicDeviceTest, DeviceIsSupported_WhenRdmaAvailable)
{
    SetupFullRdmaSupport();

    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    EXPECT_TRUE(dev.is_supported());
    EXPECT_EQ(dev.get_index(), test_index);
    EXPECT_EQ(dev.get_name(), "enp226s0");
    EXPECT_EQ(dev.get_product_name(), "AMD AINIC Test");
    EXPECT_EQ(dev.get_vendor_name(), "AMD");
}

TEST_F(NicDeviceTest, DeviceIsNotSupported_WhenNoRdma)
{
    SetupNoRdmaSupport();

    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    EXPECT_FALSE(dev.is_supported());
}

TEST_F(NicDeviceTest, GetSupportedMetrics_AllEnabled)
{
    SetupFullRdmaSupport();

    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);
    auto               supported = dev.get_supported_metrics();

    EXPECT_TRUE(supported.bits.rx_rdma_ucast_bytes);
    EXPECT_TRUE(supported.bits.tx_rdma_ucast_bytes);
    EXPECT_TRUE(supported.bits.rx_rdma_ucast_pkts);
    EXPECT_TRUE(supported.bits.tx_rdma_ucast_pkts);
    EXPECT_TRUE(supported.bits.rx_rdma_cnp_pkts);
    EXPECT_TRUE(supported.bits.tx_rdma_cnp_pkts);
}

TEST_F(NicDeviceTest, GetNicMetrics_ReturnsCorrectValues)
{
    SetupStatisticsData();

    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);
    auto               m = dev.get_nic_metrics();

    EXPECT_EQ(m.rx_rdma_ucast_bytes, 1000000ULL);
    EXPECT_EQ(m.tx_rdma_ucast_bytes, 2000000ULL);
    EXPECT_EQ(m.rx_rdma_ucast_pkts, 5000ULL);
    EXPECT_EQ(m.tx_rdma_ucast_pkts, 6000ULL);
    EXPECT_EQ(m.rx_rdma_cnp_pkts, 100ULL);
    EXPECT_EQ(m.tx_rdma_cnp_pkts, 200ULL);
}

TEST_F(NicDeviceTest, GetNicMetrics_ReturnsZeros_WhenNoRdmaPorts)
{
    // Setup ASIC info
    amdsmi_nic_asic_info_t asic_info{};
    std::strncpy(asic_info.product_name, "Test NIC", sizeof(asic_info.product_name) - 1);
    std::strncpy(asic_info.vendor_name, "Test Vendor", sizeof(asic_info.vendor_name) - 1);

    EXPECT_CALL(*mock_driver, get_nic_asic_info(test_handle, _))
        .Times(AtLeast(1))
        .WillRepeatedly(
            DoAll(SetArgPointee<1>(asic_info), Return(AMDSMI_STATUS_SUCCESS)));

    // Setup with RDMA device but no ports
    amdsmi_nic_port_info_t port_info{};
    port_info.num_ports = 1;
    std::strncpy(port_info.ports[0].netdev, "enp226s0",
                 sizeof(port_info.ports[0].netdev) - 1);

    EXPECT_CALL(*mock_driver, get_nic_port_info(test_handle, _))
        .Times(AtLeast(1))
        .WillRepeatedly(
            DoAll(SetArgPointee<1>(port_info), Return(AMDSMI_STATUS_SUCCESS)));

    amdsmi_nic_rdma_devices_info_t rdma_info{};
    rdma_info.num_rdma_dev                    = 1;
    rdma_info.rdma_dev_info[0].num_rdma_ports = 0;  // No ports

    EXPECT_CALL(*mock_driver, get_nic_rdma_dev_info(test_handle, _))
        .Times(AtLeast(1))
        .WillRepeatedly(
            DoAll(SetArgPointee<1>(rdma_info), Return(AMDSMI_STATUS_SUCCESS)));

    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    EXPECT_FALSE(dev.is_supported());

    auto m = dev.get_nic_metrics();
    EXPECT_EQ(m.rx_rdma_ucast_bytes, 0ULL);
    EXPECT_EQ(m.tx_rdma_ucast_bytes, 0ULL);
}

}  // namespace rocprofsys::pmc::collectors::nic::testing
