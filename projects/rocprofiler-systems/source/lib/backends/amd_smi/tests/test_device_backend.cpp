// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/amd_smi/backend.hpp"
#include "backends/amd_smi/device.hpp"
#include "mock_wrapper.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::DoAll;
using ::testing::HasSubstr;
using ::testing::InSequence;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::SetArrayArgument;
using ::testing::StrictMock;

namespace rocprofsys::backends::amd_smi
{

using MockApi   = StrictMock<testing::gmock_backend_api>;
using factory_t = backend_factory<testing::mock_backend>;
using sut_t     = factory_t::backend_t;
using DeviceSut = device<sut_t>;

constexpr testing::mock_status_t k_ok     = testing::mock_backend::STATUS_SUCCESS;
constexpr testing::mock_status_t k_err    = 1;
constexpr std::uint64_t          k_handle = 0xABCDABCDULL;

class DeviceBackendTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        testing::g_mock_backend = std::make_unique<MockApi>();
        m_session               = std::make_shared<sut_t>();

        // get_metrics() always probes SDMA support via get_gpu_process_list.
        // Tests that verify specific SDMA behaviour override this with their own
        // EXPECT_CALL.
        EXPECT_CALL(*testing::g_mock_backend,
                    get_gpu_process_list(_, NotNull(), IsNull()))
            .Times(AnyNumber())
            .WillRepeatedly(Return(k_ok));
    }

    void TearDown() override { testing::g_mock_backend.reset(); }

    std::shared_ptr<sut_t> m_session;
};

// ── get_gpu_asic_info ─────────────────────────────────────────────────────────

TEST_F(DeviceBackendTest, get_gpu_asic_info_maps_market_name_to_product_name)
{
    const testing::mock_asic_info_t raw{ .market_name = "Radeon RX 7900 XTX",
                                         .vendor_name = "AMD" };
    EXPECT_CALL(*testing::g_mock_backend, get_gpu_asic_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto info = sut.get_gpu_asic_info();
    EXPECT_EQ(info.product_name, "Radeon RX 7900 XTX");
    EXPECT_EQ(info.vendor_name, "AMD");
}

TEST_F(DeviceBackendTest, get_gpu_asic_info_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_gpu_asic_info(k_handle, _))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_gpu_asic_info()), std::runtime_error);
}

TEST_F(DeviceBackendTest, get_gpu_asic_info_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_gpu_asic_info(k_handle, _))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(
        {
            try
            {
                static_cast<void>(sut.get_gpu_asic_info());
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_gpu_asic_info"));
                throw;
            }
        },
        std::runtime_error);
}

// ── get_metrics — power ───────────────────────────────────────────────────

TEST_F(DeviceBackendTest, get_metrics_maps_power_fields)
{
    testing::mock_gpu_metrics_t raw{};
    raw.current_socket_power = 150;
    raw.average_socket_power = 120;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.current_socket_power, 150U);
    EXPECT_EQ(m.average_socket_power, 120U);
}

// ── get_metrics — temperature ────────────────────────────────────────────

TEST_F(DeviceBackendTest, get_metrics_maps_temperature_fields)
{
    testing::mock_gpu_metrics_t raw{};
    raw.temperature_hotspot = 85;
    raw.temperature_edge    = 72;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.hotspot_temperature, 85U);
    EXPECT_EQ(m.edge_temperature, 72U);
}

// ── get_metrics — activity ────────────────────────────────────────────────

TEST_F(DeviceBackendTest, get_metrics_maps_activity_fields)
{
    testing::mock_gpu_metrics_t raw{};
    raw.average_gfx_activity = 80;
    raw.average_umc_activity = 40;
    raw.average_mm_activity  = 60;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.gfx_activity, 80U);
    EXPECT_EQ(m.umc_activity, 40U);
    EXPECT_EQ(m.mm_activity, 60U);
}

// ── get_metrics — PCIe ────────────────────────────────────────────────────

TEST_F(DeviceBackendTest, get_metrics_pcie_fields_zeroed_when_sentinel)
{
    // mock_gpu_metrics_t default-initialises all pcie fields to sentinel values
    const testing::mock_gpu_metrics_t raw{};

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.pcie.link.width, 0U);
    EXPECT_EQ(m.pcie.link.speed, 0U);
    EXPECT_EQ(m.pcie.bandwidth.acc, 0U);
    EXPECT_EQ(m.pcie.bandwidth.inst, 0U);
}

TEST_F(DeviceBackendTest, get_metrics_pcie_fields_populated_when_supported)
{
    testing::mock_gpu_metrics_t raw{};
    raw.pcie_link_width     = 16;
    raw.pcie_link_speed     = 32;
    raw.pcie_bandwidth_acc  = 10000;
    raw.pcie_bandwidth_inst = 9000;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.pcie.link.width, 16U);
    EXPECT_EQ(m.pcie.link.speed, 32U);
    EXPECT_EQ(m.pcie.bandwidth.acc, 10000U);
    EXPECT_EQ(m.pcie.bandwidth.inst, 9000U);
}

// ── get_metrics — XGMI ───────────────────────────────────────────────────

TEST_F(DeviceBackendTest, get_metrics_xgmi_link_zeroed_when_sentinel)
{
    // xgmi_link_width and xgmi_link_speed default to 0xFFFF in mock_gpu_metrics_t
    const testing::mock_gpu_metrics_t raw{};

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.xgmi.link.width, 0U);
    EXPECT_EQ(m.xgmi.link.speed, 0U);
}

TEST_F(DeviceBackendTest, get_metrics_xgmi_link_populated_when_supported)
{
    testing::mock_gpu_metrics_t raw{};
    raw.xgmi_link_width = 2;
    raw.xgmi_link_speed = 25;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.xgmi.link.width, 2U);
    EXPECT_EQ(m.xgmi.link.speed, 25U);
}

TEST_F(DeviceBackendTest, get_metrics_xgmi_data_acc_populated_when_supported)
{
    testing::mock_gpu_metrics_t raw{};
    raw.xgmi_read_data_acc[0]  = 1000;
    raw.xgmi_write_data_acc[0] = 2000;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.xgmi.data_acc.read[0], 1000U);
    EXPECT_EQ(m.xgmi.data_acc.write[0], 2000U);
}

// ── get_metrics — clocks ─────────────────────────────────────────────────

TEST_F(DeviceBackendTest, get_metrics_clocks_zeroed_when_sentinel)
{
    // current_gfxclk and current_uclk default to 0xFFFF in mock_gpu_metrics_t
    const testing::mock_gpu_metrics_t raw{};

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.gfx_clock_mhz, 0U);
    EXPECT_EQ(m.mem_clock_mhz, 0U);
}

TEST_F(DeviceBackendTest, get_metrics_clocks_populated_when_supported)
{
    testing::mock_gpu_metrics_t raw{};
    raw.current_gfxclk = 2100;
    raw.current_uclk   = 1200;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.gfx_clock_mhz, 2100U);
    EXPECT_EQ(m.mem_clock_mhz, 1200U);
}

// ── get_metrics — VCN / JPEG activity ────────────────────────────────────

TEST_F(DeviceBackendTest, get_metrics_vcn_activity_is_copied)
{
    testing::mock_gpu_metrics_t raw{};
    raw.vcn_activity[0] = 55;
    raw.vcn_activity[1] = 77;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.vcn_activity[0], 55U);
    EXPECT_EQ(m.vcn_activity[1], 77U);
}

TEST_F(DeviceBackendTest, get_metrics_jpeg_activity_is_copied)
{
    testing::mock_gpu_metrics_t raw{};
    raw.jpeg_activity[0]  = 11;
    raw.jpeg_activity[31] = 22;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.jpeg_activity[0], 11U);
    EXPECT_EQ(m.jpeg_activity[31], 22U);
}

TEST_F(DeviceBackendTest, get_metrics_xcp_vcn_busy_is_copied)
{
    testing::mock_gpu_metrics_t raw{};
    raw.xcp_stats[0].vcn_busy[0] = 33;
    raw.xcp_stats[0].vcn_busy[3] = 44;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.xcp_stats[0].vcn_busy[0], 33U);
    EXPECT_EQ(m.xcp_stats[0].vcn_busy[3], 44U);
}

TEST_F(DeviceBackendTest, get_metrics_xcp_jpeg_busy_is_copied)
{
    testing::mock_gpu_metrics_t raw{};
    raw.xcp_stats[0].jpeg_busy[0]  = 55;
    raw.xcp_stats[0].jpeg_busy[39] = 66;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto m = sut.get_metrics();
    EXPECT_EQ(m.xcp_stats[0].jpeg_busy[0], 55U);
    EXPECT_EQ(m.xcp_stats[0].jpeg_busy[39], 66U);
}

// ── get_metrics — error handling ─────────────────────────────────────────

TEST_F(DeviceBackendTest, get_metrics_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, _))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_metrics()), std::runtime_error);
}

TEST_F(DeviceBackendTest, get_metrics_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, _))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(
        {
            try
            {
                static_cast<void>(sut.get_metrics());
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_gpu_metrics_info"));
                throw;
            }
        },
        std::runtime_error);
}

// ── get_memory_usage ──────────────────────────────────────────────────────────

TEST_F(DeviceBackendTest, get_memory_usage_returns_reported_bytes)
{
    constexpr std::uint64_t k_bytes = 8ULL * 1024 * 1024 * 1024;
    EXPECT_CALL(*testing::g_mock_backend, get_memory_usage(k_handle, _, NotNull()))
        .WillOnce(DoAll(SetArgPointee<2>(k_bytes), Return(k_ok)));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_EQ(sut.get_memory_usage(), k_bytes);
}

TEST_F(DeviceBackendTest, get_memory_usage_passes_vram_type)
{
    constexpr auto k_vram = testing::mock_backend::MEM_TYPE_VRAM;
    EXPECT_CALL(*testing::g_mock_backend, get_memory_usage(k_handle, k_vram, _))
        .WillOnce(DoAll(SetArgPointee<2>(0U), Return(k_ok)));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_EQ(sut.get_memory_usage(), 0U);
}

TEST_F(DeviceBackendTest, get_memory_usage_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_memory_usage(k_handle, _, _))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_memory_usage()), std::runtime_error);
}

TEST_F(DeviceBackendTest, get_memory_usage_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_memory_usage(k_handle, _, _))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(
        {
            try
            {
                static_cast<void>(sut.get_memory_usage());
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_gpu_memory_usage"));
                throw;
            }
        },
        std::runtime_error);
}

// ── get_hotspot_temperature / get_edge_temperature ───────────────────────────

TEST_F(DeviceBackendTest, get_hotspot_temperature_returns_reported_value)
{
    constexpr std::int64_t k_temp = 85000;
    EXPECT_CALL(*testing::g_mock_backend,
                get_temp_metric(k_handle, testing::mock_backend::TEMPERATURE_TYPE_HOTSPOT,
                                testing::mock_backend::TEMP_CURRENT, NotNull()))
        .WillOnce(DoAll(SetArgPointee<3>(k_temp), Return(k_ok)));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_EQ(sut.get_hotspot_temperature(), k_temp);
}

TEST_F(DeviceBackendTest, get_hotspot_temperature_passes_hotspot_sensor_type)
{
    constexpr auto k_hotspot = testing::mock_backend::TEMPERATURE_TYPE_HOTSPOT;
    constexpr auto k_edge    = testing::mock_backend::TEMPERATURE_TYPE_EDGE;
    static_assert(k_hotspot != k_edge, "test requires distinct sensor type values");

    EXPECT_CALL(*testing::g_mock_backend,
                get_temp_metric(k_handle, k_hotspot, _, NotNull()))
        .WillOnce(DoAll(SetArgPointee<3>(std::int64_t{ 0 }), Return(k_ok)));

    DeviceSut sut{ m_session, k_handle };
    static_cast<void>(sut.get_hotspot_temperature());
}

TEST_F(DeviceBackendTest, get_hotspot_temperature_throws_on_backend_error)
{
    EXPECT_CALL(
        *testing::g_mock_backend,
        get_temp_metric(k_handle, testing::mock_backend::TEMPERATURE_TYPE_HOTSPOT, _, _))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_hotspot_temperature()), std::runtime_error);
}

TEST_F(DeviceBackendTest, get_edge_temperature_returns_reported_value)
{
    constexpr std::int64_t k_temp = 72000;
    EXPECT_CALL(*testing::g_mock_backend,
                get_temp_metric(k_handle, testing::mock_backend::TEMPERATURE_TYPE_EDGE,
                                testing::mock_backend::TEMP_CURRENT, NotNull()))
        .WillOnce(DoAll(SetArgPointee<3>(k_temp), Return(k_ok)));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_EQ(sut.get_edge_temperature(), k_temp);
}

TEST_F(DeviceBackendTest, get_edge_temperature_passes_edge_sensor_type)
{
    constexpr auto k_edge = testing::mock_backend::TEMPERATURE_TYPE_EDGE;

    EXPECT_CALL(*testing::g_mock_backend, get_temp_metric(k_handle, k_edge, _, NotNull()))
        .WillOnce(DoAll(SetArgPointee<3>(std::int64_t{ 0 }), Return(k_ok)));

    DeviceSut sut{ m_session, k_handle };
    static_cast<void>(sut.get_edge_temperature());
}

TEST_F(DeviceBackendTest, get_edge_temperature_throws_on_backend_error)
{
    EXPECT_CALL(
        *testing::g_mock_backend,
        get_temp_metric(k_handle, testing::mock_backend::TEMPERATURE_TYPE_EDGE, _, _))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_edge_temperature()), std::runtime_error);
}

// ── NIC methods ──────────────────────────────────────────────────────────────

TEST_F(DeviceBackendTest, get_nic_asic_info_maps_product_and_vendor_names)
{
    const testing::mock_nic_asic_info_t raw{ .product_name = "CX7",
                                             .vendor_name  = "Mellanox" };
    EXPECT_CALL(*testing::g_mock_backend, get_nic_asic_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto info = sut.get_nic_asic_info();
    EXPECT_EQ(info.product_name, "CX7");
    EXPECT_EQ(info.vendor_name, "Mellanox");
}

TEST_F(DeviceBackendTest, get_nic_asic_info_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_nic_asic_info(k_handle, _))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_nic_asic_info()), std::runtime_error);
}

TEST_F(DeviceBackendTest, get_nic_port_info_returns_empty_when_no_ports)
{
    testing::mock_nic_port_info_t raw{};
    raw.num_ports = 0;
    EXPECT_CALL(*testing::g_mock_backend, get_nic_port_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto info = sut.get_nic_port_info();
    EXPECT_TRUE(info.device_name.empty());
}

TEST_F(DeviceBackendTest, get_nic_port_info_returns_netdev_for_first_port)
{
    testing::mock_nic_port_info_t raw{};
    raw.num_ports       = 1;
    raw.ports[0].netdev = "mlx5_0";
    EXPECT_CALL(*testing::g_mock_backend, get_nic_port_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto info = sut.get_nic_port_info();
    EXPECT_EQ(info.device_name, "mlx5_0");
}

TEST_F(DeviceBackendTest, get_nic_port_info_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_nic_port_info(k_handle, _))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_nic_port_info()), std::runtime_error);
}

TEST_F(DeviceBackendTest, get_nic_rdma_info_returns_zero_when_no_rdma_devices)
{
    testing::mock_nic_rdma_devices_info_t raw{};
    raw.num_rdma_dev = 0;
    EXPECT_CALL(*testing::g_mock_backend, get_nic_rdma_dev_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto info = sut.get_nic_rdma_info();
    EXPECT_EQ(info.port_count, 0U);
}

TEST_F(DeviceBackendTest, get_nic_rdma_info_returns_port_count)
{
    testing::mock_nic_rdma_devices_info_t raw{};
    raw.num_rdma_dev                    = 1;
    raw.rdma_dev_info[0].num_rdma_ports = 4;
    EXPECT_CALL(*testing::g_mock_backend, get_nic_rdma_dev_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto info = sut.get_nic_rdma_info();
    EXPECT_EQ(info.port_count, 4U);
}

TEST_F(DeviceBackendTest, get_nic_rdma_info_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_nic_rdma_dev_info(k_handle, _))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_nic_rdma_info()), std::runtime_error);
}

TEST_F(DeviceBackendTest, get_nic_rdma_port_statistics_returns_empty_when_count_zero)
{
    InSequence seq;
    EXPECT_CALL(
        *testing::g_mock_backend,
        get_nic_rdma_port_statistics(k_handle, std::uint8_t{ 0 }, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<2>(0U), Return(k_ok)));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_TRUE(sut.get_nic_rdma_port_statistics(0).empty());
}

TEST_F(DeviceBackendTest, get_nic_rdma_port_statistics_maps_name_and_value)
{
    const testing::mock_nic_stat_t stat{ .name = "rx_bytes", .value = 99999 };

    InSequence seq;
    EXPECT_CALL(
        *testing::g_mock_backend,
        get_nic_rdma_port_statistics(k_handle, std::uint8_t{ 0 }, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<2>(1U), Return(k_ok)));
    EXPECT_CALL(
        *testing::g_mock_backend,
        get_nic_rdma_port_statistics(k_handle, std::uint8_t{ 0 }, NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<2>(1U), SetArrayArgument<3>(&stat, &stat + 1),
                        Return(k_ok)));

    DeviceSut  sut{ m_session, k_handle };
    const auto stats = sut.get_nic_rdma_port_statistics(0);
    ASSERT_EQ(stats.size(), 1U);
    EXPECT_EQ(stats[0].name, "rx_bytes");
    EXPECT_EQ(stats[0].value, 99999U);
}

TEST_F(DeviceBackendTest, get_nic_rdma_port_statistics_throws_on_count_error)
{
    EXPECT_CALL(*testing::g_mock_backend,
                get_nic_rdma_port_statistics(k_handle, _, NotNull(), IsNull()))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_nic_rdma_port_statistics(0)),
                 std::runtime_error);
}

TEST_F(DeviceBackendTest, get_nic_rdma_port_statistics_throws_on_data_error)
{
    InSequence seq;
    EXPECT_CALL(*testing::g_mock_backend,
                get_nic_rdma_port_statistics(k_handle, _, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<2>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_nic_rdma_port_statistics(k_handle, _, NotNull(), NotNull()))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_nic_rdma_port_statistics(0)),
                 std::runtime_error);
}

// ── SDMA ─────────────────────────────────────────────────────────────────────

TEST_F(DeviceBackendTest, get_raw_sdma_usage_returns_zero_when_no_processes)
{
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(0U), Return(k_ok)));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_EQ(sut.get_raw_sdma_usage(), 0U);
}

TEST_F(DeviceBackendTest, get_raw_sdma_usage_throws_when_count_fails)
{
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), IsNull()))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_raw_sdma_usage()), std::runtime_error);
}

TEST_F(DeviceBackendTest, get_raw_sdma_usage_accumulates_sdma_usage)
{
    const testing::mock_proc_info_t procs[] = { { .sdma_usage = 100 },
                                                { .sdma_usage = 200 } };
    InSequence                      seq;
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(2U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(2U), SetArrayArgument<2>(procs, procs + 2),
                        Return(k_ok)));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_EQ(sut.get_raw_sdma_usage(), 300U);
}

TEST_F(DeviceBackendTest, get_raw_sdma_usage_throws_when_data_fetch_fails)
{
    InSequence seq;
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), NotNull()))
        .WillOnce(Return(k_err));

    DeviceSut sut{ m_session, k_handle };
    EXPECT_THROW(static_cast<void>(sut.get_raw_sdma_usage()), std::runtime_error);
}

}  // namespace rocprofsys::backends::amd_smi
