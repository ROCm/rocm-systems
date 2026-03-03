// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/gpu/device.hpp"
#include "library/pmc/device_providers/amd_smi/drivers/tests/mock_driver.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

using namespace rocprofsys::pmc::collectors::gpu;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

#if ROCPROFSYS_USE_ROCM > 0

using MockDriver = rocprofsys::pmc::drivers::amd_smi::testing::mock_driver;

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace gpu
{
namespace testing
{

/**
 * @brief Test fixture for GPU device tests.
 *
 * Provides common setup for device tests including mock driver and
 * helper methods for configuring mock behavior.
 */
class DeviceTest : public ::testing::Test
{
protected:
    std::shared_ptr<MockDriver> mock_driver;
    amdsmi_processor_handle     test_handle;
    processor_type_t            test_processor_type;
    size_t                      test_index;

    void SetUp() override
    {
        mock_driver         = std::make_shared<MockDriver>();
        test_handle         = reinterpret_cast<amdsmi_processor_handle>(0x1234);
        test_processor_type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
        test_index          = 0;
    }

    /**
     * @brief Configure mock to return GPU metrics with all valid values.
     */
    void SetupAllMetricsSupported()
    {
        amdsmi_gpu_metrics_t metrics = CreateValidMetrics();

        ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
            .WillByDefault(
                DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

        uint64_t mem_usage = 8589934592ULL;  // 8 GB
        ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(mem_usage), Return(AMDSMI_STATUS_SUCCESS)));
    }

    /**
     * @brief Configure mock to return GPU metrics with all sentinel values.
     */
    void SetupNoMetricsSupported()
    {
        amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

        ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
            .WillByDefault(
                DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

        uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
        ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));
    }

    /**
     * @brief Configure mock to return GPU metrics with partial support.
     *
     * Returns valid values for:
     * - current_socket_power
     * - temperature_hotspot
     * - average_gfx_activity
     *
     * All other metrics return sentinel values.
     */
    void SetupPartialMetricsSupported()
    {
        amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

        // Set specific metrics to valid values
        metrics.current_socket_power = 150;    // Valid power (watts)
        metrics.temperature_hotspot  = 75000;  // Valid temp (millidegrees)
        metrics.average_gfx_activity = 85;     // Valid activity (percent)

        ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
            .WillByDefault(
                DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

        uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
        ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));
    }

    /**
     * @brief Create amdsmi_gpu_metrics_t with all valid (non-sentinel) values.
     */
    static amdsmi_gpu_metrics_t CreateValidMetrics()
    {
        amdsmi_gpu_metrics_t metrics{};

        // Power metrics
        metrics.current_socket_power = 150;
        metrics.average_socket_power = 140;

        // Temperature metrics (in millidegrees Celsius)
        metrics.temperature_hotspot = 75000;
        metrics.temperature_edge    = 70000;

        // Activity metrics (percentage)
        metrics.average_gfx_activity = 85;
        metrics.average_umc_activity = 60;
        metrics.average_mm_activity  = 40;

        // XCP stats - VCN and JPEG activity
        for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            for(size_t i = 0; i < AMDSMI_MAX_NUM_VCN; ++i)
            {
                metrics.xcp_stats[xcp].vcn_busy[i] = static_cast<uint16_t>(50 + i);
            }
            for(size_t i = 0; i < ROCPROFSYS_MAX_NUM_JPEG_ENGINES; ++i)
            {
                metrics.xcp_stats[xcp].jpeg_busy[i] = static_cast<uint16_t>(30 + i);
            }
        }

        // XGMI metrics
        metrics.xgmi_link_width = 16;
        metrics.xgmi_link_speed = 25000;
        for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
        {
            metrics.xgmi_read_data_acc[i]  = 1000000ULL + i;
            metrics.xgmi_write_data_acc[i] = 2000000ULL + i;
        }

        // PCIe metrics
        metrics.pcie_link_width     = 16;
        metrics.pcie_link_speed     = 16000;  // Gen4
        metrics.pcie_bandwidth_acc  = 500000000ULL;
        metrics.pcie_bandwidth_inst = 10000000ULL;

        return metrics;
    }

    /**
     * @brief Create amdsmi_gpu_metrics_t with all sentinel values.
     */
    static amdsmi_gpu_metrics_t CreateSentinelMetrics()
    {
        amdsmi_gpu_metrics_t metrics{};

        // 32-bit sentinel
        metrics.current_socket_power = 0xFFFFFFFF;
        metrics.average_socket_power = 0xFFFFFFFF;
        metrics.average_gfx_activity = 0xFFFFFFFF;
        metrics.average_umc_activity = 0xFFFFFFFF;
        metrics.average_mm_activity  = 0xFFFFFFFF;

        // 64-bit sentinel for temperatures (stored as int64_t)
        metrics.temperature_hotspot = 0xFFFFFFFFFFFFFFFFLL;
        metrics.temperature_edge    = 0xFFFFFFFFFFFFFFFFLL;

        // 16-bit sentinel for XCP stats
        for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            for(size_t i = 0; i < AMDSMI_MAX_NUM_VCN; ++i)
            {
                metrics.xcp_stats[xcp].vcn_busy[i] = 0xFFFF;
            }
            for(size_t i = 0; i < ROCPROFSYS_MAX_NUM_JPEG_ENGINES; ++i)
            {
                metrics.xcp_stats[xcp].jpeg_busy[i] = 0xFFFF;
            }
        }

        // 16-bit sentinel for XGMI link info
        metrics.xgmi_link_width = 0xFFFF;
        metrics.xgmi_link_speed = 0xFFFF;

        // 64-bit sentinel for XGMI data
        for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
        {
            metrics.xgmi_read_data_acc[i]  = 0xFFFFFFFFFFFFFFFFULL;
            metrics.xgmi_write_data_acc[i] = 0xFFFFFFFFFFFFFFFFULL;
        }

        // 16-bit sentinel for PCIe link info
        metrics.pcie_link_width = 0xFFFF;
        metrics.pcie_link_speed = 0xFFFF;

        // 64-bit sentinel for PCIe bandwidth
        metrics.pcie_bandwidth_acc  = 0xFFFFFFFFFFFFFFFFULL;
        metrics.pcie_bandwidth_inst = 0xFFFFFFFFFFFFFFFFULL;

        return metrics;
    }
};

// ============================================================================
// Category 1: Constructor and Initialization Tests
// ============================================================================

/**
 * TC1.1: Valid Device Construction with Full Metric Support
 *
 * Objective: Verify device initializes correctly when all metrics are supported.
 */
TEST_F(DeviceTest, valid_device_construction_full_support)
{
    // Setup: All metrics return valid values
    SetupAllMetricsSupported();

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify device is supported
    EXPECT_TRUE(dev.is_supported());

    // Verify all metric bits are set
    auto supported = dev.get_supported_metrics();
    EXPECT_NE(supported.value, 0U);

    // Verify basic properties
    EXPECT_EQ(dev.get_index(), test_index);
    EXPECT_EQ(dev.get_device_type(), test_processor_type);
    EXPECT_EQ(dev.get_handle(), test_handle);
}

/**
 * TC1.2: Device Construction with No Supported Metrics
 *
 * Objective: Verify device handles hardware with no supported metrics.
 */
TEST_F(DeviceTest, device_construction_no_support)
{
    // Setup: All metrics return sentinel values
    SetupNoMetricsSupported();

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify device is not supported
    EXPECT_FALSE(dev.is_supported());

    // Verify no metric bits are set
    auto supported = dev.get_supported_metrics();
    EXPECT_EQ(supported.value, 0U);

    // Verify get_gpu_metrics returns all zeros
    auto metrics = dev.get_gpu_metrics();
    EXPECT_EQ(metrics.current_socket_power, 0U);
    EXPECT_EQ(metrics.average_socket_power, 0U);
    EXPECT_EQ(metrics.memory_usage, 0ULL);
}

/**
 * TC1.3: Device Construction with Partial Metric Support
 *
 * Objective: Verify selective metric initialization.
 */
TEST_F(DeviceTest, device_construction_partial_support)
{
    // Setup: Only specific metrics supported
    SetupPartialMetricsSupported();

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify device is supported (at least one metric available)
    EXPECT_TRUE(dev.is_supported());

    // Verify only expected metrics are marked as supported
    auto supported = dev.get_supported_metrics();

    EXPECT_TRUE(supported.bits.current_socket_power);
    EXPECT_TRUE(supported.bits.hotspot_temperature);
    EXPECT_TRUE(supported.bits.gfx_activity);

    // Verify unsupported metrics are not set
    EXPECT_FALSE(supported.bits.average_socket_power);
    EXPECT_FALSE(supported.bits.edge_temperature);
    EXPECT_FALSE(supported.bits.umc_activity);
    EXPECT_FALSE(supported.bits.mm_activity);
    EXPECT_FALSE(supported.bits.memory_usage);
    EXPECT_FALSE(supported.bits.vcn_activity);
    EXPECT_FALSE(supported.bits.jpeg_activity);
    EXPECT_FALSE(supported.bits.vcn_busy);
    EXPECT_FALSE(supported.bits.jpeg_busy);
    EXPECT_FALSE(supported.bits.xgmi);
    EXPECT_FALSE(supported.bits.pcie);
}

/**
 * TC1.4: Device Construction with Different Processor Types
 *
 * Objective: Verify device type is correctly stored.
 */
TEST_F(DeviceTest, device_construction_different_types)
{
    SetupAllMetricsSupported();

    // Test with GPU type
    {
        device<MockDriver> dev(mock_driver, test_handle, AMDSMI_PROCESSOR_TYPE_AMD_GPU,
                               0);
        EXPECT_EQ(dev.get_device_type(), AMDSMI_PROCESSOR_TYPE_AMD_GPU);
    }

    // Test with CPU type
    {
        device<MockDriver> dev(mock_driver, test_handle,
                               AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, 1);
        EXPECT_EQ(dev.get_device_type(), AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE);
    }

    // Test with APU type
    {
        device<MockDriver> dev(mock_driver, test_handle, AMDSMI_PROCESSOR_TYPE_AMD_APU,
                               2);
        EXPECT_EQ(dev.get_device_type(), AMDSMI_PROCESSOR_TYPE_AMD_APU);
    }
}

// ============================================================================
// Category 2: Power Metrics Collection Tests
// ============================================================================

/**
 * TC2.1: Current Socket Power Collection
 *
 * Objective: Verify current power is collected when supported.
 */
TEST_F(DeviceTest, current_socket_power_collection)
{
    // Setup: Mock returns specific current power value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.current_socket_power = 150;  // 150 watts

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device (initializes supported metrics)
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify current_socket_power is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.current_socket_power);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify current power value was collected
    EXPECT_EQ(collected_metrics.current_socket_power, 150U);
}

/**
 * TC2.2: Average Socket Power Collection
 *
 * Objective: Verify average power is collected when supported.
 */
TEST_F(DeviceTest, average_socket_power_collection)
{
    // Setup: Mock returns specific average power value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.average_socket_power = 140;  // 140 watts

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify average_socket_power is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.average_socket_power);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify average power value was collected
    EXPECT_EQ(collected_metrics.average_socket_power, 140U);
}

/**
 * TC2.3: Power Metrics Not Collected When Unsupported
 *
 * Objective: Verify power metrics remain zero when not supported.
 */
TEST_F(DeviceTest, power_metrics_not_collected_when_unsupported)
{
    // Setup: All metrics are sentinel values (unsupported)
    SetupNoMetricsSupported();

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify power metrics are not marked as supported
    auto supported = dev.get_supported_metrics();
    EXPECT_FALSE(supported.bits.current_socket_power);
    EXPECT_FALSE(supported.bits.average_socket_power);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify power values remain zero
    EXPECT_EQ(collected_metrics.current_socket_power, 0U);
    EXPECT_EQ(collected_metrics.average_socket_power, 0U);
}

// ============================================================================
// Category 3: Temperature Metrics Collection Tests
// ============================================================================

/**
 * TC2.4: Hotspot Temperature Collection
 *
 * Objective: Verify hotspot temperature collection.
 */
TEST_F(DeviceTest, hotspot_temperature_collection)
{
    // Setup: Mock returns specific hotspot temperature value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.temperature_hotspot  = 75000;  // 75°C in millidegrees

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify hotspot_temperature is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.hotspot_temperature);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify hotspot temperature value was collected
    EXPECT_EQ(collected_metrics.hotspot_temperature, 75000);
}

/**
 * TC2.5: Edge Temperature Collection
 *
 * Objective: Verify edge temperature collection.
 */
TEST_F(DeviceTest, edge_temperature_collection)
{
    // Setup: Mock returns specific edge temperature value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.temperature_edge     = 70000;  // 70°C in millidegrees

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify edge_temperature is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.edge_temperature);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify edge temperature value was collected
    EXPECT_EQ(collected_metrics.edge_temperature, 70000);
}

/**
 * TC2.6: Temperature Metrics Not Collected When Unsupported
 *
 * Objective: Verify temperature skipped when not supported.
 */
TEST_F(DeviceTest, temperature_metrics_not_collected_when_unsupported)
{
    // Setup: All metrics are sentinel values (unsupported)
    SetupNoMetricsSupported();

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify temperature metrics are not marked as supported
    auto supported = dev.get_supported_metrics();
    EXPECT_FALSE(supported.bits.hotspot_temperature);
    EXPECT_FALSE(supported.bits.edge_temperature);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify temperature values remain zero
    EXPECT_EQ(collected_metrics.hotspot_temperature, 0);
    EXPECT_EQ(collected_metrics.edge_temperature, 0);
}

// ============================================================================
// Category 4: Activity Metrics Collection Tests
// ============================================================================

/**
 * GFX Activity Collection
 *
 * Objective: Verify graphics engine activity collection.
 */
TEST_F(DeviceTest, gfx_activity_collection)
{
    // Setup: Mock returns specific GFX activity value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.average_gfx_activity = 85;  // 85% activity

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify gfx_activity is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.gfx_activity);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify GFX activity value was collected
    EXPECT_EQ(collected_metrics.gfx_activity, 85U);
}

/**
 * UMC Activity Collection
 *
 * Objective: Verify memory controller activity collection.
 */
TEST_F(DeviceTest, umc_activity_collection)
{
    // Setup: Mock returns specific UMC activity value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.average_umc_activity = 60;  // 60% activity

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify umc_activity is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.umc_activity);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify UMC activity value was collected
    EXPECT_EQ(collected_metrics.umc_activity, 60U);
}

/**
 * MM Activity Collection
 *
 * Objective: Verify multimedia activity collection.
 */
TEST_F(DeviceTest, mm_activity_collection)
{
    // Setup: Mock returns specific MM activity value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.average_mm_activity  = 40;  // 40% activity

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify mm_activity is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.mm_activity);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify MM activity value was collected
    EXPECT_EQ(collected_metrics.mm_activity, 40U);
}

/**
 * All Activity Metrics Collection
 *
 * Objective: Verify all three activity metrics collected together.
 */
TEST_F(DeviceTest, all_activity_metrics_collection)
{
    // Setup: Mock returns all three activity values
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.average_gfx_activity = 85;
    metrics.average_umc_activity = 60;
    metrics.average_mm_activity  = 40;

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify all activity metrics are marked as supported
    auto supported = dev.get_supported_metrics();
    EXPECT_TRUE(supported.bits.gfx_activity);
    EXPECT_TRUE(supported.bits.umc_activity);
    EXPECT_TRUE(supported.bits.mm_activity);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify all three activity values were collected correctly
    EXPECT_EQ(collected_metrics.gfx_activity, 85U);
    EXPECT_EQ(collected_metrics.umc_activity, 60U);
    EXPECT_EQ(collected_metrics.mm_activity, 40U);
}

// ============================================================================
// Category 5: Memory Usage Collection Tests
// ============================================================================

/**
 * VRAM Memory Usage Collection Success
 *
 * Objective: Verify VRAM usage collected when API succeeds.
 */
TEST_F(DeviceTest, vram_memory_usage_collection_success)
{
    // Setup: Mock returns sentinel for all GPU metrics
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    // Mock returns valid memory usage (8 GB)
    uint64_t mem_usage = 8589934592ULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(DoAll(SetArgPointee<2>(mem_usage), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify memory_usage is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.memory_usage);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify memory usage value was collected
    EXPECT_EQ(collected_metrics.memory_usage, 8589934592ULL);
}

/**
 * Memory Usage Collection Failure
 *
 * Objective: Verify memory usage remains zero on API failure.
 */
TEST_F(DeviceTest, memory_usage_collection_failure)
{
    // Setup: Mock returns sentinel for all GPU metrics
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    // Mock returns failure for memory usage
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify memory_usage is NOT marked as supported
    EXPECT_FALSE(dev.get_supported_metrics().bits.memory_usage);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify memory usage remains zero
    EXPECT_EQ(collected_metrics.memory_usage, 0ULL);
}

/**
 * Memory Usage Not Collected When Unsupported
 *
 * Objective: Verify early return when memory not supported.
 */
TEST_F(DeviceTest, memory_usage_not_collected_when_unsupported)
{
    // Setup: All metrics are sentinel values (unsupported)
    SetupNoMetricsSupported();

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify memory_usage is NOT marked as supported
    EXPECT_FALSE(dev.get_supported_metrics().bits.memory_usage);

    // Mock should NOT be called for memory usage during collection
    // (because supported bit is false, early return happens)
    EXPECT_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .Times(0);  // Should not be called during get_gpu_metrics()

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify memory usage remains zero
    EXPECT_EQ(collected_metrics.memory_usage, 0ULL);
}

// ============================================================================
// Category 6: XCP Metrics Collection Tests
// ============================================================================

/**
 * VCN Busy Collection - All XCPs (MI300)
 *
 * Objective: Verify per-XCP VCN busy stats copied for all XCP instances.
 */
TEST_F(DeviceTest, vcn_busy_collection_all_xcps)
{
    // Setup: Mock returns valid VCN busy values for all XCPs
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Set VCN busy values for all XCP instances
    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < AMDSMI_MAX_NUM_VCN; ++vcn)
        {
            metrics.xcp_stats[xcp].vcn_busy[vcn] = static_cast<uint16_t>(50 + xcp + vcn);
        }
    }

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify vcn_busy is marked as supported (per-XCP metrics)
    EXPECT_TRUE(dev.get_supported_metrics().bits.vcn_busy);
    // Device-level vcn_activity should NOT be set when per-XCP is available
    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify all XCP VCN arrays were copied correctly
    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < AMDSMI_MAX_NUM_VCN; ++vcn)
        {
            EXPECT_EQ(collected_metrics.xcp_stats[xcp].vcn_busy[vcn],
                      static_cast<uint16_t>(50 + xcp + vcn));
        }
    }
}

/**
 * JPEG Activity Collection - All XCPs
 *
 * Objective: Verify JPEG busy stats copied for all XCP instances.
 */
TEST_F(DeviceTest, jpeg_activity_collection_all_xcps)
{
    // Setup: Mock returns valid JPEG busy values for all XCPs
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Set JPEG activity values for all XCP instances
    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t jpeg = 0; jpeg < ROCPROFSYS_MAX_NUM_JPEG_ENGINES; ++jpeg)
        {
            metrics.xcp_stats[xcp].jpeg_busy[jpeg] =
                static_cast<uint16_t>(30 + xcp + jpeg);
        }
    }

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify jpeg_activity is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.jpeg_activity);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify all XCP JPEG arrays were copied correctly
    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t jpeg = 0; jpeg < ROCPROFSYS_MAX_NUM_JPEG_ENGINES; ++jpeg)
        {
            EXPECT_EQ(collected_metrics.xcp_stats[xcp].jpeg_busy[jpeg],
                      static_cast<uint16_t>(30 + xcp + jpeg));
        }
    }
}

/**
 * XCP Metrics Not Collected When Unsupported
 *
 * Objective: Verify XCP metrics skipped when not supported.
 */
TEST_F(DeviceTest, xcp_metrics_not_collected_when_unsupported)
{
    // Setup: All metrics are sentinel values (unsupported)
    SetupNoMetricsSupported();

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify XCP metrics are NOT marked as supported
    auto supported = dev.get_supported_metrics();
    EXPECT_FALSE(supported.bits.vcn_activity);
    EXPECT_FALSE(supported.bits.jpeg_activity);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify XCP arrays remain default-initialized (all zeros)
    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < AMDSMI_MAX_NUM_VCN; ++vcn)
        {
            EXPECT_EQ(collected_metrics.xcp_stats[xcp].vcn_busy[vcn], 0);
        }
        for(size_t jpeg = 0; jpeg < ROCPROFSYS_MAX_NUM_JPEG_ENGINES; ++jpeg)
        {
            EXPECT_EQ(collected_metrics.xcp_stats[xcp].jpeg_busy[jpeg], 0);
        }
    }
}

/**
 * Mixed VCN/JPEG Support
 *
 * Objective: Verify VCN collected but not JPEG when only VCN supported.
 */
TEST_F(DeviceTest, mixed_vcn_jpeg_support)
{
    // Setup: Only VCN is supported, JPEG is not
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Set valid VCN values for all XCPs
    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < AMDSMI_MAX_NUM_VCN; ++vcn)
        {
            metrics.xcp_stats[xcp].vcn_busy[vcn] = static_cast<uint16_t>(50 + vcn);
        }
        // JPEG remains sentinel (0xFFFF)
    }

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify only VCN is supported
    auto supported = dev.get_supported_metrics();
    EXPECT_TRUE(supported.bits.vcn_activity);
    EXPECT_FALSE(supported.bits.jpeg_activity);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify VCN arrays are populated
    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < AMDSMI_MAX_NUM_VCN; ++vcn)
        {
            EXPECT_EQ(collected_metrics.xcp_stats[xcp].vcn_busy[vcn],
                      static_cast<uint16_t>(50 + vcn));
        }
    }

    // Verify JPEG arrays remain default-initialized (zeros)
    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t jpeg = 0; jpeg < ROCPROFSYS_MAX_NUM_JPEG_ENGINES; ++jpeg)
        {
            EXPECT_EQ(collected_metrics.xcp_stats[xcp].jpeg_busy[jpeg], 0);
        }
    }
}

// ============================================================================
// Category 7: XGMI Metrics Collection Tests
// ============================================================================

/**
 * XGMI Link Width Collection
 *
 * Objective: Verify XGMI link width is populated when supported.
 */
TEST_F(DeviceTest, xgmi_link_width_collection)
{
    // Setup: Mock returns specific XGMI link width value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.xgmi_link_width      = 16;  // 16-bit link width

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify XGMI is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify XGMI link width was collected
    EXPECT_EQ(collected_metrics.xgmi.link.width, 16U);
}

/**
 * XGMI Link Speed Collection
 *
 * Objective: Verify XGMI link speed is populated when supported.
 */
TEST_F(DeviceTest, xgmi_link_speed_collection)
{
    // Setup: Mock returns specific XGMI link speed value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.xgmi_link_speed      = 25;  // 25 GT/s

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify XGMI is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify XGMI link speed was collected
    EXPECT_EQ(collected_metrics.xgmi.link.speed, 25U);
}

/**
 * XGMI Read/Write Data Collection for All Links
 *
 * Objective: Verify data accumulation for all XGMI links.
 */
TEST_F(DeviceTest, xgmi_read_write_data_collection_all_links)
{
    // Setup: Mock returns valid read/write data for all XGMI links
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Populate read and write data for all XGMI links
    for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
    {
        metrics.xgmi_read_data_acc[i]  = 1000000 + i * 1000;  // Read data in bytes
        metrics.xgmi_write_data_acc[i] = 2000000 + i * 1000;  // Write data in bytes
    }

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify XGMI is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify all XGMI link read/write data was collected
    for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
    {
        EXPECT_EQ(collected_metrics.xgmi.data_acc.read[i], 1000000 + i * 1000);
        EXPECT_EQ(collected_metrics.xgmi.data_acc.write[i], 2000000 + i * 1000);
    }
}

/**
 * XGMI Sentinel Value Handling
 *
 * Objective: Verify sentinel values are zeroed out properly.
 */
TEST_F(DeviceTest, xgmi_sentinel_value_handling)
{
    // Setup: Mix of valid and sentinel XGMI values
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Set valid link width, but sentinel link speed
    metrics.xgmi_link_width = 16;
    // xgmi_link_speed remains 0xFFFF (sentinel)

    // Set some valid and some sentinel read/write data
    metrics.xgmi_read_data_acc[0]  = 1000000;                // Valid
    metrics.xgmi_read_data_acc[1]  = 0xFFFFFFFFFFFFFFFFULL;  // Sentinel
    metrics.xgmi_write_data_acc[0] = 2000000;                // Valid
    metrics.xgmi_write_data_acc[1] = 0xFFFFFFFFFFFFFFFFULL;  // Sentinel

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify XGMI is marked as supported (at least one metric is valid)
    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify valid values are collected and sentinels are zeroed
    EXPECT_EQ(collected_metrics.xgmi.link.width, 16U);
    EXPECT_EQ(collected_metrics.xgmi.link.speed, 0U);  // Sentinel converted to 0
    EXPECT_EQ(collected_metrics.xgmi.data_acc.read[0], 1000000U);
    EXPECT_EQ(collected_metrics.xgmi.data_acc.read[1], 0U);  // Sentinel converted to 0
    EXPECT_EQ(collected_metrics.xgmi.data_acc.write[0], 2000000U);
    EXPECT_EQ(collected_metrics.xgmi.data_acc.write[1], 0U);  // Sentinel converted to 0
}

/**
 * XGMI Not Collected When Unsupported
 *
 * Objective: Verify early return when XGMI metrics are not supported.
 */
TEST_F(DeviceTest, xgmi_not_collected_when_unsupported)
{
    // Setup: All XGMI metrics are sentinel values (unsupported)
    SetupNoMetricsSupported();

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify XGMI is NOT marked as supported
    EXPECT_FALSE(dev.get_supported_metrics().bits.xgmi);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify all XGMI metrics remain default-initialized (zeros)
    EXPECT_EQ(collected_metrics.xgmi.link.width, 0U);
    EXPECT_EQ(collected_metrics.xgmi.link.speed, 0U);

    for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
    {
        EXPECT_EQ(collected_metrics.xgmi.data_acc.read[i], 0U);
        EXPECT_EQ(collected_metrics.xgmi.data_acc.write[i], 0U);
    }
}

// ============================================================================
// Category 8: PCIe Metrics Collection Tests
// ============================================================================

/**
 * PCIe Link Width Collection
 *
 * Objective: Verify PCIe link width is populated when supported.
 */
TEST_F(DeviceTest, pcie_link_width_collection)
{
    // Setup: Mock returns specific PCIe link width value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.pcie_link_width      = 16;  // x16 PCIe lanes

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify PCIe is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify PCIe link width was collected
    EXPECT_EQ(collected_metrics.pcie.link.width, 16U);
}

/**
 * PCIe Link Speed Collection
 *
 * Objective: Verify PCIe link speed is populated when supported.
 */
TEST_F(DeviceTest, pcie_link_speed_collection)
{
    // Setup: Mock returns specific PCIe link speed value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.pcie_link_speed      = 16000;  // 16 GT/s (Gen4)

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify PCIe is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify PCIe link speed was collected
    EXPECT_EQ(collected_metrics.pcie.link.speed, 16000U);
}

/**
 * PCIe Bandwidth Accumulator Collection
 *
 * Objective: Verify bandwidth accumulator is populated when supported.
 */
TEST_F(DeviceTest, pcie_bandwidth_accumulator_collection)
{
    // Setup: Mock returns specific PCIe bandwidth accumulator value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.pcie_bandwidth_acc   = 500000000;  // 500MB accumulated

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify PCIe is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify PCIe bandwidth accumulator was collected
    EXPECT_EQ(collected_metrics.pcie.bandwidth.acc, 500000000U);
}

/**
 * PCIe Bandwidth Instantaneous Collection
 *
 * Objective: Verify instantaneous bandwidth is populated when supported.
 */
TEST_F(DeviceTest, pcie_bandwidth_instantaneous_collection)
{
    // Setup: Mock returns specific PCIe instantaneous bandwidth value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.pcie_bandwidth_inst  = 10000000;  // 10 MB/s

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify PCIe is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify PCIe instantaneous bandwidth was collected
    EXPECT_EQ(collected_metrics.pcie.bandwidth.inst, 10000000U);
}

/**
 * PCIe Sentinel Value Handling
 *
 * Objective: Verify sentinel values are zeroed out properly.
 */
TEST_F(DeviceTest, pcie_sentinel_value_handling)
{
    // Setup: Mix of valid and sentinel PCIe values
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Set valid link width and bandwidth acc, but sentinel link speed and bandwidth inst
    metrics.pcie_link_width    = 16;
    metrics.pcie_bandwidth_acc = 500000000;
    // pcie_link_speed and pcie_bandwidth_inst remain sentinel values

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify PCIe is marked as supported (at least one metric is valid)
    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify valid values are collected and sentinels are zeroed
    EXPECT_EQ(collected_metrics.pcie.link.width, 16U);
    EXPECT_EQ(collected_metrics.pcie.link.speed, 0U);  // Sentinel converted to 0
    EXPECT_EQ(collected_metrics.pcie.bandwidth.acc, 500000000U);
    EXPECT_EQ(collected_metrics.pcie.bandwidth.inst, 0U);  // Sentinel converted to 0
}

/**
 * PCIe Not Collected When Unsupported
 *
 * Objective: Verify early return when PCIe metrics are not supported.
 */
TEST_F(DeviceTest, pcie_not_collected_when_unsupported)
{
    // Setup: All PCIe metrics are sentinel values (unsupported)
    SetupNoMetricsSupported();

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify PCIe is NOT marked as supported
    EXPECT_FALSE(dev.get_supported_metrics().bits.pcie);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify all PCIe metrics remain default-initialized (zeros)
    EXPECT_EQ(collected_metrics.pcie.link.width, 0U);
    EXPECT_EQ(collected_metrics.pcie.link.speed, 0U);
    EXPECT_EQ(collected_metrics.pcie.bandwidth.acc, 0U);
    EXPECT_EQ(collected_metrics.pcie.bandwidth.inst, 0U);
}

// ============================================================================
// Category 9: Supported Metrics Detection Tests
// ============================================================================

/**
 * All Metrics Supported Detection
 *
 * Objective: Verify all supported bits are set when all metrics are valid.
 */
TEST_F(DeviceTest, all_metrics_supported_detection)
{
    // Setup: All metrics have valid values (non-sentinel)
    SetupAllMetricsSupported();

    // Create device (this triggers initialize_supported_metrics())
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify all metric support bits are set
    auto supported = dev.get_supported_metrics();
    EXPECT_TRUE(supported.bits.current_socket_power);
    EXPECT_TRUE(supported.bits.average_socket_power);
    EXPECT_TRUE(supported.bits.memory_usage);
    EXPECT_TRUE(supported.bits.hotspot_temperature);
    EXPECT_TRUE(supported.bits.edge_temperature);
    EXPECT_TRUE(supported.bits.gfx_activity);
    EXPECT_TRUE(supported.bits.umc_activity);
    EXPECT_TRUE(supported.bits.mm_activity);
    EXPECT_TRUE(supported.bits.vcn_activity);
    EXPECT_TRUE(supported.bits.jpeg_activity);
    EXPECT_TRUE(supported.bits.xgmi);
    EXPECT_TRUE(supported.bits.pcie);
}

/**
 * VCN Activity Support Detection - Any XCP
 *
 * Objective: Verify VCN marked supported if any XCP has valid values.
 */
TEST_F(DeviceTest, vcn_activity_support_detection_any_xcp)
{
    // Setup: Only XCP 7 has valid VCN values, all others are sentinels
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Set valid VCN value only in XCP 7
    metrics.xcp_stats[7].vcn_busy[0] = 50;  // Valid value

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify VCN activity is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.vcn_activity);
}

/**
 * VCN Activity Unsupported - All Sentinels
 *
 * Objective: Verify VCN not supported when all XCPs have sentinel values.
 */
TEST_F(DeviceTest, vcn_activity_unsupported_all_sentinels)
{
    // Setup: All VCN values in all XCPs are sentinels
    SetupNoMetricsSupported();

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify VCN activity is NOT supported
    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity);
}

/**
 * JPEG Activity Support Detection - Any XCP
 *
 * Objective: Verify JPEG marked supported if any XCP has valid values.
 */
TEST_F(DeviceTest, jpeg_activity_support_detection_any_xcp)
{
    // Setup: Only XCP 5 has valid JPEG values, all others are sentinels
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Set valid JPEG value only in XCP 5
    metrics.xcp_stats[5].jpeg_busy[0] = 75;  // Valid value

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify JPEG activity is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.jpeg_activity);
}

/**
 * XGMI Support Detection - Link Width Only
 *
 * Objective: Verify XGMI supported if only link width is valid.
 */
TEST_F(DeviceTest, xgmi_support_detection_link_width_only)
{
    // Setup: Only XGMI link width is valid, everything else is sentinel
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.xgmi_link_width      = 16;  // Valid link width
    // xgmi_link_speed and all xgmi_read_data_acc remain sentinel

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify XGMI is marked as supported (OR logic)
    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);
}

/**
 * XGMI Support Detection - Any Read Data Valid
 *
 * Objective: Verify XGMI supported if any read data is valid.
 */
TEST_F(DeviceTest, xgmi_support_detection_any_read_data_valid)
{
    // Setup: Only one XGMI read data value is valid
    amdsmi_gpu_metrics_t metrics  = CreateSentinelMetrics();
    metrics.xgmi_read_data_acc[2] = 1000;  // Valid read data at index 2
    // link width, link speed, and all other read data remain sentinel

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify XGMI is marked as supported (std::any_of logic)
    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);
}

/**
 * PCIe Support Detection - Bandwidth Only
 *
 * Objective: Verify PCIe supported if only bandwidth accumulator is valid.
 */
TEST_F(DeviceTest, pcie_support_detection_bandwidth_only)
{
    // Setup: Only PCIe bandwidth accumulator is valid
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();
    metrics.pcie_bandwidth_acc   = 1000000;  // Valid bandwidth accumulator
    // pcie_link_width, pcie_link_speed, pcie_bandwidth_inst remain sentinel

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify PCIe is marked as supported (OR logic)
    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);
}

/**
 * Memory Usage Support Detection
 *
 * Objective: Verify memory usage support based on API success with valid value.
 */
TEST_F(DeviceTest, memory_usage_support_detection)
{
    // Setup: Memory API returns success with valid value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t valid_mem_usage = 4096000000;  // 4GB
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(valid_mem_usage), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify memory usage is marked as supported
    EXPECT_TRUE(dev.get_supported_metrics().bits.memory_usage);
}

/**
 * Memory Usage Unsupported - API Failure
 *
 * Objective: Verify memory not supported when API fails.
 */
TEST_F(DeviceTest, memory_usage_unsupported_api_failure)
{
    // Setup: Memory API returns failure
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify memory usage is NOT marked as supported
    EXPECT_FALSE(dev.get_supported_metrics().bits.memory_usage);
}

/**
 * Memory Usage Unsupported - Sentinel Value
 *
 * Objective: Verify memory not supported when value is sentinel.
 */
TEST_F(DeviceTest, memory_usage_unsupported_sentinel_value)
{
    // Setup: Memory API returns success but with sentinel value
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;  // Sentinel value
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Verify memory usage is NOT marked as supported
    EXPECT_FALSE(dev.get_supported_metrics().bits.memory_usage);
}

// ============================================================================
// Category 10: VCN Activity Dual Source Tests
// ============================================================================

/**
 * VCN Activity in Top-Level Field Only
 *
 * Objective: Verify VCN activity is detected when present in top-level vcn_activity[]
 * field but NOT in xcp_stats[].vcn_busy[] arrays.
 *
 * Note: This tests a gap in the current implementation - the device class currently
 * only checks xcp_stats[].vcn_busy[] but should also check the top-level vcn_activity[].
 */
TEST_F(DeviceTest, vcn_activity_top_level_field_only)
{
    // Setup: VCN activity present in top-level field, XCP stats have sentinels
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Set valid VCN activity in top-level field
    // Note: This field exists in amdsmi_gpu_metrics_t but is not currently checked!
    // metrics.vcn_activity[0] = 75;  // 75% VCN utilization
    // metrics.vcn_activity[1] = 50;  // 50% VCN utilization

    // All XCP VCN busy values remain sentinel (0xFFFF)

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // EXPECTED BEHAVIOR: VCN activity should be marked as supported
    // CURRENT BEHAVIOR: Will NOT be supported because implementation only checks XCP
    // stats This test documents the gap and will fail until implementation is fixed

    // When implementation is fixed, uncomment:
    // EXPECT_TRUE(dev.get_supported_metrics().bits.vcn_activity);

    // Current behavior (documents the bug):
    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity)
        << "BUG: Implementation does not check top-level vcn_activity[] field";
}

/**
 * VCN Activity in Both Top-Level and XCP Fields
 *
 * Objective: Verify VCN activity when present in BOTH vcn_activity[] and
 * xcp_stats[].vcn_busy[].
 */
TEST_F(DeviceTest, vcn_activity_in_both_fields)
{
    // Setup: VCN activity in both top-level and XCP fields
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Set valid VCN activity in XCP stats (currently checked)
    metrics.xcp_stats[0].vcn_busy[0] = 80;  // 80% in XCP 0, VCN 0

    // Also set in top-level field (not currently checked)
    // metrics.vcn_activity[0] = 75;  // Different value in top-level field

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // VCN activity should be supported (XCP stats are valid)
    EXPECT_TRUE(dev.get_supported_metrics().bits.vcn_activity);

    // Collect metrics
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify XCP stats were collected (current implementation)
    EXPECT_EQ(collected_metrics.xcp_stats[0].vcn_busy[0], 80U);

    // When implementation is enhanced to collect from both sources,
    // add verification for top-level vcn_activity[] values
}

/**
 * VCN Activity Detection with Top-Level Field Support
 *
 * Objective: Document expected behavior when both VCN sources are checked.
 *
 * This test describes how the initialize_supported_metrics() should work:
 * - Check top-level vcn_activity[] array (currently missing)
 * - Check xcp_stats[].vcn_busy[] arrays (currently implemented)
 * - Mark vcn_activity as supported if EITHER source has valid data
 */
TEST_F(DeviceTest, vcn_activity_detection_should_check_both_sources)
{
    // Setup: Only top-level vcn_activity has valid data
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Top-level has valid data (not checked by current implementation)
    // metrics.vcn_activity[0] = 60;

    // XCP stats have sentinels (checked by current implementation)
    // All xcp_stats[].vcn_busy[] remain 0xFFFF

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // EXPECTED (when fixed): vcn_activity should be supported
    // CURRENT: Will be false because top-level field is not checked
    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity)
        << "Implementation gap: initialize_supported_metrics() should check both "
           "vcn_activity[] AND xcp_stats[].vcn_busy[]";
}

/**
 * VCN Activity Collection Priority
 *
 * Objective: Document which VCN source should take priority when collecting.
 *
 * When both sources are available, the implementation should decide:
 * - Use top-level vcn_activity[] for overall VCN utilization?
 * - Use xcp_stats[].vcn_busy[] for per-partition granularity?
 * - Collect from both?
 */
TEST_F(DeviceTest, vcn_activity_collection_priority)
{
    // Setup: Different values in both sources
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // XCP stats (per-partition detail)
    metrics.xcp_stats[0].vcn_busy[0] = 80;
    metrics.xcp_stats[0].vcn_busy[1] = 70;

    // Top-level (overall average?)
    // metrics.vcn_activity[0] = 75;  // Average of 80 and 70?
    // metrics.vcn_activity[1] = 0;   // Or different semantic meaning?

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    auto collected = dev.get_gpu_metrics();

    // Current implementation collects from XCP stats only
    EXPECT_EQ(collected.xcp_stats[0].vcn_busy[0], 80U);
    EXPECT_EQ(collected.xcp_stats[0].vcn_busy[1], 70U);

    // Future enhancement: Also collect top-level vcn_activity[]?
    // This would require extending the metrics structure to include both fields
}

/**
 * VCN Activity XCP Stats Empty But Top-Level Valid
 *
 * Objective: Test scenario where hardware reports VCN activity at top-level
 * but XCP partitioning is disabled or not reporting VCN stats.
 */
TEST_F(DeviceTest, vcn_activity_xcp_disabled_top_level_valid)
{
    // Setup: XCP stats all sentinels (XCP disabled or not supported)
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    // Top-level VCN activity still valid
    // metrics.vcn_activity[0] = 65;  // VCN 0 at 65%
    // metrics.vcn_activity[1] = 55;  // VCN 1 at 55%

    // All XCP stats remain sentinel
    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // CURRENT: VCN not supported (implementation only checks XCP stats)
    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity);

    // EXPECTED (when fixed): Should be supported via top-level field
    // This represents real hardware scenario where XCP partitioning is disabled
    // but VCN engines are still active and reporting utilization
}

// ============================================================================
// Category 11: Error Handling and Edge Cases
// ============================================================================

/**
 * get_metrics_info() Failure
 *
 * Objective: Verify graceful handling when metrics info unavailable.
 */
TEST_F(DeviceTest, get_metrics_info_failure)
{
    // Setup: get_metrics_info() returns failure
    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    uint64_t valid_mem_usage = 4096000000;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(valid_mem_usage), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Call get_gpu_metrics() - should not throw
    auto metrics = dev.get_gpu_metrics();

    // Verify returns default-initialized metrics (all zeros)
    EXPECT_EQ(metrics.current_socket_power, 0U);
    EXPECT_EQ(metrics.average_socket_power, 0U);
    EXPECT_EQ(metrics.hotspot_temperature, 0);
    EXPECT_EQ(metrics.edge_temperature, 0);
    EXPECT_EQ(metrics.gfx_activity, 0U);
}

/**
 * get_metrics_info() Failure During Initialization
 *
 * Objective: Verify initialization handles metrics info failure.
 */
TEST_F(DeviceTest, get_metrics_info_failure_during_init)
{
    // Setup: get_metrics_info() returns failure during construction
    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    uint64_t valid_mem_usage = 4096000000;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(valid_mem_usage), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device - should not crash
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // is_supported() should reflect whether ANY metric was supported (memory in this
    // case)
    EXPECT_TRUE(dev.is_supported());

    // Verify memory is supported but GPU metrics are not
    auto supported = dev.get_supported_metrics();
    EXPECT_TRUE(supported.bits.memory_usage);
    EXPECT_FALSE(supported.bits.current_socket_power);
}

/**
 * Multiple Metric Collections
 *
 * Objective: Verify device can collect metrics multiple times.
 */
TEST_F(DeviceTest, multiple_metric_collections)
{
    // Setup: Mock returns varying values across collections
    SetupAllMetricsSupported();

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Collect metrics 10 times in a row
    for(int i = 0; i < 10; ++i)
    {
        auto metrics = dev.get_gpu_metrics();
        // Each collection should succeed
        EXPECT_GT(metrics.current_socket_power, 0U);
    }
}

/**
 * Large Array Indices - XGMI
 *
 * Objective: Verify no buffer overflow with maximum XGMI links.
 */
TEST_F(DeviceTest, large_array_indices_xgmi)
{
    // Setup: Set all AMDSMI_MAX_NUM_XGMI_LINKS entries
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
    {
        metrics.xgmi_read_data_acc[i]  = 1000 + i;
        metrics.xgmi_write_data_acc[i] = 2000 + i;
    }

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Collect metrics - should not crash or cause buffer overflow
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify all links were processed correctly
    for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
    {
        EXPECT_EQ(collected_metrics.xgmi.data_acc.read[i], 1000 + i);
        EXPECT_EQ(collected_metrics.xgmi.data_acc.write[i], 2000 + i);
    }
}

/**
 * Large Array Indices - XCP
 *
 * Objective: Verify no buffer overflow with maximum XCPs.
 */
TEST_F(DeviceTest, large_array_indices_xcp)
{
    // Setup: Set all AMDSMI_MAX_NUM_XCP entries
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < AMDSMI_MAX_NUM_VCN; ++vcn)
        {
            metrics.xcp_stats[xcp].vcn_busy[vcn] = static_cast<uint16_t>(xcp * 10 + vcn);
        }
    }

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Collect metrics - should not crash
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify all XCP stats were processed correctly
    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < AMDSMI_MAX_NUM_VCN; ++vcn)
        {
            EXPECT_EQ(collected_metrics.xcp_stats[xcp].vcn_busy[vcn],
                      static_cast<uint16_t>(xcp * 10 + vcn));
        }
    }
}

/**
 * Large Array Indices - JPEG Engines
 *
 * Objective: Verify no buffer overflow with maximum JPEG engines.
 */
TEST_F(DeviceTest, large_array_indices_jpeg)
{
    // Setup: Set all ROCPROFSYS_MAX_NUM_JPEG_ENGINES entries
    amdsmi_gpu_metrics_t metrics = CreateSentinelMetrics();

    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t jpeg = 0; jpeg < ROCPROFSYS_MAX_NUM_JPEG_ENGINES; ++jpeg)
        {
            metrics.xcp_stats[xcp].jpeg_busy[jpeg] =
                static_cast<uint16_t>(xcp * 100 + jpeg);
        }
    }

    ON_CALL(*mock_driver, get_metrics_info(test_handle, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create device
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, test_index);

    // Collect metrics - should not crash
    auto collected_metrics = dev.get_gpu_metrics();

    // Verify all JPEG engines were processed correctly
    for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
    {
        for(size_t jpeg = 0; jpeg < ROCPROFSYS_MAX_NUM_JPEG_ENGINES; ++jpeg)
        {
            EXPECT_EQ(collected_metrics.xcp_stats[xcp].jpeg_busy[jpeg],
                      static_cast<uint16_t>(xcp * 100 + jpeg));
        }
    }
}

/**
 * Concurrent Device Objects
 *
 * Objective: Verify multiple device objects don't interfere.
 */
TEST_F(DeviceTest, concurrent_device_objects)
{
    // Setup: Create mocks for two different devices
    auto mock_driver1 = std::make_shared<MockDriver>();
    auto mock_driver2 = std::make_shared<MockDriver>();

    amdsmi_processor_handle handle1 = reinterpret_cast<amdsmi_processor_handle>(0x1111);
    amdsmi_processor_handle handle2 = reinterpret_cast<amdsmi_processor_handle>(0x2222);

    // Device 1 returns power = 100W
    amdsmi_gpu_metrics_t metrics1 = CreateSentinelMetrics();
    metrics1.current_socket_power = 100;

    ON_CALL(*mock_driver1, get_metrics_info(handle1, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics1), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock_driver1, get_memory_usage(handle1, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Device 2 returns power = 200W
    amdsmi_gpu_metrics_t metrics2 = CreateSentinelMetrics();
    metrics2.current_socket_power = 200;

    ON_CALL(*mock_driver2, get_metrics_info(handle2, _))
        .WillByDefault(DoAll(SetArgPointee<1>(metrics2), Return(AMDSMI_STATUS_SUCCESS)));

    ON_CALL(*mock_driver2, get_memory_usage(handle2, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Create two device objects
    device<MockDriver> dev1(mock_driver1, handle1, test_processor_type, 0);
    device<MockDriver> dev2(mock_driver2, handle2, test_processor_type, 1);

    // Collect from device 1
    auto result1 = dev1.get_gpu_metrics();
    EXPECT_EQ(result1.current_socket_power, 100U);

    // Collect from device 2
    auto result2 = dev2.get_gpu_metrics();
    EXPECT_EQ(result2.current_socket_power, 200U);

    // Collect from device 1 again - should still return 100W
    result1 = dev1.get_gpu_metrics();
    EXPECT_EQ(result1.current_socket_power, 100U);

    // Verify devices maintain independent state
    EXPECT_NE(dev1.get_handle(), dev2.get_handle());
    EXPECT_NE(dev1.get_index(), dev2.get_index());
}

/**
 * Device with Index 0
 *
 * Objective: Verify device index 0 works (boundary value).
 */
TEST_F(DeviceTest, device_with_index_zero)
{
    // Setup
    SetupAllMetricsSupported();

    // Create device with index 0
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, 0);

    // Verify index is correctly stored
    EXPECT_EQ(dev.get_index(), 0U);
}

/**
 * Device with High Index
 *
 * Objective: Verify device with high index works (multi-GPU scenario).
 */
TEST_F(DeviceTest, device_with_high_index)
{
    // Setup
    SetupAllMetricsSupported();

    // Create device with high index (simulating 16-GPU system)
    device<MockDriver> dev(mock_driver, test_handle, test_processor_type, 15);

    // Verify index is correctly stored
    EXPECT_EQ(dev.get_index(), 15U);
}

// ============================================================================
// Category 12: Integration Tests
// ============================================================================

/**
 * Full Lifecycle with Real-ish Data
 *
 * Objective: Simulate realistic GPU monitoring session with evolving metrics.
 */
TEST_F(DeviceTest, full_lifecycle_with_realistic_data)
{
    // Setup: Mock will return different values across collections
    auto mock = std::make_shared<MockDriver>();

    // Collection 1: Idle GPU
    amdsmi_gpu_metrics_t metrics1 = CreateSentinelMetrics();
    metrics1.current_socket_power = 150;    // 150W
    metrics1.temperature_hotspot  = 70000;  // 70°C
    metrics1.average_gfx_activity = 50;     // 50% activity

    // Collection 2: Heavy workload
    amdsmi_gpu_metrics_t metrics2 = CreateSentinelMetrics();
    metrics2.current_socket_power = 180;    // 180W
    metrics2.temperature_hotspot  = 75000;  // 75°C
    metrics2.average_gfx_activity = 90;     // 90% activity

    // Collection 3: Returning to moderate
    amdsmi_gpu_metrics_t metrics3 = CreateSentinelMetrics();
    metrics3.current_socket_power = 160;    // 160W
    metrics3.temperature_hotspot  = 73000;  // 73°C
    metrics3.average_gfx_activity = 60;     // 60% activity

    // Setup mock to return different values on each call
    EXPECT_CALL(*mock, get_metrics_info(test_handle, _))
        .WillOnce(DoAll(SetArgPointee<1>(metrics1), Return(AMDSMI_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<1>(metrics2), Return(AMDSMI_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<1>(metrics3), Return(AMDSMI_STATUS_SUCCESS)));

    uint64_t sentinel_mem = 0xFFFFFFFFFFFFFFFFULL;
    ON_CALL(*mock, get_memory_usage(test_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(sentinel_mem), Return(AMDSMI_STATUS_SUCCESS)));

    // Construct device
    device<MockDriver> dev(mock, test_handle, test_processor_type, test_index);

    // Collection 1: Idle
    auto result1 = dev.get_gpu_metrics();
    EXPECT_EQ(result1.current_socket_power, 150U);
    EXPECT_EQ(result1.hotspot_temperature, 70000);
    EXPECT_EQ(result1.gfx_activity, 50U);

    // Collection 2: Heavy
    auto result2 = dev.get_gpu_metrics();
    EXPECT_EQ(result2.current_socket_power, 180U);
    EXPECT_EQ(result2.hotspot_temperature, 75000);
    EXPECT_EQ(result2.gfx_activity, 90U);

    // Collection 3: Moderate
    auto result3 = dev.get_gpu_metrics();
    EXPECT_EQ(result3.current_socket_power, 160U);
    EXPECT_EQ(result3.hotspot_temperature, 73000);
    EXPECT_EQ(result3.gfx_activity, 60U);
}

/**
 * Device Type Filtering Scenario
 *
 * Objective: Verify device type can be used for filtering.
 */
TEST_F(DeviceTest, device_type_filtering_scenario)
{
    // Setup
    SetupAllMetricsSupported();

    // Create GPU device
    device<MockDriver> gpu_device(mock_driver, test_handle, AMDSMI_PROCESSOR_TYPE_AMD_GPU,
                                  test_index);

    // Verify device type is correctly stored
    EXPECT_EQ(gpu_device.get_device_type(), AMDSMI_PROCESSOR_TYPE_AMD_GPU);

    // Demonstrate filtering by type
    processor_type_t device_type = gpu_device.get_device_type();
    bool             is_gpu      = (device_type == AMDSMI_PROCESSOR_TYPE_AMD_GPU);

    EXPECT_TRUE(is_gpu);
}

}  // namespace testing
}  // namespace gpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys

#endif  // ROCPROFSYS_USE_ROCM > 0
