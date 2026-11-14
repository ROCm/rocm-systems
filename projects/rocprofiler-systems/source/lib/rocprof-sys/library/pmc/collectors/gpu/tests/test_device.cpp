// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/gpu/device.hpp"
#include "mock_gpu_backend.hpp"

#include <array>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

using namespace rocprofsys::pmc::collectors::gpu;
using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Throw;

using MockBackend =
    ::testing::StrictMock<rocprofsys::backends::amd_smi::testing::mock_gpu_backend>;

namespace rocprofsys::pmc::collectors::gpu::testing
{

/**
 * @brief Test fixture for GPU device tests.
 *
 * Provides common setup for device tests including mock backend and
 * helper methods for configuring mock behavior.
 */
class DeviceTest : public ::testing::Test
{
protected:
    std::shared_ptr<MockBackend> mock_backend;
    size_t                       test_index;

    void SetUp() override
    {
        mock_backend = std::make_shared<MockBackend>();
        test_index   = 0;

        EXPECT_CALL(*mock_backend, get_gpu_asic_info())
            .Times(AnyNumber())
            .WillRepeatedly(
                Return(asic_info{ .product_name = "Test GPU", .vendor_name = "AMD" }));

        EXPECT_CALL(*mock_backend, get_hotspot_temperature())
            .Times(AnyNumber())
            .WillRepeatedly(Throw(std::runtime_error("temperature not supported")));
        EXPECT_CALL(*mock_backend, get_edge_temperature())
            .Times(AnyNumber())
            .WillRepeatedly(Throw(std::runtime_error("temperature not supported")));
    }

    /**
     * @brief Setup SDMA mock expectations for any device mock.
     * Call this for any mock that will have devices constructed with it.
     */
    template <typename MockPtr>
    static void SetupSDMAExpectations(MockPtr& mock)
    {
        EXPECT_CALL(*mock, probe_sdma_gpu_support())
            .Times(AnyNumber())
            .WillRepeatedly(Return(true));

        EXPECT_CALL(*mock, get_raw_sdma_usage())
            .Times(AnyNumber())
            .WillRepeatedly(Return(0));
    }

    /**
     * @brief Configure mock to return GPU metrics with all valid values.
     */
    void SetupAllMetricsSupported()
    {
        const metrics met = CreateValidMetrics();

        EXPECT_CALL(*mock_backend, get_metrics())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(met));

        EXPECT_CALL(*mock_backend, get_memory_usage())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(8589934592ULL));

        EXPECT_CALL(*mock_backend, get_hotspot_temperature())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(std::int64_t{ 75 }));
        EXPECT_CALL(*mock_backend, get_edge_temperature())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(std::int64_t{ 70 }));

        SetupSDMAExpectations(mock_backend);
    }

    /**
     * @brief Configure mock to return GPU metrics with all sentinel values.
     */
    void SetupNoMetricsSupported()
    {
        const metrics met = CreateSentinelMetrics();

        EXPECT_CALL(*mock_backend, get_metrics())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(met));

        EXPECT_CALL(*mock_backend, get_memory_usage())
            .Times(AtLeast(1))
            .WillRepeatedly(Throw(std::runtime_error("not supported")));

        EXPECT_CALL(*mock_backend, probe_sdma_gpu_support())
            .Times(AnyNumber())
            .WillRepeatedly(Return(false));

        EXPECT_CALL(*mock_backend, get_raw_sdma_usage())
            .Times(AnyNumber())
            .WillRepeatedly(Return(0));
    }

    /**
     * @brief Configure mock to return GPU metrics with partial support.
     *
     * Returns valid values for:
     * - current_socket_power
     * - hotspot_temperature
     * - gfx_activity
     *
     * All other metrics return sentinel values.
     */
    void SetupPartialMetricsSupported()
    {
        metrics met = CreateSentinelMetrics();

        met.current_socket_power = 150;
        met.gfx_activity         = 85;

        EXPECT_CALL(*mock_backend, get_metrics())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(met));

        EXPECT_CALL(*mock_backend, get_memory_usage())
            .Times(AtLeast(1))
            .WillRepeatedly(Throw(std::runtime_error("not supported")));

        EXPECT_CALL(*mock_backend, get_hotspot_temperature())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(std::int64_t{ 75 }));

        SetupSDMAExpectations(mock_backend);
    }

    /**
     * @brief Default temperature expectations for a mock not using fixture SetUp().
     */
    template <typename MockPtr>
    static void SetupTemperatureExpectationsUnsupported(MockPtr& m)
    {
        EXPECT_CALL(*m, get_hotspot_temperature())
            .Times(AnyNumber())
            .WillRepeatedly(Throw(std::runtime_error("temperature not supported")));
        EXPECT_CALL(*m, get_edge_temperature())
            .Times(AnyNumber())
            .WillRepeatedly(Throw(std::runtime_error("temperature not supported")));
    }

    /**
     * @brief Create metrics with all valid (non-sentinel) values.
     */
    static metrics CreateValidMetrics()
    {
        metrics met{};

        met.current_socket_power = 150;
        met.average_socket_power = 140;
        met.hotspot_temperature  = 75;
        met.edge_temperature     = 70;
        met.gfx_activity         = 85;
        met.umc_activity         = 60;
        met.mm_activity          = 40;

        for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
        {
            for(size_t i = 0; i < MAX_NUM_VCN; ++i)
            {
                met.xcp_stats[xcp].vcn_busy[i] = static_cast<std::uint16_t>(50 + i);
            }
            for(size_t i = 0; i < MAX_NUM_JPEG_V1; ++i)
            {
                met.xcp_stats[xcp].jpeg_busy[i] = static_cast<std::uint16_t>(30 + i);
            }
        }

        met.xgmi.link.width = 16;
        met.xgmi.link.speed = 25000;
        for(size_t i = 0; i < MAX_NUM_XGMI_LINKS; ++i)
        {
            met.xgmi.data_acc.read[i]  = 1000000ULL + i;
            met.xgmi.data_acc.write[i] = 2000000ULL + i;
        }

        met.pcie.link.width     = 16;
        met.pcie.link.speed     = 16000;
        met.pcie.bandwidth.acc  = 500000000ULL;
        met.pcie.bandwidth.inst = 10000000ULL;

        met.gfx_clock_mhz = 1500;
        met.mem_clock_mhz = 1200;

        return met;
    }

    /**
     * @brief Create metrics with all sentinel values.
     */
    static metrics CreateSentinelMetrics()
    {
        metrics met{};

        met.current_socket_power = 0xFFFF;
        met.average_socket_power = 0xFFFF;
        met.gfx_activity         = 0xFFFF;
        met.umc_activity         = 0xFFFF;
        met.mm_activity          = 0xFFFF;

        met.hotspot_temperature = 0xFFFF;
        met.edge_temperature    = 0xFFFF;

        for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
        {
            for(size_t i = 0; i < MAX_NUM_VCN; ++i)
            {
                met.xcp_stats[xcp].vcn_busy[i] = 0xFFFF;
            }
            for(size_t i = 0; i < MAX_NUM_JPEG_V1; ++i)
            {
                met.xcp_stats[xcp].jpeg_busy[i] = 0xFFFF;
            }
        }

        for(size_t i = 0; i < MAX_NUM_VCN; ++i)
        {
            met.vcn_activity[i] = 0xFFFF;
        }
        for(size_t i = 0; i < MAX_NUM_JPEG; ++i)
        {
            met.jpeg_activity[i] = 0xFFFF;
        }

        met.xgmi.link.width = 0xFFFF;
        met.xgmi.link.speed = 0xFFFF;
        for(size_t i = 0; i < MAX_NUM_XGMI_LINKS; ++i)
        {
            met.xgmi.data_acc.read[i]  = 0xFFFFFFFFFFFFFFFFULL;
            met.xgmi.data_acc.write[i] = 0xFFFFFFFFFFFFFFFFULL;
        }

        met.pcie.link.width     = 0xFFFF;
        met.pcie.link.speed     = 0xFFFF;
        met.pcie.bandwidth.acc  = 0xFFFFFFFFFFFFFFFFULL;
        met.pcie.bandwidth.inst = 0xFFFFFFFFFFFFFFFFULL;

        met.gfx_clock_mhz = 0xFFFF;
        met.mem_clock_mhz = 0xFFFF;

        return met;
    }
};

// ============================================================================
// Category 1: Constructor and Initialization Tests
// ============================================================================

/**
 * TestCase1.1: Valid Device Construction with Full Metric Support
 *
 * Objective: Verify device initializes correctly when all metrics are supported.
 */
TEST_F(DeviceTest, valid_device_construction_full_support)
{
    SetupAllMetricsSupported();

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.is_supported());

    auto const supported = dev.get_supported_metrics();
    EXPECT_NE(supported.value, 0U);

    EXPECT_EQ(dev.get_index(), test_index);
}

/**
 * TestCase1.2: Device Construction with No Supported Metrics
 *
 * Objective: Verify device handles hardware with no supported metrics.
 */
TEST_F(DeviceTest, device_construction_no_support)
{
    SetupNoMetricsSupported();

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_FALSE(dev.is_supported());

    auto const supported = dev.get_supported_metrics();
    EXPECT_EQ(supported.value, 0U);

    auto met = dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);
    EXPECT_EQ(met.current_socket_power, 0U);
    EXPECT_EQ(met.average_socket_power, 0U);
    EXPECT_EQ(met.memory_usage, 0ULL);
}

/**
 * TestCase1.3: Device Construction with Partial Metric Support
 *
 * Objective: Verify selective metric initialization.
 */
TEST_F(DeviceTest, device_construction_partial_support)
{
    SetupPartialMetricsSupported();

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.is_supported());

    auto const supported = dev.get_supported_metrics();

    EXPECT_TRUE(supported.bits.current_socket_power);
    EXPECT_TRUE(supported.bits.hotspot_temperature);
    EXPECT_TRUE(supported.bits.gfx_activity);

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
 * TestCase1.4: Device Construction with Different Indices
 *
 * Objective: Verify device index is correctly stored for different device instances.
 */
TEST_F(DeviceTest, device_construction_different_indices)
{
    SetupAllMetricsSupported();

    {
        const device<MockBackend> dev(mock_backend, 0);
        EXPECT_EQ(dev.get_index(), 0U);
    }

    {
        const device<MockBackend> dev(mock_backend, 1);
        EXPECT_EQ(dev.get_index(), 1U);
    }

    {
        const device<MockBackend> dev(mock_backend, 2);
        EXPECT_EQ(dev.get_index(), 2U);
    }
}

// ============================================================================
// Category 2: Power Metrics Collection Tests
// ============================================================================

/**
 * TestCase2.1: Current Socket Power Collection
 *
 * Objective: Verify current power is collected when supported.
 */
TEST_F(DeviceTest, current_socket_power_collection)
{
    metrics met              = CreateSentinelMetrics();
    met.current_socket_power = 150;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.current_socket_power);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.current_socket_power, 150U);
}

/**
 * TestCase2.2: Average Socket Power Collection
 *
 * Objective: Verify average power is collected when supported.
 */
TEST_F(DeviceTest, average_socket_power_collection)
{
    metrics met              = CreateSentinelMetrics();
    met.average_socket_power = 140;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.average_socket_power);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.average_socket_power, 140U);
}

/**
 * TestCase2.3: Power Metrics Not Collected When Unsupported
 *
 * Objective: Verify power metrics remain zero when not supported.
 */
TEST_F(DeviceTest, power_metrics_not_collected_when_unsupported)
{
    SetupNoMetricsSupported();

    device<MockBackend> dev(mock_backend, test_index);

    auto const supported = dev.get_supported_metrics();
    EXPECT_FALSE(supported.bits.current_socket_power);
    EXPECT_FALSE(supported.bits.average_socket_power);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.current_socket_power, 0U);
    EXPECT_EQ(collected.average_socket_power, 0U);
}

// ============================================================================
// Category 3: Temperature Metrics Collection Tests
// ============================================================================

/**
 * TestCase2.4: Hotspot Temperature Collection
 *
 * Objective: Verify hotspot temperature collection.
 */
TEST_F(DeviceTest, hotspot_temperature_collection)
{
    const metrics met = CreateSentinelMetrics();

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    EXPECT_CALL(*mock_backend, get_hotspot_temperature())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::int64_t{ 75 }));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.hotspot_temperature);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.hotspot_temperature, 75);
}

/**
 * TestCase2.5: Edge Temperature Collection
 *
 * Objective: Verify edge temperature collection.
 */
TEST_F(DeviceTest, edge_temperature_collection)
{
    const metrics met = CreateSentinelMetrics();

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    EXPECT_CALL(*mock_backend, get_edge_temperature())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::int64_t{ 70 }));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.edge_temperature);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.edge_temperature, 70);
}

/**
 * TestCase2.6: Temperature Metrics Not Collected When Unsupported
 *
 * Objective: Verify temperature skipped when not supported.
 */
TEST_F(DeviceTest, temperature_metrics_not_collected_when_unsupported)
{
    SetupNoMetricsSupported();

    device<MockBackend> dev(mock_backend, test_index);

    auto const supported = dev.get_supported_metrics();
    EXPECT_FALSE(supported.bits.hotspot_temperature);
    EXPECT_FALSE(supported.bits.edge_temperature);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.hotspot_temperature, 0);
    EXPECT_EQ(collected.edge_temperature, 0);
}

// ============================================================================
// Category 4: Activity Metrics Collection Tests
// ============================================================================

TEST_F(DeviceTest, gfx_activity_collection)
{
    metrics met      = CreateSentinelMetrics();
    met.gfx_activity = 85;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.gfx_activity);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.gfx_activity, 85U);
}

TEST_F(DeviceTest, umc_activity_collection)
{
    metrics met      = CreateSentinelMetrics();
    met.umc_activity = 60;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.umc_activity);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.umc_activity, 60U);
}

TEST_F(DeviceTest, mm_activity_collection)
{
    metrics met     = CreateSentinelMetrics();
    met.mm_activity = 40;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.mm_activity);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.mm_activity, 40U);
}

TEST_F(DeviceTest, all_activity_metrics_collection)
{
    metrics met      = CreateSentinelMetrics();
    met.gfx_activity = 85;
    met.umc_activity = 60;
    met.mm_activity  = 40;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    auto const supported = dev.get_supported_metrics();
    EXPECT_TRUE(supported.bits.gfx_activity);
    EXPECT_TRUE(supported.bits.umc_activity);
    EXPECT_TRUE(supported.bits.mm_activity);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.gfx_activity, 85U);
    EXPECT_EQ(collected.umc_activity, 60U);
    EXPECT_EQ(collected.mm_activity, 40U);
}

// ============================================================================
// Category 5: Memory Usage Collection Tests
// ============================================================================

TEST_F(DeviceTest, vram_memory_usage_collection_success)
{
    const metrics met = CreateSentinelMetrics();

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(8589934592ULL));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.memory_usage);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.memory_usage, 8589934592ULL);
}

TEST_F(DeviceTest, memory_usage_collection_failure)
{
    const metrics met = CreateSentinelMetrics();

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_FALSE(dev.get_supported_metrics().bits.memory_usage);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.memory_usage, 0ULL);
}

TEST_F(DeviceTest, memory_usage_not_collected_when_unsupported)
{
    SetupNoMetricsSupported();

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_FALSE(dev.get_supported_metrics().bits.memory_usage);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.memory_usage, 0ULL);
}

// ============================================================================
// Category 6: XCP Metrics Collection Tests
// ============================================================================

TEST_F(DeviceTest, vcn_busy_collection_all_xcps)
{
    metrics met = CreateSentinelMetrics();

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < MAX_NUM_VCN; ++vcn)
        {
            met.xcp_stats[xcp].vcn_busy[vcn] = static_cast<std::uint16_t>(50 + xcp + vcn);
        }
    }

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.vcn_busy);
    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < MAX_NUM_VCN; ++vcn)
        {
            EXPECT_EQ(collected.xcp_stats[xcp].vcn_busy[vcn],
                      static_cast<std::uint16_t>(50 + xcp + vcn));
        }
    }
}

TEST_F(DeviceTest, jpeg_activity_collection_all_xcps)
{
    metrics met = CreateSentinelMetrics();

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t jpeg = 0; jpeg < MAX_NUM_JPEG_V1; ++jpeg)
        {
            met.xcp_stats[xcp].jpeg_busy[jpeg] =
                static_cast<std::uint16_t>(30 + xcp + jpeg);
        }
    }

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.jpeg_busy);
    EXPECT_FALSE(dev.get_supported_metrics().bits.jpeg_activity);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t jpeg = 0; jpeg < MAX_NUM_JPEG_V1; ++jpeg)
        {
            EXPECT_EQ(collected.xcp_stats[xcp].jpeg_busy[jpeg],
                      static_cast<std::uint16_t>(30 + xcp + jpeg));
        }
    }
}

TEST_F(DeviceTest, xcp_metrics_not_collected_when_unsupported)
{
    SetupNoMetricsSupported();

    device<MockBackend> dev(mock_backend, test_index);

    auto const supported = dev.get_supported_metrics();
    EXPECT_FALSE(supported.bits.vcn_busy);
    EXPECT_FALSE(supported.bits.jpeg_busy);
    EXPECT_FALSE(supported.bits.vcn_activity);
    EXPECT_FALSE(supported.bits.jpeg_activity);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < MAX_NUM_VCN; ++vcn)
        {
            EXPECT_EQ(collected.xcp_stats[xcp].vcn_busy[vcn], 0);
        }
        for(size_t jpeg = 0; jpeg < MAX_NUM_JPEG_V1; ++jpeg)
        {
            EXPECT_EQ(collected.xcp_stats[xcp].jpeg_busy[jpeg], 0);
        }
    }
}

TEST_F(DeviceTest, mixed_vcn_jpeg_support)
{
    metrics met = CreateSentinelMetrics();

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < MAX_NUM_VCN; ++vcn)
        {
            met.xcp_stats[xcp].vcn_busy[vcn] = static_cast<std::uint16_t>(50 + vcn);
        }
    }

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    auto const supported = dev.get_supported_metrics();
    EXPECT_TRUE(supported.bits.vcn_busy);
    EXPECT_FALSE(supported.bits.jpeg_busy);
    EXPECT_FALSE(supported.bits.vcn_activity);
    EXPECT_FALSE(supported.bits.jpeg_activity);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < MAX_NUM_VCN; ++vcn)
        {
            EXPECT_EQ(collected.xcp_stats[xcp].vcn_busy[vcn],
                      static_cast<std::uint16_t>(50 + vcn));
        }
    }

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t jpeg = 0; jpeg < MAX_NUM_JPEG_V1; ++jpeg)
        {
            EXPECT_EQ(collected.xcp_stats[xcp].jpeg_busy[jpeg], 0);
        }
    }
}

// ============================================================================
// Category 7: XGMI Metrics Collection Tests
// ============================================================================

TEST_F(DeviceTest, xgmi_link_width_collection)
{
    metrics met         = CreateSentinelMetrics();
    met.xgmi.link.width = 16;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.xgmi.link.width, 16U);
}

TEST_F(DeviceTest, xgmi_link_speed_collection)
{
    metrics met         = CreateSentinelMetrics();
    met.xgmi.link.speed = 25;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.xgmi.link.speed, 25U);
}

TEST_F(DeviceTest, xgmi_read_write_data_collection_all_links)
{
    metrics met = CreateSentinelMetrics();

    for(size_t i = 0; i < MAX_NUM_XGMI_LINKS; ++i)
    {
        met.xgmi.data_acc.read[i]  = 1000000 + i * 1000;
        met.xgmi.data_acc.write[i] = 2000000 + i * 1000;
    }

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    for(size_t i = 0; i < MAX_NUM_XGMI_LINKS; ++i)
    {
        EXPECT_EQ(collected.xgmi.data_acc.read[i], 1000000 + i * 1000);
        EXPECT_EQ(collected.xgmi.data_acc.write[i], 2000000 + i * 1000);
    }
}

TEST_F(DeviceTest, xgmi_sentinel_value_handling)
{
    metrics met = CreateSentinelMetrics();

    met.xgmi.link.width        = 16;
    met.xgmi.data_acc.read[0]  = 1000000;
    met.xgmi.data_acc.write[0] = 2000000;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.xgmi.link.width, 16U);
    EXPECT_EQ(collected.xgmi.link.speed, 0U);
    EXPECT_EQ(collected.xgmi.data_acc.read[0], 1000000U);
    EXPECT_EQ(collected.xgmi.data_acc.read[1], 0U);
    EXPECT_EQ(collected.xgmi.data_acc.write[0], 2000000U);
    EXPECT_EQ(collected.xgmi.data_acc.write[1], 0U);
}

TEST_F(DeviceTest, xgmi_not_collected_when_unsupported)
{
    SetupNoMetricsSupported();

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_FALSE(dev.get_supported_metrics().bits.xgmi);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.xgmi.link.width, 0U);
    EXPECT_EQ(collected.xgmi.link.speed, 0U);

    for(size_t i = 0; i < MAX_NUM_XGMI_LINKS; ++i)
    {
        EXPECT_EQ(collected.xgmi.data_acc.read[i], 0U);
        EXPECT_EQ(collected.xgmi.data_acc.write[i], 0U);
    }
}

// ============================================================================
// Category 8: PCIe Metrics Collection Tests
// ============================================================================

TEST_F(DeviceTest, pcie_link_width_collection)
{
    metrics met         = CreateSentinelMetrics();
    met.pcie.link.width = 16;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.pcie.link.width, 16U);
}

TEST_F(DeviceTest, pcie_link_speed_collection)
{
    metrics met         = CreateSentinelMetrics();
    met.pcie.link.speed = 16000;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.pcie.link.speed, 16000U);
}

TEST_F(DeviceTest, pcie_bandwidth_accumulator_collection)
{
    metrics met            = CreateSentinelMetrics();
    met.pcie.bandwidth.acc = 500000000;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.pcie.bandwidth.acc, 500000000U);
}

TEST_F(DeviceTest, pcie_bandwidth_instantaneous_collection)
{
    metrics met             = CreateSentinelMetrics();
    met.pcie.bandwidth.inst = 10000000;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.pcie.bandwidth.inst, 10000000U);
}

TEST_F(DeviceTest, pcie_sentinel_value_handling)
{
    metrics met            = CreateSentinelMetrics();
    met.pcie.link.width    = 16;
    met.pcie.bandwidth.acc = 500000000;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.pcie.link.width, 16U);
    EXPECT_EQ(collected.pcie.link.speed, 0U);
    EXPECT_EQ(collected.pcie.bandwidth.acc, 500000000U);
    EXPECT_EQ(collected.pcie.bandwidth.inst, 0U);
}

TEST_F(DeviceTest, pcie_not_collected_when_unsupported)
{
    SetupNoMetricsSupported();

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_FALSE(dev.get_supported_metrics().bits.pcie);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.pcie.link.width, 0U);
    EXPECT_EQ(collected.pcie.link.speed, 0U);
    EXPECT_EQ(collected.pcie.bandwidth.acc, 0U);
    EXPECT_EQ(collected.pcie.bandwidth.inst, 0U);
}

// ============================================================================
// Category 9: Supported Metrics Detection Tests
// ============================================================================

TEST_F(DeviceTest, all_metrics_supported_detection)
{
    SetupAllMetricsSupported();

    const device<MockBackend> dev(mock_backend, test_index);

    auto const supported = dev.get_supported_metrics();
    EXPECT_TRUE(supported.bits.current_socket_power);
    EXPECT_TRUE(supported.bits.average_socket_power);
    EXPECT_TRUE(supported.bits.memory_usage);
    EXPECT_TRUE(supported.bits.hotspot_temperature);
    EXPECT_TRUE(supported.bits.edge_temperature);
    EXPECT_TRUE(supported.bits.gfx_activity);
    EXPECT_TRUE(supported.bits.umc_activity);
    EXPECT_TRUE(supported.bits.mm_activity);
    EXPECT_TRUE(supported.bits.vcn_busy);
    EXPECT_TRUE(supported.bits.jpeg_busy);
    EXPECT_FALSE(supported.bits.vcn_activity);
    EXPECT_FALSE(supported.bits.jpeg_activity);
    EXPECT_TRUE(supported.bits.xgmi);
    EXPECT_TRUE(supported.bits.pcie);
    EXPECT_TRUE(supported.bits.sdma_usage);
    EXPECT_TRUE(supported.bits.gfx_clock);
    EXPECT_TRUE(supported.bits.mem_clock);
}

TEST_F(DeviceTest, vcn_activity_support_detection_any_xcp)
{
    metrics met = CreateSentinelMetrics();

    met.xcp_stats[7].vcn_busy[0] = 50;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.vcn_busy);
    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity);
}

TEST_F(DeviceTest, vcn_activity_unsupported_all_sentinels)
{
    SetupNoMetricsSupported();

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity);
}

TEST_F(DeviceTest, jpeg_activity_support_detection_any_xcp)
{
    metrics met = CreateSentinelMetrics();

    met.xcp_stats[5].jpeg_busy[0] = 75;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.jpeg_busy);
    EXPECT_FALSE(dev.get_supported_metrics().bits.jpeg_activity);
}

TEST_F(DeviceTest, xgmi_support_detection_link_width_only)
{
    metrics met         = CreateSentinelMetrics();
    met.xgmi.link.width = 16;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);
}

TEST_F(DeviceTest, xgmi_support_detection_any_read_data_valid)
{
    metrics met               = CreateSentinelMetrics();
    met.xgmi.data_acc.read[2] = 1000;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.xgmi);
}

TEST_F(DeviceTest, pcie_support_detection_bandwidth_only)
{
    metrics met            = CreateSentinelMetrics();
    met.pcie.bandwidth.acc = 1000000;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.pcie);
}

TEST_F(DeviceTest, memory_usage_support_detection)
{
    const metrics met = CreateSentinelMetrics();

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(4096000000ULL));

    SetupSDMAExpectations(mock_backend);

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.memory_usage);
}

TEST_F(DeviceTest, memory_usage_unsupported_api_failure)
{
    const metrics met = CreateSentinelMetrics();

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_FALSE(dev.get_supported_metrics().bits.memory_usage);
}

// ============================================================================
// Category 10: VCN Activity Dual Source Tests
// ============================================================================

TEST_F(DeviceTest, vcn_activity_top_level_field_only)
{
    const metrics met = CreateSentinelMetrics();

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity)
        << "BUG: Implementation does not check top-level vcn_activity[] field";
}

TEST_F(DeviceTest, vcn_activity_in_both_fields)
{
    metrics met = CreateSentinelMetrics();

    met.xcp_stats[0].vcn_busy[0] = 80;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.vcn_busy);
    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.xcp_stats[0].vcn_busy[0], 80U);
}

TEST_F(DeviceTest, vcn_activity_detection_should_check_both_sources)
{
    const metrics met = CreateSentinelMetrics();

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity)
        << "Implementation gap: initialize_supported_metrics() should check both "
           "vcn_activity[] AND xcp_stats[].vcn_busy[]";
}

TEST_F(DeviceTest, vcn_activity_collection_priority)
{
    metrics met = CreateSentinelMetrics();

    met.xcp_stats[0].vcn_busy[0] = 80;
    met.xcp_stats[0].vcn_busy[1] = 70;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.xcp_stats[0].vcn_busy[0], 80U);
    EXPECT_EQ(collected.xcp_stats[0].vcn_busy[1], 70U);
}

TEST_F(DeviceTest, vcn_activity_xcp_disabled_top_level_valid)
{
    const metrics met = CreateSentinelMetrics();

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_activity);
}

// ============================================================================
// Category 11: Error Handling and Edge Cases
// ============================================================================

TEST_F(DeviceTest, get_metrics_info_failure)
{
    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(4096000000ULL));

    EXPECT_CALL(*mock_backend, get_raw_sdma_usage())
        .Times(AnyNumber())
        .WillRepeatedly(Return(0));

    device<MockBackend> dev(mock_backend, test_index);

    auto met = dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(met.current_socket_power, 0U);
    EXPECT_EQ(met.average_socket_power, 0U);
    EXPECT_EQ(met.hotspot_temperature, 0);
    EXPECT_EQ(met.edge_temperature, 0);
    EXPECT_EQ(met.gfx_activity, 0U);
}

TEST_F(DeviceTest, get_metrics_info_failure_during_init)
{
    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(4096000000ULL));

    EXPECT_CALL(*mock_backend, get_raw_sdma_usage())
        .Times(AnyNumber())
        .WillRepeatedly(Return(0));

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.is_supported());

    auto const supported = dev.get_supported_metrics();
    EXPECT_TRUE(supported.bits.memory_usage);
    EXPECT_FALSE(supported.bits.current_socket_power);
}

TEST_F(DeviceTest, multiple_metric_collections)
{
    SetupAllMetricsSupported();

    device<MockBackend> dev(mock_backend, test_index);

    for(int i = 0; i < 10; ++i)
    {
        auto met = dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);
        EXPECT_GT(met.current_socket_power, 0U);
    }
}

TEST_F(DeviceTest, large_array_indices_xgmi)
{
    metrics met = CreateSentinelMetrics();

    for(size_t i = 0; i < MAX_NUM_XGMI_LINKS; ++i)
    {
        met.xgmi.data_acc.read[i]  = 1000 + i;
        met.xgmi.data_acc.write[i] = 2000 + i;
    }

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    for(size_t i = 0; i < MAX_NUM_XGMI_LINKS; ++i)
    {
        EXPECT_EQ(collected.xgmi.data_acc.read[i], 1000 + i);
        EXPECT_EQ(collected.xgmi.data_acc.write[i], 2000 + i);
    }
}

TEST_F(DeviceTest, large_array_indices_xcp)
{
    metrics met = CreateSentinelMetrics();

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < MAX_NUM_VCN; ++vcn)
        {
            met.xcp_stats[xcp].vcn_busy[vcn] = static_cast<std::uint16_t>(xcp * 10 + vcn);
        }
    }

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t vcn = 0; vcn < MAX_NUM_VCN; ++vcn)
        {
            EXPECT_EQ(collected.xcp_stats[xcp].vcn_busy[vcn],
                      static_cast<std::uint16_t>(xcp * 10 + vcn));
        }
    }
}

TEST_F(DeviceTest, large_array_indices_jpeg)
{
    metrics met = CreateSentinelMetrics();

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t jpeg = 0; jpeg < MAX_NUM_JPEG_V1; ++jpeg)
        {
            met.xcp_stats[xcp].jpeg_busy[jpeg] =
                static_cast<std::uint16_t>(xcp * 100 + jpeg);
        }
    }

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t jpeg = 0; jpeg < MAX_NUM_JPEG_V1; ++jpeg)
        {
            EXPECT_EQ(collected.xcp_stats[xcp].jpeg_busy[jpeg],
                      static_cast<std::uint16_t>(xcp * 100 + jpeg));
        }
    }
}

TEST_F(DeviceTest, concurrent_device_objects)
{
    auto mock_backend1 = std::make_shared<MockBackend>();
    auto mock_backend2 = std::make_shared<MockBackend>();

    metrics met1              = CreateSentinelMetrics();
    met1.current_socket_power = 100;

    EXPECT_CALL(*mock_backend1, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met1));

    EXPECT_CALL(*mock_backend1, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend1);

    EXPECT_CALL(*mock_backend1, get_gpu_asic_info())
        .Times(AnyNumber())
        .WillRepeatedly(
            Return(asic_info{ .product_name = "GPU1", .vendor_name = "AMD" }));

    SetupTemperatureExpectationsUnsupported(mock_backend1);

    metrics met2              = CreateSentinelMetrics();
    met2.current_socket_power = 200;

    EXPECT_CALL(*mock_backend2, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met2));

    EXPECT_CALL(*mock_backend2, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend2);

    EXPECT_CALL(*mock_backend2, get_gpu_asic_info())
        .Times(AnyNumber())
        .WillRepeatedly(
            Return(asic_info{ .product_name = "GPU2", .vendor_name = "AMD" }));

    SetupTemperatureExpectationsUnsupported(mock_backend2);

    device<MockBackend> dev1(mock_backend1, 0);
    device<MockBackend> dev2(mock_backend2, 1);

    auto result1 =
        dev1.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);
    EXPECT_EQ(result1.current_socket_power, 100U);

    auto result2 =
        dev2.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);
    EXPECT_EQ(result2.current_socket_power, 200U);

    result1 = dev1.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);
    EXPECT_EQ(result1.current_socket_power, 100U);

    EXPECT_NE(dev1.get_index(), dev2.get_index());
}

TEST_F(DeviceTest, device_with_index_zero)
{
    SetupAllMetricsSupported();

    const device<MockBackend> dev(mock_backend, 0);

    EXPECT_EQ(dev.get_index(), 0U);
}

TEST_F(DeviceTest, device_with_high_index)
{
    SetupAllMetricsSupported();

    const device<MockBackend> dev(mock_backend, 15);

    EXPECT_EQ(dev.get_index(), 15U);
}

// ============================================================================
// Category 12: Integration Tests
// ============================================================================

TEST_F(DeviceTest, full_lifecycle_with_realistic_data)
{
    auto mock = std::make_shared<MockBackend>();

    metrics init_met              = CreateSentinelMetrics();
    init_met.current_socket_power = 150;
    init_met.hotspot_temperature  = 70;
    init_met.gfx_activity         = 50;

    metrics met1              = CreateSentinelMetrics();
    met1.current_socket_power = 150;
    met1.hotspot_temperature  = 70;
    met1.gfx_activity         = 50;

    metrics met2              = CreateSentinelMetrics();
    met2.current_socket_power = 180;
    met2.hotspot_temperature  = 75;
    met2.gfx_activity         = 90;

    metrics met3              = CreateSentinelMetrics();
    met3.current_socket_power = 160;
    met3.hotspot_temperature  = 73;
    met3.gfx_activity         = 60;

    EXPECT_CALL(*mock, get_metrics())
        .WillOnce(Return(init_met))
        .WillOnce(Return(met1))
        .WillOnce(Return(met2))
        .WillOnce(Return(met3));

    EXPECT_CALL(*mock, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock);

    EXPECT_CALL(*mock, get_gpu_asic_info())
        .Times(AnyNumber())
        .WillRepeatedly(
            Return(asic_info{ .product_name = "Test GPU", .vendor_name = "AMD" }));

    EXPECT_CALL(*mock, get_hotspot_temperature())
        .WillOnce(Return(std::int64_t{ 70 }))
        .WillOnce(Return(std::int64_t{ 70 }))
        .WillOnce(Return(std::int64_t{ 75 }))
        .WillOnce(Return(std::int64_t{ 73 }));
    EXPECT_CALL(*mock, get_edge_temperature())
        .Times(AnyNumber())
        .WillRepeatedly(Throw(std::runtime_error("temperature not supported")));

    device<MockBackend> dev(mock, test_index);

    auto result1 = dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);
    EXPECT_EQ(result1.current_socket_power, 150U);
    EXPECT_EQ(result1.hotspot_temperature, 70);
    EXPECT_EQ(result1.gfx_activity, 50U);

    auto result2 = dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);
    EXPECT_EQ(result2.current_socket_power, 180U);
    EXPECT_EQ(result2.hotspot_temperature, 75);
    EXPECT_EQ(result2.gfx_activity, 90U);

    auto result3 = dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);
    EXPECT_EQ(result3.current_socket_power, 160U);
    EXPECT_EQ(result3.hotspot_temperature, 73);
    EXPECT_EQ(result3.gfx_activity, 60U);
}

/**
 * TestCase12.1: SDMA Delta Computation
 *
 * Objective: Verify SDMA usage percentage is computed correctly from deltas.
 */
TEST_F(DeviceTest, sdma_delta_computation)
{
    SetupAllMetricsSupported();

    EXPECT_CALL(*mock_backend, get_raw_sdma_usage())
        .WillOnce(Return(5000000ULL))
        .WillOnce(Return(15000000ULL));

    device<MockBackend> dev(mock_backend, test_index);
    ASSERT_TRUE(dev.is_supported());
    ASSERT_TRUE(dev.get_supported_metrics().bits.sdma_usage);

    enabled_metrics enabled;
    enabled.bits.sdma_usage = 1;

    auto metrics1 = dev.get_metrics(enabled, 1000000000ULL);
    EXPECT_EQ(metrics1.sdma_usage, 0U);

    auto metrics2 = dev.get_metrics(enabled, 2000000000ULL);
    EXPECT_GE(metrics2.sdma_usage, 0U);
    EXPECT_LE(metrics2.sdma_usage, 100U);
}

// ============================================================================
// Category 13: Clock Metrics Collection Tests (gfx_clock / mem_clock)
// ============================================================================

TEST_F(DeviceTest, gfx_clock_collection)
{
    metrics met       = CreateSentinelMetrics();
    met.gfx_clock_mhz = 1500;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.gfx_clock);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.gfx_clock_mhz, 1500U);
}

TEST_F(DeviceTest, mem_clock_collection)
{
    metrics met       = CreateSentinelMetrics();
    met.mem_clock_mhz = 1200;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Throw(std::runtime_error("not supported")));

    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.mem_clock);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.mem_clock_mhz, 1200U);
}

TEST_F(DeviceTest, clock_metrics_not_collected_when_unsupported)
{
    SetupNoMetricsSupported();

    device<MockBackend> dev(mock_backend, test_index);

    auto const supported = dev.get_supported_metrics();
    EXPECT_FALSE(supported.bits.gfx_clock);
    EXPECT_FALSE(supported.bits.mem_clock);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.gfx_clock_mhz, 0U);
    EXPECT_EQ(collected.mem_clock_mhz, 0U);
}

// Sentinel-preservation regression tests (originally added in PR #5145).
// These verify that the device layer copies arrays verbatim — including
// per-engine sentinel values — so the processor layer can filter them.

TEST_F(DeviceTest, vcn_busy_collection_preserves_sentinels)
{
    constexpr std::uint16_t SENTINEL_16 = 0xFFFF;
    metrics                 met         = CreateSentinelMetrics();

    met.xcp_stats[0].vcn_busy[0] = 80;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));
    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AnyNumber())
        .WillRepeatedly(Throw(std::runtime_error("not supported")));
    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.vcn_busy);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.xcp_stats[0].vcn_busy[0], 80U);
    for(size_t vcn = 1; vcn < MAX_NUM_VCN; ++vcn)
    {
        EXPECT_EQ(collected.xcp_stats[0].vcn_busy[vcn], SENTINEL_16)
            << "Sentinel in vcn_busy[" << vcn
            << "] must be preserved for processor-layer filtering";
    }
}

TEST_F(DeviceTest, jpeg_busy_collection_preserves_sentinels)
{
    constexpr std::uint16_t SENTINEL_16 = 0xFFFF;
    metrics                 met         = CreateSentinelMetrics();

    met.xcp_stats[0].jpeg_busy[0] = 60;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));
    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AnyNumber())
        .WillRepeatedly(Throw(std::runtime_error("not supported")));
    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.jpeg_busy);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.xcp_stats[0].jpeg_busy[0], 60U);
    for(size_t jpeg = 1; jpeg < MAX_NUM_JPEG_V1; ++jpeg)
    {
        EXPECT_EQ(collected.xcp_stats[0].jpeg_busy[jpeg], SENTINEL_16)
            << "Sentinel in jpeg_busy[" << jpeg
            << "] must be preserved for processor-layer filtering";
    }
}

TEST_F(DeviceTest, vcn_activity_device_level_preserves_sentinels)
{
    constexpr std::uint16_t SENTINEL_16 = 0xFFFF;
    metrics                 met         = CreateSentinelMetrics();

    met.vcn_activity[0] = 42;

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));
    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AnyNumber())
        .WillRepeatedly(Throw(std::runtime_error("not supported")));
    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);

    EXPECT_TRUE(dev.get_supported_metrics().bits.vcn_activity);
    EXPECT_FALSE(dev.get_supported_metrics().bits.vcn_busy);

    auto collected =
        dev.get_metrics(enabled_metrics{ .value = 0xFFFFFFFF }, 1000000000ULL);

    EXPECT_EQ(collected.vcn_activity[0], 42U);
    for(size_t i = 1; i < MAX_NUM_VCN; ++i)
    {
        EXPECT_EQ(collected.vcn_activity[i], SENTINEL_16)
            << "Sentinel in vcn_activity[" << i
            << "] must be preserved for processor-layer filtering";
    }
}

TEST_F(DeviceTest, memory_usage_unsupported_sentinel_value)
{
    const metrics met = CreateSentinelMetrics();

    EXPECT_CALL(*mock_backend, get_metrics())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(met));

    constexpr std::uint64_t SENTINEL_MEM = 0xFFFFFFFFFFFFFFFFULL;
    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(SENTINEL_MEM));

    SetupSDMAExpectations(mock_backend);

    const device<MockBackend> dev(mock_backend, test_index);

    EXPECT_FALSE(dev.get_supported_metrics().bits.memory_usage);
}

namespace
{
bool
power_activity_and_clocks_match(const metrics& got, const metrics& expected)
{
    return got.current_socket_power == expected.current_socket_power &&
           got.average_socket_power == expected.average_socket_power &&
           got.gfx_activity == expected.gfx_activity &&
           got.umc_activity == expected.umc_activity &&
           got.mm_activity == expected.mm_activity &&
           got.gfx_clock_mhz == expected.gfx_clock_mhz &&
           got.mem_clock_mhz == expected.mem_clock_mhz;
}

bool
codec_counters_match(const metrics& got, const metrics& expected)
{
    if(got.vcn_activity != expected.vcn_activity ||
       got.jpeg_activity != expected.jpeg_activity)
    {
        return false;
    }
    for(size_t xcp = 0; xcp < got.xcp_stats.size(); ++xcp)
    {
        if(got.xcp_stats[xcp].vcn_busy != expected.xcp_stats[xcp].vcn_busy ||
           got.xcp_stats[xcp].jpeg_busy != expected.xcp_stats[xcp].jpeg_busy)
        {
            return false;
        }
    }
    return true;
}

bool
links_match(const metrics& got, const metrics& expected)
{
    return got.xgmi.link.width == expected.xgmi.link.width &&
           got.xgmi.link.speed == expected.xgmi.link.speed &&
           got.xgmi.data_acc.read == expected.xgmi.data_acc.read &&
           got.xgmi.data_acc.write == expected.xgmi.data_acc.write &&
           got.pcie.link.width == expected.pcie.link.width &&
           got.pcie.link.speed == expected.pcie.link.speed &&
           got.pcie.bandwidth.acc == expected.pcie.bandwidth.acc &&
           got.pcie.bandwidth.inst == expected.pcie.bandwidth.inst;
}

bool
usage_and_temperature_match(const metrics& got, const metrics& expected)
{
    return got.memory_usage == expected.memory_usage &&
           got.hotspot_temperature == expected.hotspot_temperature &&
           got.edge_temperature == expected.edge_temperature &&
           got.sdma_usage == expected.sdma_usage;
}

void
expect_metrics_equal(const metrics& got, const metrics& expected)
{
    if(power_activity_and_clocks_match(got, expected) &&
       codec_counters_match(got, expected) && links_match(got, expected) &&
       usage_and_temperature_match(got, expected))
    {
        return;
    }
    ADD_FAILURE() << "collected metrics differ from the fields that were enabled";
}
}  // namespace

// An enabled bit copies that field. Anything left off stays at the default.
TEST_F(DeviceTest, enabled_bit_copies_only_power_activity_or_clock)
{
    metrics raw      = CreateValidMetrics();
    raw.memory_usage = raw.xgmi.data_acc.read[0];

    EXPECT_CALL(*mock_backend, get_metrics()).WillRepeatedly([&raw] { return raw; });
    EXPECT_CALL(*mock_backend, get_memory_usage()).WillRepeatedly([&raw] {
        return raw.memory_usage;
    });
    EXPECT_CALL(*mock_backend, get_hotspot_temperature()).WillRepeatedly([&raw] {
        return raw.hotspot_temperature;
    });
    EXPECT_CALL(*mock_backend, get_edge_temperature()).WillRepeatedly([&raw] {
        return raw.edge_temperature;
    });
    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);
    expect_metrics_equal(dev.get_metrics({}, 0), metrics{});

    enabled_metrics mask{};
    metrics         expected{};

    mask.bits.current_socket_power = 1;
    expected.current_socket_power  = raw.current_socket_power;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask                           = {};
    expected                       = {};
    mask.bits.average_socket_power = 1;
    expected.average_socket_power  = raw.average_socket_power;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask                   = {};
    expected               = {};
    mask.bits.gfx_activity = 1;
    expected.gfx_activity  = raw.gfx_activity;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask                   = {};
    expected               = {};
    mask.bits.umc_activity = 1;
    expected.umc_activity  = raw.umc_activity;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask                  = {};
    expected              = {};
    mask.bits.mm_activity = 1;
    expected.mm_activity  = raw.mm_activity;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask                   = {};
    expected               = {};
    mask.bits.gfx_clock    = 1;
    expected.gfx_clock_mhz = raw.gfx_clock_mhz;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask                   = {};
    expected               = {};
    mask.bits.mem_clock    = 1;
    expected.mem_clock_mhz = raw.mem_clock_mhz;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);
}

TEST_F(DeviceTest, enabled_bit_copies_only_codec_link_memory_or_temperature)
{
    metrics raw      = CreateValidMetrics();
    raw.memory_usage = raw.xgmi.data_acc.read[0];

    EXPECT_CALL(*mock_backend, get_metrics()).WillRepeatedly([&raw] { return raw; });
    EXPECT_CALL(*mock_backend, get_memory_usage()).WillRepeatedly([&raw] {
        return raw.memory_usage;
    });
    EXPECT_CALL(*mock_backend, get_hotspot_temperature()).WillRepeatedly([&raw] {
        return raw.hotspot_temperature;
    });
    EXPECT_CALL(*mock_backend, get_edge_temperature()).WillRepeatedly([&raw] {
        return raw.edge_temperature;
    });
    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);
    enabled_metrics     mask{};
    metrics             expected{};

    mask.bits.vcn_busy = 1;
    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        expected.xcp_stats[xcp].vcn_busy = raw.xcp_stats[xcp].vcn_busy;
    }
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask                = {};
    expected            = {};
    mask.bits.jpeg_busy = 1;
    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        expected.xcp_stats[xcp].jpeg_busy = raw.xcp_stats[xcp].jpeg_busy;
    }
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask           = {};
    expected       = {};
    mask.bits.xgmi = 1;
    expected.xgmi  = raw.xgmi;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask           = {};
    expected       = {};
    mask.bits.pcie = 1;
    expected.pcie  = raw.pcie;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask                   = {};
    expected               = {};
    mask.bits.memory_usage = 1;
    expected.memory_usage  = raw.memory_usage;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask                          = {};
    expected                      = {};
    mask.bits.hotspot_temperature = 1;
    expected.hotspot_temperature  = raw.hotspot_temperature;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask                       = {};
    expected                   = {};
    mask.bits.edge_temperature = 1;
    expected.edge_temperature  = raw.edge_temperature;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);
}

// Device-level activity stays unsupported while the per-XCP busy counters are
// valid, so this uses a second device built from sentinel busy values.
TEST_F(DeviceTest, enabled_bit_copies_only_device_codec_activity)
{
    metrics    raw         = CreateValidMetrics();
    const auto vcn_sample  = static_cast<std::uint16_t>(raw.gfx_activity);
    const auto jpeg_sample = static_cast<std::uint16_t>(raw.mm_activity);
    raw                    = CreateSentinelMetrics();
    raw.vcn_activity[0]    = vcn_sample;
    raw.jpeg_activity[0]   = jpeg_sample;

    EXPECT_CALL(*mock_backend, get_metrics()).WillRepeatedly([&raw] { return raw; });
    EXPECT_CALL(*mock_backend, get_memory_usage())
        .Times(AnyNumber())
        .WillRepeatedly(Throw(std::runtime_error("not supported")));
    SetupSDMAExpectations(mock_backend);

    device<MockBackend> dev(mock_backend, test_index);
    enabled_metrics     mask{};
    metrics             expected{};

    mask.bits.vcn_activity = 1;
    expected.vcn_activity  = raw.vcn_activity;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);

    mask                    = {};
    expected                = {};
    mask.bits.jpeg_activity = 1;
    expected.jpeg_activity  = raw.jpeg_activity;
    expect_metrics_equal(dev.get_metrics(mask, 0), expected);
}

}  // namespace rocprofsys::pmc::collectors::gpu::testing
