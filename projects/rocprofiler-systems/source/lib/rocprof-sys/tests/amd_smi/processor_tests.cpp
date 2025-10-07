#include "library/amd_smi/processor.hpp"

#include <algorithm>
#include <amd_smi/amdsmi.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

// Mock driver API for processor testing
struct mock_processor_driver
{
    MOCK_METHOD(amdsmi_status_t, get_power_info,
                (amdsmi_processor_handle, amdsmi_power_info_t*), ());
    MOCK_METHOD(amdsmi_status_t, get_activity,
                (amdsmi_processor_handle, amdsmi_engine_usage_t*), ());
    MOCK_METHOD(amdsmi_status_t, get_memory_usage,
                (amdsmi_processor_handle, amdsmi_memory_type_t, uint64_t*), ());
    MOCK_METHOD(amdsmi_status_t, get_temperature_metric,
                (amdsmi_processor_handle, amdsmi_temperature_type_t,
                 amdsmi_temperature_metric_t, int64_t*),
                ());
    MOCK_METHOD(amdsmi_status_t, get_metrics_info,
                (amdsmi_processor_handle, amdsmi_gpu_metrics_t*), ());
};

class ProcessorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mock_driver      = std::make_shared<StrictMock<mock_processor_driver>>();
        processor_handle = reinterpret_cast<amdsmi_processor_handle>(0x12345);
        processor_type   = AMDSMI_PROCESSOR_TYPE_AMD_GPU;

        // Setup successful responses for all metrics

        expcted_power_info.average_socket_power = 150;
        expcted_power_info.current_socket_power = 140;
        uint64_t memory_usage                   = 8192;
        int64_t  temperature                    = 123;

        amdsmi_gpu_metrics_t gpu_metrics = {};
        // Initialize XCP VCN and JPEG activity arrays
        std::for_each(std::begin(gpu_metrics.xcp_stats), std::end(gpu_metrics.xcp_stats),
                      [](amdsmi_gpu_xcp_metrics_t& xcp_stats) mutable {
                          std::for_each(std::begin(xcp_stats.vcn_busy),
                                        std::end(xcp_stats.vcn_busy),
                                        [](auto& vcn_busy) { vcn_busy = 40; });
                          std::for_each(std::begin(xcp_stats.jpeg_busy),
                                        std::end(xcp_stats.jpeg_busy),
                                        [](auto& jpeg_busy) { jpeg_busy = 40; });
                      });

        ON_CALL(*mock_driver, get_power_info(_, _))
            .WillByDefault(DoAll(SetArgPointee<1>(expcted_power_info),
                                 Return(AMDSMI_STATUS_SUCCESS)));

        ON_CALL(*mock_driver, get_activity(_, _))
            .WillByDefault(DoAll(SetArgPointee<1>(expected_engine_usage),
                                 Return(AMDSMI_STATUS_SUCCESS)));

        ON_CALL(*mock_driver, get_memory_usage(_, _, _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(memory_usage), Return(AMDSMI_STATUS_SUCCESS)));

        ON_CALL(*mock_driver, get_temperature_metric(_, _, _, _))
            .WillByDefault(
                DoAll(SetArgPointee<3>(temperature), Return(AMDSMI_STATUS_SUCCESS)));

        ON_CALL(*mock_driver, get_metrics_info(_, _))
            .WillByDefault(
                DoAll(SetArgPointee<1>(gpu_metrics), Return(AMDSMI_STATUS_SUCCESS)));
    }

    void TearDown() override
    {
        test_processor.reset();
        mock_driver.reset();
    }

    std::shared_ptr<StrictMock<mock_processor_driver>> mock_driver;
    amdsmi_processor_handle                            processor_handle;
    processor_type_t                                   processor_type;
    std::unique_ptr<rocprofsys::amd_smi::processor<mock_processor_driver>> test_processor;

    amdsmi_power_info_t   expcted_power_info    = {};
    amdsmi_engine_usage_t expected_engine_usage = {};
};

TEST_F(ProcessorTest, ConstructorInitializesCorrectly)
{
    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info);

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);
    EXPECT_EQ(test_processor->get_processor_type(), AMDSMI_PROCESSOR_TYPE_AMD_GPU);
}

TEST_F(ProcessorTest, GetProcessorTypeReturnsCorrectType)
{
    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info);
    // Test with CPU type
    auto cpu_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, AMDSMI_PROCESSOR_TYPE_AMD_CPU, 0);
    EXPECT_EQ(cpu_processor->get_processor_type(), AMDSMI_PROCESSOR_TYPE_AMD_CPU);
}

TEST_F(ProcessorTest, GetSupportedMetricsAllSupported)
{
    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info);

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);
    auto metrics = test_processor->get_supported_metrics();

    EXPECT_TRUE(metrics.bits.average_socket_power);
    EXPECT_TRUE(metrics.bits.current_socket_power);
    EXPECT_TRUE(metrics.bits.gfx_activity);
    EXPECT_TRUE(metrics.bits.mm_activity);
    EXPECT_TRUE(metrics.bits.umc_activity);
    EXPECT_TRUE(metrics.bits.memory_usage);
    EXPECT_TRUE(metrics.bits.edge_temperature);
    EXPECT_TRUE(metrics.bits.hotspot_temperature);
}

TEST_F(ProcessorTest, GetSupportedMetricsPowerInfoFails)
{
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info);
    EXPECT_CALL(*mock_driver, get_power_info(processor_handle, _))
        .WillOnce(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    auto metrics = test_processor->get_supported_metrics();

    EXPECT_FALSE(metrics.bits.average_socket_power);
    EXPECT_FALSE(metrics.bits.current_socket_power);
    EXPECT_TRUE(metrics.bits.gfx_activity);
    EXPECT_TRUE(metrics.bits.memory_usage);
    EXPECT_TRUE(metrics.bits.edge_temperature);
    EXPECT_TRUE(metrics.bits.hotspot_temperature);
}

TEST_F(ProcessorTest, GetSmiMetricsSuccess)
{
    uint64_t             memory_usage = 8192;
    amdsmi_gpu_metrics_t gpu_metrics  = {};
    gpu_metrics.average_socket_power  = 150;
    gpu_metrics.current_socket_power  = 140;
    gpu_metrics.average_gfx_activity  = 75;
    gpu_metrics.average_umc_activity  = 50;
    gpu_metrics.average_mm_activity   = 60;
    gpu_metrics.temperature_hotspot   = 25;
    gpu_metrics.temperature_edge      = 35;

    std::for_each(
        std::begin(gpu_metrics.xcp_stats), std::end(gpu_metrics.xcp_stats),
        [](amdsmi_gpu_xcp_metrics_t& xcp_stats) mutable {
            std::for_each(std::begin(xcp_stats.vcn_busy), std::end(xcp_stats.vcn_busy),
                          [](auto& vcn_busy) { vcn_busy = 30; });
            std::for_each(std::begin(xcp_stats.jpeg_busy), std::end(xcp_stats.jpeg_busy),
                          [](auto& jpeg_busy) { jpeg_busy = 40; });
        });

    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_power_info);

    // Calls for get_smi_metrics
    EXPECT_CALL(*mock_driver, get_metrics_info(processor_handle, _))
        .Times(2)
        .WillRepeatedly(
            DoAll(SetArgPointee<1>(gpu_metrics), Return(AMDSMI_STATUS_SUCCESS)));

    EXPECT_CALL(*mock_driver, get_memory_usage(processor_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .Times(2)
        .WillRepeatedly(
            DoAll(SetArgPointee<2>(memory_usage), Return(AMDSMI_STATUS_SUCCESS)));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    auto metrics = test_processor->get_smi_metrics();

    EXPECT_EQ(metrics.average_socket_power, 150);
    EXPECT_EQ(metrics.current_socket_power, 140);
    EXPECT_EQ(metrics.memory_usage, 8192);
    EXPECT_EQ(metrics.gfx_activity, 75);
    EXPECT_EQ(metrics.umc_activity, 50);
    EXPECT_EQ(metrics.mm_activity, 60);
    EXPECT_EQ(metrics.hotspot_temperature, 25);
    EXPECT_EQ(metrics.edge_temperature, 35);
}

TEST_F(ProcessorTest, GetSmiMetricsGpuMetricsFails)
{
    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info(processor_handle, _))
        .Times(2)
        .WillRepeatedly(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    EXPECT_THROW(test_processor->get_smi_metrics(), std::runtime_error);
}

TEST_F(ProcessorTest, GetSmiMetricsMemoryUsageFails)
{
    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info).Times(2);
    EXPECT_CALL(*mock_driver, get_memory_usage(processor_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .Times(2)
        .WillRepeatedly(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    // Should not throw, just log error and continue
    EXPECT_NO_THROW(test_processor->get_smi_metrics());
}

TEST_F(ProcessorTest, GetIndexReturnsCorrectIndex)
{
    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info);

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 5);

    EXPECT_EQ(test_processor->get_index(), 5);
}

TEST_F(ProcessorTest, GetSupportedMetricsActivityFails)
{
    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info);
    EXPECT_CALL(*mock_driver, get_activity(processor_handle, _))
        .WillOnce(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    auto metrics = test_processor->get_supported_metrics();

    EXPECT_TRUE(metrics.bits.average_socket_power);
    EXPECT_TRUE(metrics.bits.current_socket_power);
    EXPECT_FALSE(metrics.bits.gfx_activity);
    EXPECT_FALSE(metrics.bits.mm_activity);
    EXPECT_FALSE(metrics.bits.umc_activity);
}

TEST_F(ProcessorTest, GetSupportedMetricsMemoryUsageFails)
{
    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info);
    EXPECT_CALL(*mock_driver, get_memory_usage(processor_handle, AMDSMI_MEM_TYPE_VRAM, _))
        .WillOnce(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    auto metrics = test_processor->get_supported_metrics();

    EXPECT_FALSE(metrics.bits.memory_usage);
}

TEST_F(ProcessorTest, GetSupportedMetricsHotspotTemperatureFails)
{
    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_metrics_info);
    EXPECT_CALL(*mock_driver,
                get_temperature_metric(processor_handle, AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
                                       AMDSMI_TEMP_CURRENT, _))
        .WillOnce(Return(AMDSMI_STATUS_NOT_SUPPORTED));
    EXPECT_CALL(*mock_driver,
                get_temperature_metric(processor_handle, AMDSMI_TEMPERATURE_TYPE_EDGE,
                                       AMDSMI_TEMP_CURRENT, _))
        .WillOnce(Return(AMDSMI_STATUS_SUCCESS));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    auto metrics = test_processor->get_supported_metrics();

    EXPECT_FALSE(metrics.bits.hotspot_temperature);
    EXPECT_TRUE(metrics.bits.edge_temperature);
}

TEST_F(ProcessorTest, GetSupportedMetricsEdgeTemperatureFails)
{
    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_metrics_info);
    EXPECT_CALL(*mock_driver,
                get_temperature_metric(processor_handle, AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
                                       AMDSMI_TEMP_CURRENT, _))
        .WillOnce(Return(AMDSMI_STATUS_SUCCESS));
    EXPECT_CALL(*mock_driver,
                get_temperature_metric(processor_handle, AMDSMI_TEMPERATURE_TYPE_EDGE,
                                       AMDSMI_TEMP_CURRENT, _))
        .WillOnce(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    auto metrics = test_processor->get_supported_metrics();

    EXPECT_TRUE(metrics.bits.hotspot_temperature);
    EXPECT_FALSE(metrics.bits.edge_temperature);
}

TEST_F(ProcessorTest, GetSupportedMetricsGpuMetricsInfoFails)
{
    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info(processor_handle, _))
        .WillRepeatedly(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    auto metrics = test_processor->get_supported_metrics();
    rocprofsys::amd_smi::smi_metric_options expected_metrics = { .bits{
        .current_socket_power = 1,
        .average_socket_power = 1,
        .memory_usage         = 1,
        .hotspot_temperature  = 1,
        .edge_temperature     = 1,
        .gfx_activity         = 1,
        .umc_activity         = 1,
        .mm_activity          = 1,
        .vcn_activity         = 0,
        .jpeg_activity        = 0 } };
    EXPECT_EQ(metrics.value, expected_metrics.value);
}

TEST_F(ProcessorTest, GetSupportedMetricsTemperatureNotSupported)
{
    int64_t unsupported_temp = rocprofsys::amd_smi::metric_value_not_supported;

    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_metrics_info);
    EXPECT_CALL(*mock_driver,
                get_temperature_metric(processor_handle, AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
                                       AMDSMI_TEMP_CURRENT, _))
        .WillOnce(
            DoAll(SetArgPointee<3>(unsupported_temp), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*mock_driver,
                get_temperature_metric(processor_handle, AMDSMI_TEMPERATURE_TYPE_EDGE,
                                       AMDSMI_TEMP_CURRENT, _))
        .WillOnce(
            DoAll(SetArgPointee<3>(unsupported_temp), Return(AMDSMI_STATUS_SUCCESS)));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    auto metrics = test_processor->get_supported_metrics();

    EXPECT_FALSE(metrics.bits.hotspot_temperature);
    EXPECT_FALSE(metrics.bits.edge_temperature);
}

TEST_F(ProcessorTest, GetSupportedMetricsPowerNotSupported)
{
    amdsmi_power_info_t power_info  = {};
    power_info.average_socket_power = rocprofsys::amd_smi::metric_value_not_supported;
    power_info.current_socket_power = rocprofsys::amd_smi::metric_value_not_supported;

    EXPECT_CALL(*mock_driver, get_activity);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info);
    EXPECT_CALL(*mock_driver, get_power_info(processor_handle, _))
        .WillOnce(DoAll(SetArgPointee<1>(power_info), Return(AMDSMI_STATUS_SUCCESS)));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    auto metrics = test_processor->get_supported_metrics();

    EXPECT_FALSE(metrics.bits.average_socket_power);
    EXPECT_FALSE(metrics.bits.current_socket_power);
}

TEST_F(ProcessorTest, GetSupportedMetricsGfxActivityNotSupported)
{
    amdsmi_engine_usage_t engine_usage = {};
    engine_usage.gfx_activity          = rocprofsys::amd_smi::metric_value_not_supported;

    EXPECT_CALL(*mock_driver, get_power_info);
    EXPECT_CALL(*mock_driver, get_memory_usage);
    EXPECT_CALL(*mock_driver, get_temperature_metric).Times(2);
    EXPECT_CALL(*mock_driver, get_metrics_info);
    EXPECT_CALL(*mock_driver, get_activity(processor_handle, _))
        .WillOnce(DoAll(SetArgPointee<1>(engine_usage), Return(AMDSMI_STATUS_SUCCESS)));

    test_processor =
        std::make_unique<rocprofsys::amd_smi::processor<mock_processor_driver>>(
            mock_driver, processor_handle, processor_type, 0);

    auto metrics = test_processor->get_supported_metrics();

    EXPECT_FALSE(metrics.bits.gfx_activity);
    // mm_activity and umc_activity should still be true as they don't check
    // values
    EXPECT_TRUE(metrics.bits.mm_activity);
    EXPECT_TRUE(metrics.bits.umc_activity);
}
