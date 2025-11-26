// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#include "library/amd_smi/metrics.hpp"
#include "library/amd_smi/test/gmock_mocks.hpp"
#include "library/amd_smi/test/testable_sampler.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace rocprofsys
{
namespace amd_smi
{
namespace testing
{
#if ROCPROFSYS_USE_ROCM > 0

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

class SamplerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create mocks (NiceMock suppresses warnings about uninteresting calls)
        driver   = std::make_shared<NiceMock<MockDriver>>();
        clock    = std::make_shared<NiceMock<MockClock>>();
        storage  = std::make_shared<NiceMock<MockStorage>>();
        state    = std::make_shared<NiceMock<MockState>>();
        gpu_caps = std::make_shared<NiceMock<MockGpuCapabilities>>();

        // Set up default behaviors
        driver->SetUpDefaults();
        clock->SetUpDefaults();
        storage->SetUpCapture();
        state->SetUpDefaults();
        gpu_caps->SetUpDefaults();

        // Create sampler
        sampler =
            std::make_unique<TestableSampler>(driver, clock, storage, state, gpu_caps);
    }

    std::shared_ptr<NiceMock<MockDriver>>          driver;
    std::shared_ptr<NiceMock<MockClock>>           clock;
    std::shared_ptr<NiceMock<MockStorage>>         storage;
    std::shared_ptr<NiceMock<MockState>>           state;
    std::shared_ptr<NiceMock<MockGpuCapabilities>> gpu_caps;
    std::unique_ptr<TestableSampler>               sampler;
};

// ============================================================================
// State Management Tests
// ============================================================================

TEST_F(SamplerTest, ReturnsNulloptWhenStateNotActive)
{
    EXPECT_CALL(*state, get_state()).WillOnce(Return(State::PreInit));

    settings s{};
    auto     result = sampler->sample(0, s);

    EXPECT_FALSE(result.has_value());
    EXPECT_FALSE(storage->has_samples());
}

TEST_F(SamplerTest, ReturnsNulloptInChildProcess)
{
    EXPECT_CALL(*state, is_child_process()).WillOnce(Return(true));

    settings s{};
    auto     result = sampler->sample(0, s);

    EXPECT_FALSE(result.has_value());
}

TEST_F(SamplerTest, ReturnsNulloptWhenDisabled)
{
    EXPECT_CALL(*state, get_state()).WillOnce(Return(State::Disabled));

    settings s{};
    auto     result = sampler->sample(0, s);

    EXPECT_FALSE(result.has_value());
}

TEST_F(SamplerTest, ReturnsNulloptWhenFinalized)
{
    EXPECT_CALL(*state, get_state()).WillOnce(Return(State::Finalized));

    settings s{};
    auto     result = sampler->sample(0, s);

    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Basic Metrics Collection Tests
// ============================================================================

TEST_F(SamplerTest, CollectsBasicMetricsWhenEnabled)
{
    amdsmi_engine_usage_t mock_usage{};
    mock_usage.gfx_activity = 75;
    mock_usage.umc_activity = 50;
    mock_usage.mm_activity  = 25;

    amdsmi_power_info_t mock_power{};
    mock_power.current_socket_power = 150;

    EXPECT_CALL(*driver, get_gpu_activity(_, _))
        .WillOnce(DoAll(SetArgPointee<1>(mock_usage), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*driver, get_temp_metric(_, _, _, _))
        .WillOnce(DoAll(SetArgPointee<3>(60), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*driver, get_power_info(_, _))
        .WillOnce(DoAll(SetArgPointee<1>(mock_power), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*driver, get_gpu_memory_usage(_, _, _))
        .WillOnce(DoAll(SetArgPointee<2>(2ULL * 1024 * 1024 * 1024),
                        Return(AMDSMI_STATUS_SUCCESS)));

    settings s{ .busy          = true,
                .temp          = true,
                .power         = true,
                .mem_usage     = true,
                .vcn_activity  = false,
                .jpeg_activity = false,
                .xgmi          = false,
                .pcie          = false };

    auto result = sampler->sample(0, s);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->m_busy_perc.gfx_activity, 75u);
    EXPECT_EQ(result->m_busy_perc.umc_activity, 50u);
    EXPECT_EQ(result->m_busy_perc.mm_activity, 25u);
    EXPECT_EQ(result->m_temp, 60);
    EXPECT_EQ(result->m_power.current_socket_power, 150u);
    EXPECT_EQ(result->m_mem_usage, 2ULL * 1024 * 1024 * 1024);
}

TEST_F(SamplerTest, SkipsDisabledMetrics)
{
    // Only temp is enabled, so only get_temp_metric should be called
    EXPECT_CALL(*driver, get_gpu_activity(_, _)).Times(0);
    EXPECT_CALL(*driver, get_temp_metric(_, _, _, _))
        .WillOnce(DoAll(SetArgPointee<3>(50), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*driver, get_power_info(_, _)).Times(0);
    EXPECT_CALL(*driver, get_gpu_memory_usage(_, _, _)).Times(0);

    settings s{ .busy          = false,
                .temp          = true,
                .power         = false,
                .mem_usage     = false,
                .vcn_activity  = false,
                .jpeg_activity = false,
                .xgmi          = false,
                .pcie          = false };

    sampler->sample(0, s);
}

TEST_F(SamplerTest, StoresSampleWithCorrectTimestamp)
{
    EXPECT_CALL(*clock, now_ns()).WillOnce(Return(123456789));

    settings s{ .busy = true };
    sampler->sample(0, s);

    ASSERT_TRUE(storage->has_samples());
    EXPECT_EQ(storage->last_sample().timestamp, 123456789u);
}

TEST_F(SamplerTest, StoresSampleWithCorrectDeviceId)
{
    settings s{ .busy = true };
    sampler->sample(42, s);

    ASSERT_TRUE(storage->has_samples());
    EXPECT_EQ(storage->last_sample().device_id, 42u);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(SamplerTest, HandlesNotSupportedGracefully)
{
    // Activity returns NOT_SUPPORTED, temp should still be called
    EXPECT_CALL(*driver, get_gpu_activity(_, _))
        .WillOnce(Return(AMDSMI_STATUS_NOT_SUPPORTED));
    EXPECT_CALL(*driver, get_temp_metric(_, _, _, _))
        .WillOnce(DoAll(SetArgPointee<3>(60), Return(AMDSMI_STATUS_SUCCESS)));

    settings s{ .busy = true, .temp = true };
    auto     result = sampler->sample(0, s);

    // Should still succeed - NOT_SUPPORTED is handled gracefully
    ASSERT_TRUE(result.has_value());
}

TEST_F(SamplerTest, DisablesOnlyFailingDeviceOnHardError)
{
    EXPECT_CALL(*driver, get_gpu_activity(_, _)).WillOnce(Return(AMDSMI_STATUS_INVAL));

    settings s{ .busy = true };
    auto     result = sampler->sample(0, s);

    // Device 0 should fail and be disabled
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(sampler->is_device_disabled(0));
}

TEST_F(SamplerTest, SkipsDisabledDevice)
{
    // Manually disable device 1
    sampler->disable_device(1);

    // No driver calls should be made for disabled device
    EXPECT_CALL(*driver, get_gpu_activity(_, _)).Times(0);
    EXPECT_CALL(*driver, get_handle(_)).Times(0);

    settings s{ .busy = true };
    auto     result = sampler->sample(1, s);

    EXPECT_FALSE(result.has_value());
}

TEST_F(SamplerTest, OtherDevicesContinueAfterOneDisabled)
{
    // Device 0 fails
    EXPECT_CALL(*driver, get_gpu_activity(_, _))
        .WillOnce(Return(AMDSMI_STATUS_INVAL))     // First call fails
        .WillOnce(Return(AMDSMI_STATUS_SUCCESS));  // Second call succeeds

    settings s{ .busy = true };

    auto result0 = sampler->sample(0, s);
    EXPECT_FALSE(result0.has_value());
    EXPECT_TRUE(sampler->is_device_disabled(0));

    // Device 1 should still work
    auto result1 = sampler->sample(1, s);
    EXPECT_TRUE(result1.has_value());
    EXPECT_FALSE(sampler->is_device_disabled(1));
}

// ============================================================================
// Advanced Metrics Tests (VCN, JPEG, XGMI, PCIe)
// ============================================================================

TEST_F(SamplerTest, FetchesGpuMetricsWhenAdvancedMetricsEnabled)
{
    EXPECT_CALL(*driver, get_gpu_metrics_info(_, _))
        .WillOnce(Return(AMDSMI_STATUS_SUCCESS));

    settings s{ .busy          = false,
                .temp          = false,
                .power         = false,
                .mem_usage     = false,
                .vcn_activity  = true,
                .jpeg_activity = false,
                .xgmi          = false,
                .pcie          = false };

    sampler->sample(0, s);
}

TEST_F(SamplerTest, SkipsGpuMetricsWhenOnlyBasicEnabled)
{
    EXPECT_CALL(*driver, get_gpu_metrics_info(_, _)).Times(0);

    settings s{ .busy          = true,
                .temp          = true,
                .power         = true,
                .mem_usage     = true,
                .vcn_activity  = false,
                .jpeg_activity = false,
                .xgmi          = false,
                .pcie          = false };

    sampler->sample(0, s);
}

TEST_F(SamplerTest, ProcessesPcieMetrics)
{
    amdsmi_gpu_metrics_t mock_metrics{};
    mock_metrics.pcie_link_width     = 16;
    mock_metrics.pcie_link_speed     = 32;
    mock_metrics.pcie_bandwidth_acc  = 1000;
    mock_metrics.pcie_bandwidth_inst = 500;

    EXPECT_CALL(*driver, get_gpu_metrics_info(_, _))
        .WillOnce(DoAll(SetArgPointee<1>(mock_metrics), Return(AMDSMI_STATUS_SUCCESS)));

    settings s{ .pcie = true };
    auto     result = sampler->sample(0, s);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(storage->has_samples());
}

// ============================================================================
// Verification with StrictMock
// ============================================================================

TEST(SamplerStrictTest, VerifiesExactCallSequence)
{
    auto driver  = std::make_shared<StrictMock<MockDriver>>();
    auto clock   = std::make_shared<StrictMock<MockClock>>();
    auto storage = std::make_shared<NiceMock<MockStorage>>();  // NiceMock for storage
    auto state   = std::make_shared<StrictMock<MockState>>();
    auto gpu_caps =
        std::make_shared<NiceMock<MockGpuCapabilities>>();  // NiceMock for gpu_caps

    storage->SetUpCapture();
    gpu_caps->SetUpDefaults();

    // Set up expected call sequence - only busy enabled, all others disabled
    EXPECT_CALL(*state, is_child_process()).WillOnce(Return(false));
    EXPECT_CALL(*state, get_state()).WillOnce(Return(State::Active));
    EXPECT_CALL(*clock, now_ns()).WillOnce(Return(1000000));
    EXPECT_CALL(*driver, get_handle(0)).WillOnce(Return(nullptr));
    EXPECT_CALL(*driver, get_gpu_activity(_, _)).WillOnce(Return(AMDSMI_STATUS_SUCCESS));

    TestableSampler sampler(driver, clock, storage, state, gpu_caps);

    // Explicitly set all settings to ensure only busy is enabled
    settings s{ .busy          = true,
                .temp          = false,
                .power         = false,
                .mem_usage     = false,
                .vcn_activity  = false,
                .jpeg_activity = false,
                .xgmi          = false,
                .pcie          = false };

    sampler.sample(0, s);
}

// ============================================================================
// Pure Function Tests (metrics namespace)
// ============================================================================

TEST(MetricsTest, FilterUnsupportedValueReturnsZeroForMaxUint16)
{
    // Must cast to uint16_t, otherwise UINT16_MAX is deduced as int
    EXPECT_EQ(metrics::filter_unsupported_value(static_cast<uint16_t>(UINT16_MAX)), 0);
}

TEST(MetricsTest, FilterUnsupportedValueReturnsZeroForMaxUint64)
{
    EXPECT_EQ(metrics::filter_unsupported_value(UINT64_MAX), 0ULL);
}

TEST(MetricsTest, FilterUnsupportedValuePreservesValidValues)
{
    EXPECT_EQ(metrics::filter_unsupported_value(uint16_t{ 100 }), 100);
    EXPECT_EQ(metrics::filter_unsupported_value(uint64_t{ 12345 }), 12345ULL);
    EXPECT_EQ(metrics::filter_unsupported_value(uint32_t{ 0 }), 0u);
}

TEST(MetricsTest, CopyValidMetricsFiltersMaxValues)
{
    std::array<uint16_t, 4> src = { 10, UINT16_MAX, 20, UINT16_MAX };
    std::vector<uint16_t>   dest;

    metrics::copy_valid_metrics(dest, src, UINT16_MAX);

    ASSERT_EQ(dest.size(), 2u);
    EXPECT_EQ(dest[0], 10);
    EXPECT_EQ(dest[1], 20);
}

TEST(MetricsTest, CopyValidMetricsHandlesEmptySource)
{
    std::array<uint16_t, 0> src;
    std::vector<uint16_t>   dest;

    metrics::copy_valid_metrics(dest, src, UINT16_MAX);

    EXPECT_TRUE(dest.empty());
}

TEST(MetricsTest, CopyValidMetricsHandlesAllInvalid)
{
    std::array<uint64_t, 3> src = { UINT64_MAX, UINT64_MAX, UINT64_MAX };
    std::vector<uint64_t>   dest;

    metrics::copy_valid_metrics(dest, src, UINT64_MAX);

    EXPECT_TRUE(dest.empty());
}

TEST(MetricsTest, SerializeSettingsEncodesAllFields)
{
    settings s{ .busy          = true,
                .temp          = false,
                .power         = true,
                .mem_usage     = false,
                .vcn_activity  = true,
                .jpeg_activity = false,
                .xgmi          = true,
                .pcie          = false };

    auto serialized = metrics::serialize_settings(s);

    EXPECT_NE(serialized, 0u);
}

TEST(MetricsTest, SerializeSettingsAllFalseReturnsZero)
{
    settings s{ .busy          = false,
                .temp          = false,
                .power         = false,
                .mem_usage     = false,
                .vcn_activity  = false,
                .jpeg_activity = false,
                .xgmi          = false,
                .pcie          = false };

    auto serialized = metrics::serialize_settings(s);

    EXPECT_EQ(serialized, 0u);
}

TEST(MetricsTest, ProcessXgmiMetricsDetectsValidData)
{
    amdsmi_gpu_metrics_t raw{};
    raw.xgmi_link_width = 16;
    raw.xgmi_link_speed = 32;

    auto result = metrics::process_xgmi_metrics(raw);

    EXPECT_TRUE(result.has_data);
    EXPECT_EQ(result.metrics.xgmi_link_width, 16);
    EXPECT_EQ(result.metrics.xgmi_link_speed, 32);
}

TEST(MetricsTest, ProcessXgmiMetricsFiltersMaxValues)
{
    amdsmi_gpu_metrics_t raw{};
    raw.xgmi_link_width = UINT16_MAX;
    raw.xgmi_link_speed = UINT16_MAX;
    // Also set the data arrays to max values so they get filtered
    for(auto& v : raw.xgmi_read_data_acc)
        v = UINT64_MAX;
    for(auto& v : raw.xgmi_write_data_acc)
        v = UINT64_MAX;

    auto result = metrics::process_xgmi_metrics(raw);

    EXPECT_FALSE(result.has_data);
    EXPECT_EQ(result.metrics.xgmi_link_width, 0);
    EXPECT_EQ(result.metrics.xgmi_link_speed, 0);
}

TEST(MetricsTest, ProcessPcieMetricsDetectsValidData)
{
    amdsmi_gpu_metrics_t raw{};
    raw.pcie_link_width     = 16;
    raw.pcie_link_speed     = 5;
    raw.pcie_bandwidth_acc  = 1000;
    raw.pcie_bandwidth_inst = 100;

    auto result = metrics::process_pcie_metrics(raw);

    EXPECT_TRUE(result.has_data);
    EXPECT_EQ(result.metrics.pcie_link_width, 16);
    EXPECT_EQ(result.metrics.pcie_link_speed, 5);
}

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace testing
}  // namespace amd_smi
}  // namespace rocprofsys
