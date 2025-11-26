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
#include "library/amd_smi/sampler_impl.hpp"
#include "library/amd_smi/test/mock_policies.hpp"

#include <gtest/gtest.h>

namespace rocprofsys
{
namespace amd_smi
{
namespace testing
{
#if ROCPROFSYS_USE_ROCM > 0

// Type alias for mock sampler
using MockSampler = SamplerImpl<MockConfig>;

class SamplerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        reset_all_mocks();
        mock_state = State::Active;
        sampler    = std::make_unique<MockSampler>(mock_state);
    }

    void TearDown() override { reset_all_mocks(); }

    std::atomic<State>          mock_state{ State::Active };
    std::unique_ptr<MockSampler> sampler;
};

// ============================================================================
// State Management Tests
// ============================================================================

TEST_F(SamplerTest, ReturnsNulloptWhenStateNotActive)
{
    mock_state = State::PreInit;
    settings s{};

    auto result = sampler->sample(0, s);

    EXPECT_FALSE(result.has_value());
    EXPECT_FALSE(MockStoragePolicy::has_samples());
}

TEST_F(SamplerTest, ReturnsNulloptInChildProcess)
{
    MockStatePolicy::mock_is_child = true;
    settings s{};

    auto result = sampler->sample(0, s);

    EXPECT_FALSE(result.has_value());
    EXPECT_FALSE(MockStoragePolicy::has_samples());
}

TEST_F(SamplerTest, ReturnsNulloptWhenDisabled)
{
    mock_state = State::Disabled;
    settings s{};

    auto result = sampler->sample(0, s);

    EXPECT_FALSE(result.has_value());
}

TEST_F(SamplerTest, ReturnsNulloptWhenFinalized)
{
    mock_state = State::Finalized;
    settings s{};

    auto result = sampler->sample(0, s);

    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Basic Metrics Collection Tests
// ============================================================================

TEST_F(SamplerTest, CollectsBasicMetricsWhenEnabled)
{
    MockDriverPolicy::mock_usage.gfx_activity         = 75;
    MockDriverPolicy::mock_usage.umc_activity         = 50;
    MockDriverPolicy::mock_usage.mm_activity          = 25;
    MockDriverPolicy::mock_temp                       = 60;
    MockDriverPolicy::mock_power.current_socket_power = 150;
    MockDriverPolicy::mock_mem_usage                  = 2ULL * 1024 * 1024 * 1024;

    settings s{ .busy         = true,
                .temp         = true,
                .power        = true,
                .mem_usage    = true,
                .vcn_activity = false,
                .jpeg_activity = false,
                .xgmi         = false,
                .pcie         = false };

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
    settings s{ .busy         = false,
                .temp         = true,
                .power        = false,
                .mem_usage    = false,
                .vcn_activity = false,
                .jpeg_activity = false,
                .xgmi         = false,
                .pcie         = false };

    sampler->sample(0, s);

    EXPECT_EQ(MockDriverPolicy::get_gpu_activity_calls, 0);
    EXPECT_EQ(MockDriverPolicy::get_temp_metric_calls, 1);
    EXPECT_EQ(MockDriverPolicy::get_power_info_calls, 0);
    EXPECT_EQ(MockDriverPolicy::get_gpu_memory_usage_calls, 0);
}

TEST_F(SamplerTest, StoresSampleWithCorrectTimestamp)
{
    MockClockPolicy::set_time(123456789);
    settings s{ .busy = true };

    sampler->sample(0, s);

    ASSERT_TRUE(MockStoragePolicy::has_samples());
    EXPECT_EQ(MockStoragePolicy::last_sample().timestamp, 123456789u);
}

TEST_F(SamplerTest, StoresSampleWithCorrectDeviceId)
{
    settings s{ .busy = true };

    sampler->sample(42, s);

    ASSERT_TRUE(MockStoragePolicy::has_samples());
    EXPECT_EQ(MockStoragePolicy::last_sample().device_id, 42u);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(SamplerTest, HandlesNotSupportedGracefully)
{
    MockDriverPolicy::activity_status = AMDSMI_STATUS_NOT_SUPPORTED;
    settings s{ .busy = true, .temp = true };

    auto result = sampler->sample(0, s);

    // Should still succeed - NOT_SUPPORTED is handled gracefully
    ASSERT_TRUE(result.has_value());
    // Temperature should still be collected
    EXPECT_EQ(MockDriverPolicy::get_temp_metric_calls, 1);
}

TEST_F(SamplerTest, DisablesOnlyFailingDeviceOnHardError)
{
    MockDriverPolicy::activity_status = AMDSMI_STATUS_INVAL;
    settings s{ .busy = true };

    auto result = sampler->sample(0, s);

    // Device 0 should fail and be disabled
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(sampler->is_device_disabled(0));

    // Global state should NOT be disabled - other GPUs can still work
    EXPECT_EQ(mock_state.load(), State::Active);
}

TEST_F(SamplerTest, SkipsDisabledDevice)
{
    // Manually disable device 1
    sampler->disable_device(1);

    settings s{ .busy = true };

    // Device 1 should be skipped
    auto result = sampler->sample(1, s);
    EXPECT_FALSE(result.has_value());

    // No driver calls should be made for disabled device
    EXPECT_EQ(MockDriverPolicy::get_gpu_activity_calls, 0);
}

TEST_F(SamplerTest, OtherDevicesContinueAfterOneDisabled)
{
    MockDriverPolicy::activity_status = AMDSMI_STATUS_INVAL;
    settings s{ .busy = true };

    // Device 0 fails
    auto result0 = sampler->sample(0, s);
    EXPECT_FALSE(result0.has_value());
    EXPECT_TRUE(sampler->is_device_disabled(0));

    // Reset driver to success for other devices
    MockDriverPolicy::activity_status = AMDSMI_STATUS_SUCCESS;
    MockDriverPolicy::reset_call_counts();

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
    settings s{ .busy         = false,
                .temp         = false,
                .power        = false,
                .mem_usage    = false,
                .vcn_activity = true,
                .jpeg_activity = false,
                .xgmi         = false,
                .pcie         = false };

    sampler->sample(0, s);

    EXPECT_EQ(MockDriverPolicy::get_gpu_metrics_info_calls, 1);
}

TEST_F(SamplerTest, SkipsGpuMetricsWhenOnlyBasicEnabled)
{
    settings s{ .busy         = true,
                .temp         = true,
                .power        = true,
                .mem_usage    = true,
                .vcn_activity = false,
                .jpeg_activity = false,
                .xgmi         = false,
                .pcie         = false };

    sampler->sample(0, s);

    EXPECT_EQ(MockDriverPolicy::get_gpu_metrics_info_calls, 0);
}

TEST_F(SamplerTest, ProcessesVcnMetricsDeviceLevel)
{
    MockGpuCapabilitiesPolicy::mock_vcn_device_level           = true;
    MockDriverPolicy::mock_gpu_metrics.vcn_activity[0] = 50;
    MockDriverPolicy::mock_gpu_metrics.vcn_activity[1] = 60;

    settings s{ .vcn_activity = true };

    auto result = sampler->sample(0, s);

    ASSERT_TRUE(result.has_value());
}

TEST_F(SamplerTest, ProcessesPcieMetrics)
{
    MockDriverPolicy::mock_gpu_metrics.pcie_link_width    = 16;
    MockDriverPolicy::mock_gpu_metrics.pcie_link_speed    = 32;
    MockDriverPolicy::mock_gpu_metrics.pcie_bandwidth_acc  = 1000;
    MockDriverPolicy::mock_gpu_metrics.pcie_bandwidth_inst = 500;

    settings s{ .pcie = true };

    auto result = sampler->sample(0, s);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(MockStoragePolicy::has_samples());
}

// ============================================================================
// Pure Function Tests (metrics namespace)
// ============================================================================

TEST(MetricsTest, FilterUnsupportedValueReturnsZeroForMaxUint16)
{
    EXPECT_EQ(metrics::filter_unsupported_value(UINT16_MAX), 0);
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
    settings s{ .busy         = true,
                .temp         = false,
                .power        = true,
                .mem_usage    = false,
                .vcn_activity = true,
                .jpeg_activity = false,
                .xgmi         = true,
                .pcie         = false };

    auto serialized = metrics::serialize_settings(s);

    EXPECT_NE(serialized, 0u);
}

TEST(MetricsTest, SerializeSettingsAllFalseReturnsZero)
{
    settings s{ .busy         = false,
                .temp         = false,
                .power        = false,
                .mem_usage    = false,
                .vcn_activity = false,
                .jpeg_activity = false,
                .xgmi         = false,
                .pcie         = false };

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

    auto result = metrics::process_xgmi_metrics(raw);

    EXPECT_FALSE(result.has_data);
    EXPECT_EQ(result.metrics.xgmi_link_width, 0);
    EXPECT_EQ(result.metrics.xgmi_link_speed, 0);
}

TEST(MetricsTest, ProcessPcieMetricsDetectsValidData)
{
    amdsmi_gpu_metrics_t raw{};
    raw.pcie_link_width    = 16;
    raw.pcie_link_speed    = 5;
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
