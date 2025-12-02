// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// MIT License - See LICENSE file for details.

#include "library/amd_smi/processor.hpp"
#include "mock_driver.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#if ROCPROFSYS_USE_ROCM > 0

namespace rocprofsys
{
namespace amd_smi
{
namespace testing
{

class ProcessorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_mock_driver = std::make_shared<::testing::NiceMock<mock_driver>>();
        m_mock_driver->set_up_defaults();
    }

    void TearDown() override { m_mock_driver.reset(); }

    std::shared_ptr<::testing::NiceMock<mock_driver>> m_mock_driver;
};

TEST_F(ProcessorTest, ConstructorInitializesFields)
{
    amdsmi_processor_handle handle = reinterpret_cast<amdsmi_processor_handle>(0x1234);
    processor_type_t        type   = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
    size_t                  index  = 0;

    processor<mock_driver> proc(m_mock_driver, handle, type, index);

    EXPECT_EQ(proc.get_handle(), handle);
    EXPECT_EQ(proc.get_processor_type(), type);
    EXPECT_EQ(proc.get_index(), index);
    EXPECT_TRUE(proc.is_enabled());
    EXPECT_FALSE(proc.is_disabled_due_to_error());
}

TEST_F(ProcessorTest, SetEnabledChangesState)
{
    amdsmi_processor_handle handle = reinterpret_cast<amdsmi_processor_handle>(0x1234);
    processor<mock_driver>  proc(m_mock_driver, handle, AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);

    EXPECT_TRUE(proc.is_enabled());

    proc.set_enabled(false);
    EXPECT_FALSE(proc.is_enabled());

    proc.set_enabled(true);
    EXPECT_TRUE(proc.is_enabled());
}

TEST_F(ProcessorTest, GetSmiMetricsReturnsValidMetrics)
{
    using ::testing::_;
    using ::testing::DoAll;
    using ::testing::Return;
    using ::testing::SetArgPointee;

    amdsmi_processor_handle handle = reinterpret_cast<amdsmi_processor_handle>(0x1234);

    amdsmi_engine_usage_t expected_activity{};
    expected_activity.gfx_activity = 50;
    expected_activity.umc_activity = 30;
    expected_activity.mm_activity  = 20;

    EXPECT_CALL(*m_mock_driver, get_activity(handle, _))
        .WillOnce(
            DoAll(SetArgPointee<1>(expected_activity), Return(AMDSMI_STATUS_SUCCESS)));

    processor<mock_driver> proc(m_mock_driver, handle, AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);

    auto metrics = proc.get_smi_metrics();

    EXPECT_EQ(metrics.gfx_activity, 50u);
    EXPECT_EQ(metrics.umc_activity, 30u);
    EXPECT_EQ(metrics.mm_activity, 20u);
}

TEST_F(ProcessorTest, GetSmiMetricsHandlesNotSupported)
{
    using ::testing::_;
    using ::testing::Return;

    amdsmi_processor_handle handle = reinterpret_cast<amdsmi_processor_handle>(0x1234);

    EXPECT_CALL(*m_mock_driver, get_activity(handle, _))
        .WillRepeatedly(Return(AMDSMI_STATUS_NOT_SUPPORTED));
    EXPECT_CALL(*m_mock_driver, get_power_info(handle, _))
        .WillRepeatedly(Return(AMDSMI_STATUS_NOT_SUPPORTED));
    EXPECT_CALL(*m_mock_driver, get_temperature_metric(handle, _, _, _))
        .WillRepeatedly(Return(AMDSMI_STATUS_NOT_SUPPORTED));
    EXPECT_CALL(*m_mock_driver, get_memory_usage(handle, _, _))
        .WillRepeatedly(Return(AMDSMI_STATUS_NOT_SUPPORTED));
    EXPECT_CALL(*m_mock_driver, get_metrics_info(handle, _))
        .WillRepeatedly(Return(AMDSMI_STATUS_NOT_SUPPORTED));

    processor<mock_driver> proc(m_mock_driver, handle, AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);

    auto metrics = proc.get_smi_metrics();
    EXPECT_EQ(metrics.gfx_activity, 0u);
    EXPECT_EQ(metrics.memory_usage, 0u);
}

TEST_F(ProcessorTest, MultipleProcessorsHaveDifferentIndices)
{
    amdsmi_processor_handle handle1 = reinterpret_cast<amdsmi_processor_handle>(0x1);
    amdsmi_processor_handle handle2 = reinterpret_cast<amdsmi_processor_handle>(0x2);
    amdsmi_processor_handle handle3 = reinterpret_cast<amdsmi_processor_handle>(0x3);

    processor<mock_driver> proc1(m_mock_driver, handle1, AMDSMI_PROCESSOR_TYPE_AMD_GPU,
                                 0);
    processor<mock_driver> proc2(m_mock_driver, handle2, AMDSMI_PROCESSOR_TYPE_AMD_GPU,
                                 1);
    processor<mock_driver> proc3(m_mock_driver, handle3, AMDSMI_PROCESSOR_TYPE_AMD_GPU,
                                 2);

    EXPECT_EQ(proc1.get_index(), 0u);
    EXPECT_EQ(proc2.get_index(), 1u);
    EXPECT_EQ(proc3.get_index(), 2u);
}

}  // namespace testing
}  // namespace amd_smi
}  // namespace rocprofsys

#endif  // ROCPROFSYS_USE_ROCM > 0
