// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/sdk_pmc/device.hpp"
#include "library/pmc/device_providers/rocprofiler_sdk/drivers/tests/mock_driver.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

using namespace rocprofsys::pmc::collectors::sdk_pmc;
using ::testing::_;
using ::testing::AnyNumber;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

using MockDriver = ::testing::StrictMock<
    rocprofsys::pmc::drivers::rocprofiler_sdk::testing::mock_driver>;

namespace rocprofsys::pmc::collectors::sdk_pmc::testing
{

class SdkPmcDeviceTest : public ::testing::Test
{
protected:
    std::shared_ptr<MockDriver>     mock_driver;
    rocprofiler_context_id_t        test_context;
    rocprofiler_agent_id_t          test_agent_id;
    rocprofiler_counter_config_id_t test_profile_config;
    size_t                          test_index = 0;

    void SetUp() override
    {
        mock_driver                = std::make_shared<MockDriver>();
        test_context.handle        = 1;
        test_agent_id.handle       = 42;
        test_profile_config.handle = 100;
        test_index                 = 0;
    }
};

TEST_F(SdkPmcDeviceTest, DeviceProperties)
{
    device<MockDriver> dev(mock_driver, test_context, test_agent_id, test_profile_config,
                           test_index, {});

    EXPECT_EQ(dev.get_index(), 0u);
    EXPECT_EQ(dev.get_name(), "GPU0");
    EXPECT_EQ(dev.get_vendor_name(), "AMD");
    EXPECT_TRUE(dev.is_supported());
    EXPECT_NE(dev.get_supported_metrics().value, 0u);
    EXPECT_EQ(dev.get_agent_id().handle, 42u);
    EXPECT_EQ(dev.get_profile_config().handle, 100u);
}

TEST_F(SdkPmcDeviceTest, DeviceWithIndex3)
{
    device<MockDriver> dev(mock_driver, test_context, test_agent_id, test_profile_config,
                           3, {});

    EXPECT_EQ(dev.get_index(), 3u);
    EXPECT_EQ(dev.get_name(), "GPU3");
    EXPECT_EQ(dev.get_product_name(), "GPU 3");
}

TEST_F(SdkPmcDeviceTest, SampleReturnsCounters)
{
    device<MockDriver> dev(mock_driver, test_context, test_agent_id, test_profile_config,
                           test_index, {});

    // Setup: sample returns 2 counter records
    rocprofiler_counter_record_t records[2];
    records[0].id            = { 10 };
    records[0].counter_value = 42.0;
    records[1].id            = { 20 };
    records[1].counter_value = 100.0;

    rocprofiler_counter_id_t counter_id_0 = { 1 };
    rocprofiler_counter_id_t counter_id_1 = { 2 };

    EXPECT_CALL(*mock_driver, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([&](rocprofiler_context_id_t, rocprofiler_user_data_t,
                      rocprofiler_counter_flag_t, rocprofiler_counter_record_t* out,
                      size_t* count) {
            out[0] = records[0];
            out[1] = records[1];
            *count = 2;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // Mock query_record_counter_id for each record
    EXPECT_CALL(*mock_driver, query_record_counter_id(_, _))
        .WillOnce(
            DoAll(SetArgPointee<1>(counter_id_0), Return(ROCPROFILER_STATUS_SUCCESS)))
        .WillOnce(
            DoAll(SetArgPointee<1>(counter_id_1), Return(ROCPROFILER_STATUS_SUCCESS)));

    // Mock query_counter_info for name lookup
    static const char* name_0 = "SQ_WAVES";
    static const char* name_1 = "SQ_INSTS_VALU";

    rocprofiler_counter_info_v0_t info_0{};
    info_0.name = name_0;
    rocprofiler_counter_info_v0_t info_1{};
    info_1.name = name_1;

    EXPECT_CALL(*mock_driver, query_counter_info(_, _, _))
        .WillOnce([&](rocprofiler_counter_id_t, rocprofiler_counter_info_version_id_t,
                      void* out) {
            *static_cast<rocprofiler_counter_info_v0_t*>(out) = info_0;
            return ROCPROFILER_STATUS_SUCCESS;
        })
        .WillOnce([&](rocprofiler_counter_id_t, rocprofiler_counter_info_version_id_t,
                      void* out) {
            *static_cast<rocprofiler_counter_info_v0_t*>(out) = info_1;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    enabled_metrics enabled;
    enabled.value = 1;

    auto result = dev.get_sdk_pmc_metrics(enabled, 1000000);

    ASSERT_EQ(result.counters.size(), 2u);
    EXPECT_EQ(result.counters[0].name, "SQ_WAVES");
    EXPECT_DOUBLE_EQ(result.counters[0].value, 42.0);
    EXPECT_EQ(result.counters[1].name, "SQ_INSTS_VALU");
    EXPECT_DOUBLE_EQ(result.counters[1].value, 100.0);
}

TEST_F(SdkPmcDeviceTest, SampleFailureReturnsEmpty)
{
    device<MockDriver> dev(mock_driver, test_context, test_agent_id, test_profile_config,
                           test_index, {});

    EXPECT_CALL(*mock_driver, sample_device_counting_service(_, _, _, _, _))
        .WillOnce(Return(ROCPROFILER_STATUS_ERROR));

    enabled_metrics enabled;
    enabled.value = 1;

    auto result = dev.get_sdk_pmc_metrics(enabled, 1000000);

    EXPECT_TRUE(result.counters.empty());
}

TEST_F(SdkPmcDeviceTest, SampleWithZeroRecords)
{
    device<MockDriver> dev(mock_driver, test_context, test_agent_id, test_profile_config,
                           test_index, {});

    EXPECT_CALL(*mock_driver, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([](rocprofiler_context_id_t, rocprofiler_user_data_t,
                     rocprofiler_counter_flag_t, rocprofiler_counter_record_t*,
                     size_t* count) {
            *count = 0;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    enabled_metrics enabled;
    enabled.value = 1;

    auto result = dev.get_sdk_pmc_metrics(enabled, 1000000);

    EXPECT_TRUE(result.counters.empty());
}

TEST_F(SdkPmcDeviceTest, SampleSkipsFailedCounterIdQuery)
{
    device<MockDriver> dev(mock_driver, test_context, test_agent_id, test_profile_config,
                           test_index, {});

    rocprofiler_counter_record_t records[1];
    records[0].id            = { 10 };
    records[0].counter_value = 42.0;

    EXPECT_CALL(*mock_driver, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([&](rocprofiler_context_id_t, rocprofiler_user_data_t,
                      rocprofiler_counter_flag_t, rocprofiler_counter_record_t* out,
                      size_t* count) {
            out[0] = records[0];
            *count = 1;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // query_record_counter_id fails
    EXPECT_CALL(*mock_driver, query_record_counter_id(_, _))
        .WillOnce(Return(ROCPROFILER_STATUS_ERROR));

    enabled_metrics enabled;
    enabled.value = 1;

    auto result = dev.get_sdk_pmc_metrics(enabled, 1000000);

    EXPECT_TRUE(result.counters.empty());
}

}  // namespace rocprofsys::pmc::collectors::sdk_pmc::testing
