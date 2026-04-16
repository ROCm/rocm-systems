// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/sdk_pmc/device.hpp"
#include "library/pmc/device_providers/rocprofiler_sdk/drivers/tests/mock_driver.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

using namespace rocprofsys::pmc::collectors::sdk_pmc;
using ::testing::_;
using ::testing::Return;
using ::testing::StrictMock;

using MockDriver = ::testing::StrictMock<
    rocprofsys::pmc::drivers::rocprofiler_sdk::testing::mock_driver>;
using instance_info_t =
    rocprofsys::pmc::device_providers::rocprofiler_sdk::counter_instance_info;

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

    EXPECT_EQ(dev.get_index(), 0U);
    EXPECT_EQ(dev.get_name(), "GPU 0");
    EXPECT_EQ(dev.get_vendor_name(), "AMD");
    EXPECT_TRUE(dev.is_supported());
    EXPECT_NE(dev.get_supported_metrics().value, 0U);
    EXPECT_EQ(dev.get_agent_id().handle, 42U);
    EXPECT_EQ(dev.get_profile_config().handle, 100U);
}

TEST_F(SdkPmcDeviceTest, DeviceWithIndex3)
{
    device<MockDriver> dev(mock_driver, test_context, test_agent_id, test_profile_config,
                           3, {});

    EXPECT_EQ(dev.get_index(), 3U);
    EXPECT_EQ(dev.get_name(), "GPU 3");
    EXPECT_EQ(dev.get_product_name(), "GPU 3");
}

TEST_F(SdkPmcDeviceTest, SampleWithScalarCounters)
{
    // Two scalar counters (single instance each)
    auto instances = device<MockDriver>::instance_info_vec{
        instance_info_t{ 10, "SQ_WAVES" },
        instance_info_t{ 20, "SQ_INSTS_VALU" },
    };

    device<MockDriver> dev(mock_driver, test_context, test_agent_id, test_profile_config,
                           test_index, {}, std::move(instances));

    rocprofiler_counter_record_t records[2];
    records[0].id            = 10;
    records[0].counter_value = 42.0;
    records[1].id            = 20;
    records[1].counter_value = 100.0;

    EXPECT_CALL(*mock_driver, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([&](rocprofiler_context_id_t, rocprofiler_user_data_t,
                      rocprofiler_counter_flag_t, rocprofiler_counter_record_t* out,
                      size_t* count) {
            out[0] = records[0];
            out[1] = records[1];
            *count = 2;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    enabled_metrics enabled;
    enabled.value = 1;

    auto result = dev.get_sdk_pmc_metrics(enabled, 1000000);

    ASSERT_EQ(result.counters.size(), 2U);
    EXPECT_EQ(result.counters[0].name, "SQ_WAVES");
    EXPECT_DOUBLE_EQ(result.counters[0].value, 42.0);
    EXPECT_EQ(result.counters[1].name, "SQ_INSTS_VALU");
    EXPECT_DOUBLE_EQ(result.counters[1].value, 100.0);
}

TEST_F(SdkPmcDeviceTest, SampleWithMultiDimCounters)
{
    // One counter with 4 dimension instances (WGP=0..3)
    auto instances = device<MockDriver>::instance_info_vec{
        instance_info_t{ 100, "SQC_ICACHE_HITS[WGP=0,SA=0,SE=0]" },
        instance_info_t{ 101, "SQC_ICACHE_HITS[WGP=1,SA=0,SE=0]" },
        instance_info_t{ 102, "SQC_ICACHE_HITS[WGP=2,SA=0,SE=0]" },
        instance_info_t{ 103, "SQC_ICACHE_HITS[WGP=3,SA=0,SE=0]" },
    };

    device<MockDriver> dev(mock_driver, test_context, test_agent_id, test_profile_config,
                           test_index, {}, std::move(instances));

    rocprofiler_counter_record_t records[4];
    for(int i = 0; i < 4; ++i)
    {
        records[i].id            = static_cast<uint64_t>(100 + i);
        records[i].counter_value = static_cast<double>(10 * (i + 1));
    }

    EXPECT_CALL(*mock_driver, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([&](rocprofiler_context_id_t, rocprofiler_user_data_t,
                      rocprofiler_counter_flag_t, rocprofiler_counter_record_t* out,
                      size_t* count) {
            for(int i = 0; i < 4; ++i)
                out[i] = records[i];
            *count = 4;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    enabled_metrics enabled;
    enabled.value = 1;

    auto result = dev.get_sdk_pmc_metrics(enabled, 1000000);

    ASSERT_EQ(result.counters.size(), 4U);
    EXPECT_EQ(result.counters[0].name, "SQC_ICACHE_HITS[WGP=0,SA=0,SE=0]");
    EXPECT_DOUBLE_EQ(result.counters[0].value, 10.0);
    EXPECT_EQ(result.counters[1].name, "SQC_ICACHE_HITS[WGP=1,SA=0,SE=0]");
    EXPECT_DOUBLE_EQ(result.counters[1].value, 20.0);
    EXPECT_EQ(result.counters[2].name, "SQC_ICACHE_HITS[WGP=2,SA=0,SE=0]");
    EXPECT_DOUBLE_EQ(result.counters[2].value, 30.0);
    EXPECT_EQ(result.counters[3].name, "SQC_ICACHE_HITS[WGP=3,SA=0,SE=0]");
    EXPECT_DOUBLE_EQ(result.counters[3].value, 40.0);
}

TEST_F(SdkPmcDeviceTest, SampleSkipsUnknownInstanceIds)
{
    // Only instance_id=10 is known
    auto instances = device<MockDriver>::instance_info_vec{
        instance_info_t{ 10, "SQ_WAVES" },
    };

    device<MockDriver> dev(mock_driver, test_context, test_agent_id, test_profile_config,
                           test_index, {}, std::move(instances));

    rocprofiler_counter_record_t records[2];
    records[0].id            = 10;
    records[0].counter_value = 42.0;
    records[1].id            = 999;  // unknown
    records[1].counter_value = 100.0;

    EXPECT_CALL(*mock_driver, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([&](rocprofiler_context_id_t, rocprofiler_user_data_t,
                      rocprofiler_counter_flag_t, rocprofiler_counter_record_t* out,
                      size_t* count) {
            out[0] = records[0];
            out[1] = records[1];
            *count = 2;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    enabled_metrics enabled;
    enabled.value = 1;

    auto result = dev.get_sdk_pmc_metrics(enabled, 1000000);

    ASSERT_EQ(result.counters.size(), 1U);
    EXPECT_EQ(result.counters[0].name, "SQ_WAVES");
    EXPECT_DOUBLE_EQ(result.counters[0].value, 42.0);
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

TEST_F(SdkPmcDeviceTest, GetQualifiedNames)
{
    auto instances = device<MockDriver>::instance_info_vec{
        instance_info_t{ 10, "SQ_WAVES" },
        instance_info_t{ 20, "SQ_INSTS_VALU" },
    };

    device<MockDriver> dev(mock_driver, test_context, test_agent_id, test_profile_config,
                           test_index, {}, std::move(instances));

    auto names = dev.get_qualified_names();
    ASSERT_EQ(names.size(), 2U);
    // unordered_map — order may vary, so check both are present
    EXPECT_TRUE(std::find(names.begin(), names.end(), "SQ_WAVES") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "SQ_INSTS_VALU") != names.end());
}

TEST_F(SdkPmcDeviceTest, QualifiedNameHelpers)
{
    // Test abbreviate_dimension_name
    EXPECT_EQ(abbreviate_dimension_name("DIMENSION_SHADER_ENGINE"), "SE");
    EXPECT_EQ(abbreviate_dimension_name("DIMENSION_SHADER_ARRAY"), "SA");
    EXPECT_EQ(abbreviate_dimension_name("DIMENSION_INSTANCE"), "INST");
    EXPECT_EQ(abbreviate_dimension_name("DIMENSION_WGP"), "WGP");

    // Test make_qualified_name
    EXPECT_EQ(make_qualified_name("SQ_WAVES", {}), "SQ_WAVES");
    EXPECT_EQ(make_qualified_name("SQC_ICACHE_HITS",
                                  { { "WGP", 0 }, { "SA", 1 }, { "SE", 2 } }),
              "SQC_ICACHE_HITS[WGP=0,SA=1,SE=2]");
}

}  // namespace rocprofsys::pmc::collectors::sdk_pmc::testing
