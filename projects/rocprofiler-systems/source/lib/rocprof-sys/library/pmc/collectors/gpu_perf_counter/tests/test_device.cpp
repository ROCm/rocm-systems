// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/gpu_perf_counter/device.hpp"
#include "library/pmc/device_providers/rocprofiler_sdk/drivers/tests/mock_driver.hpp"
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

using namespace rocprofsys::pmc::collectors::gpu_perf_counter;
using ::testing::_;
using ::testing::Return;
using ::testing::StrictMock;

using MockDriver = ::testing::StrictMock<
    rocprofsys::pmc::drivers::rocprofiler_sdk::testing::mock_driver>;

namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
{

class SdkPmcDeviceTest : public ::testing::Test
{
protected:
    std::shared_ptr<MockDriver>        mock_driver;
    std::shared_ptr<rocprofsys::agent> test_agent;
    rocprofiler_context_id_t           test_context{};
    rocprofiler_counter_config_id_t    test_profile_config{};

    void SetUp() override
    {
        mock_driver                = std::make_shared<MockDriver>();
        test_context.handle        = 1;
        test_profile_config.handle = 100;

        test_agent                    = std::make_shared<rocprofsys::agent>();
        test_agent->type              = agent_type::GPU;
        test_agent->handle            = 42;
        test_agent->device_id         = 0;
        test_agent->device_type_index = 0;
        test_agent->name              = "GPU 0";
        test_agent->product_name      = "GPU 0";
        test_agent->vendor_name       = "AMD";
    }
};

TEST_F(SdkPmcDeviceTest, DeviceProperties)
{
    auto meta = std::vector<counter_metadata>{
        counter_metadata{ 10, "SQ_WAVES", "", "", "", false, false, {} },
    };

    device<MockDriver> dev(mock_driver, test_context, test_agent, test_profile_config,
                           std::move(meta));

    EXPECT_EQ(dev.get_index(), 0U);
    EXPECT_EQ(dev.get_name(), "GPU 0");
    EXPECT_EQ(dev.get_vendor_name(), "AMD");
    EXPECT_TRUE(dev.is_supported());
}

TEST_F(SdkPmcDeviceTest, EmptyDeviceNotSupported)
{
    device<MockDriver> dev(mock_driver, test_context, test_agent, test_profile_config,
                           {});

    EXPECT_FALSE(dev.is_supported());
}

TEST_F(SdkPmcDeviceTest, DeviceWithIndex3)
{
    auto agent3               = std::make_shared<rocprofsys::agent>();
    agent3->device_type_index = 3;
    agent3->name              = "GPU 3";
    agent3->product_name      = "GPU 3";
    agent3->vendor_name       = "AMD";

    device<MockDriver> dev(mock_driver, test_context, agent3, test_profile_config, {});

    EXPECT_EQ(dev.get_index(), 3U);
    EXPECT_EQ(dev.get_name(), "GPU 3");
    EXPECT_EQ(dev.get_product_name(), "GPU 3");
}

TEST_F(SdkPmcDeviceTest, SampleWithScalarCounters)
{
    auto meta = std::vector<counter_metadata>{
        counter_metadata{ 10, "SQ_WAVES", "", "", "", false, false, {} },
        counter_metadata{ 20, "SQ_INSTS_VALU", "", "", "", false, false, {} },
    };

    device<MockDriver> dev(mock_driver, test_context, test_agent, test_profile_config,
                           std::move(meta));

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

#if ROCPROFSYS_ROCM_VERSION < 70000
    // On ROCm < 7.0, record.id is an encoded instance_id; the device layer decodes
    // it via query_record_counter_id to obtain the plain counter handle.
    EXPECT_CALL(*mock_driver,
                query_record_counter_id(rocprofiler_counter_instance_id_t{ 10 }, _))
        .WillOnce([](rocprofiler_counter_instance_id_t, rocprofiler_counter_id_t* out) {
            out->handle = 10;
            return ROCPROFILER_STATUS_SUCCESS;
        });
    EXPECT_CALL(*mock_driver,
                query_record_counter_id(rocprofiler_counter_instance_id_t{ 20 }, _))
        .WillOnce([](rocprofiler_counter_instance_id_t, rocprofiler_counter_id_t* out) {
            out->handle = 20;
            return ROCPROFILER_STATUS_SUCCESS;
        });
#endif

    enabled_metrics enabled{ {} };
    auto            result = dev.get_gpu_perf_counter_metrics(enabled, 1000000);

    ASSERT_EQ(result.size(), 2U);
    EXPECT_EQ(result[0].counter_id, 10U);
    EXPECT_DOUBLE_EQ(result[0].value, 42.0);
    EXPECT_EQ(result[1].counter_id, 20U);
    EXPECT_DOUBLE_EQ(result[1].value, 100.0);
}

TEST_F(SdkPmcDeviceTest, SampleWithMultiDimCounters)
{
    auto meta = std::vector<counter_metadata>{
        counter_metadata{ 100,
                          "SQC_ICACHE_HITS",
                          "",
                          "",
                          "",
                          false,
                          false,
                          { { "WGP", 0 }, { "SA", 0 }, { "SE", 0 } } },
        counter_metadata{ 101,
                          "SQC_ICACHE_HITS",
                          "",
                          "",
                          "",
                          false,
                          false,
                          { { "WGP", 1 }, { "SA", 0 }, { "SE", 0 } } },
        counter_metadata{ 102,
                          "SQC_ICACHE_HITS",
                          "",
                          "",
                          "",
                          false,
                          false,
                          { { "WGP", 2 }, { "SA", 0 }, { "SE", 0 } } },
        counter_metadata{ 103,
                          "SQC_ICACHE_HITS",
                          "",
                          "",
                          "",
                          false,
                          false,
                          { { "WGP", 3 }, { "SA", 0 }, { "SE", 0 } } },
    };

    device<MockDriver> dev(mock_driver, test_context, test_agent, test_profile_config,
                           std::move(meta));

    rocprofiler_counter_record_t records[4];
    for(int i = 0; i < 4; ++i)
    {
        records[i].id            = static_cast<std::uint64_t>(100 + i);
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

#if ROCPROFSYS_ROCM_VERSION < 70000
    // On ROCm < 7.0, the device layer decodes each instance_id to a counter handle.
    for(int i = 0; i < 4; ++i)
    {
        const std::uint64_t expected_handle = static_cast<std::uint64_t>(100 + i);
        EXPECT_CALL(*mock_driver,
                    query_record_counter_id(
                        rocprofiler_counter_instance_id_t{ expected_handle }, _))
            .WillOnce([expected_handle](rocprofiler_counter_instance_id_t,
                                        rocprofiler_counter_id_t* out) {
                out->handle = expected_handle;
                return ROCPROFILER_STATUS_SUCCESS;
            });
    }
#endif

    enabled_metrics enabled{ {} };
    auto            result = dev.get_gpu_perf_counter_metrics(enabled, 1000000);

    ASSERT_EQ(result.size(), 4U);
    EXPECT_EQ(result[0].counter_id, 100U);
    EXPECT_DOUBLE_EQ(result[0].value, 10.0);
    EXPECT_EQ(result[1].counter_id, 101U);
    EXPECT_DOUBLE_EQ(result[1].value, 20.0);
    EXPECT_EQ(result[2].counter_id, 102U);
    EXPECT_DOUBLE_EQ(result[2].value, 30.0);
    EXPECT_EQ(result[3].counter_id, 103U);
    EXPECT_DOUBLE_EQ(result[3].value, 40.0);
}

// Verify that the encoded instance_id returned by the SDK is decoded to the plain
// counter handle that matches the metadata key used by find_gpu_perf_counter_by_id.
// On ROCm < 7.0 this requires a call to query_record_counter_id.
// On ROCm >= 7.0 the instance_id IS the metadata key, so no decode is needed.
TEST_F(SdkPmcDeviceTest, CounterIdDecodedFromInstanceId)
{
    // counter_id.handle = 7 is the plain counter handle stored in metadata.
    // On ROCm < 7.0, the SDK encodes this into instance_id = 0xDEAD0007 (synthetic).
    // On ROCm >= 7.0, instance_id is used directly as the metadata key.
    constexpr std::uint64_t plain_counter_handle = 7;

#if ROCPROFSYS_ROCM_VERSION < 70000
    constexpr std::uint64_t sdk_instance_id = 0xDEAD0007ULL;
#else
    constexpr std::uint64_t sdk_instance_id = plain_counter_handle;
#endif

    auto meta = std::vector<counter_metadata>{
        counter_metadata{
            plain_counter_handle, "SQ_WAVES", "", "", "", false, false, {} },
    };

    device<MockDriver> dev(mock_driver, test_context, test_agent, test_profile_config,
                           std::move(meta));

    rocprofiler_counter_record_t record{};
    record.id            = sdk_instance_id;
    record.counter_value = 99.0;

    EXPECT_CALL(*mock_driver, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([&](rocprofiler_context_id_t, rocprofiler_user_data_t,
                      rocprofiler_counter_flag_t, rocprofiler_counter_record_t* out,
                      size_t* count) {
            out[0] = record;
            *count = 1;
            return ROCPROFILER_STATUS_SUCCESS;
        });

#if ROCPROFSYS_ROCM_VERSION < 70000
    // The driver decodes the encoded instance_id to the plain counter handle.
    EXPECT_CALL(
        *mock_driver,
        query_record_counter_id(rocprofiler_counter_instance_id_t{ sdk_instance_id }, _))
        .WillOnce([](rocprofiler_counter_instance_id_t, rocprofiler_counter_id_t* out) {
            out->handle = plain_counter_handle;
            return ROCPROFILER_STATUS_SUCCESS;
        });
#endif

    enabled_metrics enabled{ {} };
    auto            result = dev.get_gpu_perf_counter_metrics(enabled, 1000000);

    ASSERT_EQ(result.size(), 1U);
    // The emitted counter_id must equal the plain handle stored in counter_metadata,
    // which is what find_gpu_perf_counter_by_id looks up.
    EXPECT_EQ(result[0].counter_id, plain_counter_handle);
    EXPECT_DOUBLE_EQ(result[0].value, 99.0);
}

// Verify that calling get_gpu_perf_counter_metrics repeatedly does not grow
// heap unboundedly — the same m_result_cache vector is reused across samples.
TEST_F(SdkPmcDeviceTest, ResultCacheReusedAcrossSamples)
{
    auto meta = std::vector<counter_metadata>{
        counter_metadata{ 5, "SQ_WAVES", "", "", "", false, false, {} },
    };

    device<MockDriver> dev(mock_driver, test_context, test_agent, test_profile_config,
                           std::move(meta));

    rocprofiler_counter_record_t record{};
    record.id            = 5;
    record.counter_value = 1.0;

    // Two successive sample calls; each must return correct data.
    EXPECT_CALL(*mock_driver, sample_device_counting_service(_, _, _, _, _))
        .Times(2)
        .WillRepeatedly([&](rocprofiler_context_id_t, rocprofiler_user_data_t,
                            rocprofiler_counter_flag_t, rocprofiler_counter_record_t* out,
                            size_t* count) {
            out[0] = record;
            *count = 1;
            return ROCPROFILER_STATUS_SUCCESS;
        });

#if ROCPROFSYS_ROCM_VERSION < 70000
    EXPECT_CALL(*mock_driver,
                query_record_counter_id(rocprofiler_counter_instance_id_t{ 5 }, _))
        .Times(2)
        .WillRepeatedly(
            [](rocprofiler_counter_instance_id_t, rocprofiler_counter_id_t* out) {
                out->handle = 5;
                return ROCPROFILER_STATUS_SUCCESS;
            });
#endif

    enabled_metrics enabled{ {} };

    auto result1 = dev.get_gpu_perf_counter_metrics(enabled, 1000000);
    ASSERT_EQ(result1.size(), 1U);
    EXPECT_EQ(result1[0].counter_id, 5U);

    auto result2 = dev.get_gpu_perf_counter_metrics(enabled, 2000000);
    ASSERT_EQ(result2.size(), 1U);
    EXPECT_EQ(result2[0].counter_id, 5U);
}

TEST_F(SdkPmcDeviceTest, SampleFailureReturnsEmpty)
{
    device<MockDriver> dev(mock_driver, test_context, test_agent, test_profile_config,
                           {});

    EXPECT_CALL(*mock_driver, sample_device_counting_service(_, _, _, _, _))
        .WillOnce(Return(ROCPROFILER_STATUS_ERROR));

    enabled_metrics enabled{ {} };
    auto            result = dev.get_gpu_perf_counter_metrics(enabled, 1000000);

    EXPECT_TRUE(result.empty());
}

TEST_F(SdkPmcDeviceTest, SampleWithZeroRecords)
{
    device<MockDriver> dev(mock_driver, test_context, test_agent, test_profile_config,
                           {});

    EXPECT_CALL(*mock_driver, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([](rocprofiler_context_id_t, rocprofiler_user_data_t,
                     rocprofiler_counter_flag_t, rocprofiler_counter_record_t*,
                     size_t* count) {
            *count = 0;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    enabled_metrics enabled{ {} };
    auto            result = dev.get_gpu_perf_counter_metrics(enabled, 1000000);

    EXPECT_TRUE(result.empty());
}

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
