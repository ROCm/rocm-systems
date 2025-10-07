// tests/test_service_tests.cpp
#include "library/amd_smi/service.hpp"

#include <amd_smi/amdsmi.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

namespace
{
struct mock_driver_api
{
    MOCK_METHOD(amdsmi_status_t, init, (), ());
    MOCK_METHOD(amdsmi_status_t, get_version, (amdsmi_version_t*), ());
    MOCK_METHOD(amdsmi_status_t, get_socket_handles, (uint32_t*, amdsmi_socket_handle*),
                ());
    MOCK_METHOD(amdsmi_status_t, get_processor_handles,
                (amdsmi_socket_handle, uint32_t*, amdsmi_processor_handle*), ());
    MOCK_METHOD(amdsmi_status_t, get_processor_type,
                (amdsmi_processor_handle, processor_type_t*), ());
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
    MOCK_METHOD(amdsmi_status_t, shutdown, (), ());
};

std::shared_ptr<StrictMock<mock_driver_api>> g_mock_api_instance = nullptr;
// Mock driver factory
struct mock_driver_factory
{
    using driver_t = mock_driver_api;
    static std::shared_ptr<mock_driver_api> create_driver()
    {
        return g_mock_api_instance;
    }
};

};  // namespace

class ServiceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_mock_api_instance = std::make_shared<StrictMock<mock_driver_api>>();

        ON_CALL(*g_mock_api_instance, init())
            .WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*g_mock_api_instance, get_version(_))
            .WillByDefault([](amdsmi_version_t* v) {
                if(v) *v = amdsmi_version_t{ 1, 2, 3, "build123" };
                return AMDSMI_STATUS_SUCCESS;
            });
        ON_CALL(*g_mock_api_instance, get_socket_handles(_, nullptr))
            .WillByDefault(
                DoAll(SetArgPointee<0>(1), testing::Return(AMDSMI_STATUS_SUCCESS)));

        // Second call is to collect socket handlers
        ON_CALL(*g_mock_api_instance, get_socket_handles(_, testing::NotNull()))
            .WillByDefault(DoAll(SetArgPointee<0>(1),  // Set socket_count to 1
                                 testing::Return(AMDSMI_STATUS_SUCCESS)));

        ON_CALL(*g_mock_api_instance, get_processor_handles(_, _, nullptr))
            .WillByDefault(DoAll(SetArgPointee<1>(1), Return(AMDSMI_STATUS_SUCCESS)));

        ON_CALL(*g_mock_api_instance, get_processor_handles(_, _, testing::NotNull()))
            .WillByDefault(DoAll(SetArgPointee<1>(1), Return(AMDSMI_STATUS_SUCCESS)));

        ON_CALL(*g_mock_api_instance, get_processor_type(_, _))
            .WillByDefault(DoAll(SetArgPointee<1>(AMDSMI_PROCESSOR_TYPE_AMD_GPU),
                                 Return(AMDSMI_STATUS_SUCCESS)));
    }
    void TearDown() override { g_mock_api_instance.reset(); }
};

TEST_F(ServiceTest, ConstructSuccess)
{
    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_version(_));

    EXPECT_NO_THROW({ rocprofsys::amd_smi::service<mock_driver_factory> svc; });
}

TEST_F(ServiceTest, ConstructInitFail)
{
    EXPECT_CALL(*g_mock_api_instance, init()).WillOnce(Return(AMDSMI_STATUS_INIT_ERROR));
    EXPECT_THROW(
        { rocprofsys::amd_smi::service<mock_driver_factory> svc; }, std::runtime_error);
}

TEST_F(ServiceTest, ConstructVersionFail)
{
    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_version(_))
        .WillOnce(Return(AMDSMI_STATUS_INIT_ERROR));
    EXPECT_THROW(
        { rocprofsys::amd_smi::service<mock_driver_factory> svc; }, std::runtime_error);
}

TEST_F(ServiceTest, GetVersionReturnsCorrect)
{
    amdsmi_version_t version{ 4, 5, 6, "build456" };
    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_version(_))
        .WillOnce(DoAll(SetArgPointee<0>(version), Return(AMDSMI_STATUS_SUCCESS)));
    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    auto                                              v = svc.get_version();
    EXPECT_EQ(v.numeric_representation.major, 4);
    EXPECT_EQ(v.numeric_representation.minor, 5);
    EXPECT_EQ(v.numeric_representation.release, 6);
    EXPECT_EQ(v.string_representation, "build456");
}

TEST_F(ServiceTest, GetProcessorsSuccess)
{
    const auto number_of_fake_processors{ 2 };

    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_power_info).Times(number_of_fake_processors);
    EXPECT_CALL(*g_mock_api_instance, get_activity).Times(number_of_fake_processors);
    EXPECT_CALL(*g_mock_api_instance, get_memory_usage).Times(number_of_fake_processors);
    EXPECT_CALL(*g_mock_api_instance, get_temperature_metric)
        .Times(2 * number_of_fake_processors);
    EXPECT_CALL(*g_mock_api_instance, get_metrics_info).Times(number_of_fake_processors);
    EXPECT_CALL(*g_mock_api_instance, get_version(_));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, nullptr))
        .Times(1)
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, testing::NotNull()))
        .Times(1)
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(AMDSMI_STATUS_SUCCESS)));

    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, nullptr))
        .Times(1)
        .WillOnce(DoAll(SetArgPointee<1>(2), Return(AMDSMI_STATUS_SUCCESS)));

    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, testing::NotNull()))
        .Times(1)
        .WillOnce(DoAll(SetArgPointee<1>(2), Return(AMDSMI_STATUS_SUCCESS)));

    EXPECT_CALL(*g_mock_api_instance, get_processor_type(_, _))
        .Times(2)
        .WillRepeatedly(DoAll(SetArgPointee<1>(AMDSMI_PROCESSOR_TYPE_AMD_CPU),
                              Return(AMDSMI_STATUS_SUCCESS)));

    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    auto                                              processors = svc.get_processors();
    EXPECT_EQ(processors.size(), 2);
}

TEST_F(ServiceTest, GetProcessorsSocketFail)
{
    EXPECT_CALL(*g_mock_api_instance, init());
    EXPECT_CALL(*g_mock_api_instance, get_version(_));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, nullptr))
        .WillOnce(Return(AMDSMI_STATUS_INIT_ERROR));
    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    EXPECT_THROW(svc.get_processors(), std::runtime_error);
}

TEST_F(ServiceTest, GetProcessorsProcessorHandlesFail)
{
    EXPECT_CALL(*g_mock_api_instance, init());
    EXPECT_CALL(*g_mock_api_instance, get_version(_));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles).Times(2);
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, nullptr))
        .WillOnce(Return(AMDSMI_STATUS_INIT_ERROR));
    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    EXPECT_THROW(svc.get_processors(), std::runtime_error);
}

TEST_F(ServiceTest, GetProcessorsProcessorTypeFail)
{
    EXPECT_CALL(*g_mock_api_instance, init());
    EXPECT_CALL(*g_mock_api_instance, get_version(_));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, _)).Times(2);
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, _)).Times(2);
    EXPECT_CALL(*g_mock_api_instance, get_processor_type(_, _))
        .WillOnce(Return(AMDSMI_STATUS_INIT_ERROR));
    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    EXPECT_THROW(svc.get_processors(), std::runtime_error);
}

TEST_F(ServiceTest, GetProcessorsWithFilter)
{
    const auto number_of_fake_processors{ 3 };

    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_power_info).Times(number_of_fake_processors);
    EXPECT_CALL(*g_mock_api_instance, get_activity).Times(number_of_fake_processors);
    EXPECT_CALL(*g_mock_api_instance, get_memory_usage).Times(number_of_fake_processors);
    EXPECT_CALL(*g_mock_api_instance, get_temperature_metric)
        .Times(2 * number_of_fake_processors);
    EXPECT_CALL(*g_mock_api_instance, get_metrics_info).Times(number_of_fake_processors);
    EXPECT_CALL(*g_mock_api_instance, get_version(_));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, nullptr))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, testing::NotNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<1>(3), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, testing::NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(3), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_processor_type(_, _))
        .Times(3)
        .WillRepeatedly(DoAll(SetArgPointee<1>(AMDSMI_PROCESSOR_TYPE_AMD_GPU),
                              Return(AMDSMI_STATUS_SUCCESS)));

    rocprofsys::amd_smi::service<mock_driver_factory> svc;

    // Filter to return only first 2 processors
    auto filter = [](auto& processors) {
        if(processors.size() > 2) processors.resize(2);
        return processors;
    };

    auto processors = svc.get_processors(filter);
    EXPECT_EQ(processors.size(), 2);
}

TEST_F(ServiceTest, GetProcessorsMultipleSockets)
{
    const auto number_of_sockets{ 2 };
    const auto processors_per_socket{ 2 };
    const auto total_processors{ number_of_sockets * processors_per_socket };

    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_power_info).Times(total_processors);
    EXPECT_CALL(*g_mock_api_instance, get_activity).Times(total_processors);
    EXPECT_CALL(*g_mock_api_instance, get_memory_usage).Times(total_processors);
    EXPECT_CALL(*g_mock_api_instance, get_temperature_metric).Times(2 * total_processors);
    EXPECT_CALL(*g_mock_api_instance, get_metrics_info).Times(total_processors);
    EXPECT_CALL(*g_mock_api_instance, get_version(_));

    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, nullptr))
        .WillOnce(
            DoAll(SetArgPointee<0>(number_of_sockets), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, testing::NotNull()))
        .WillOnce(
            DoAll(SetArgPointee<0>(number_of_sockets), Return(AMDSMI_STATUS_SUCCESS)));

    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, nullptr))
        .Times(number_of_sockets)
        .WillRepeatedly(DoAll(SetArgPointee<1>(processors_per_socket),
                              Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, testing::NotNull()))
        .Times(number_of_sockets)
        .WillRepeatedly(DoAll(SetArgPointee<1>(processors_per_socket),
                              Return(AMDSMI_STATUS_SUCCESS)));

    EXPECT_CALL(*g_mock_api_instance, get_processor_type(_, _))
        .Times(total_processors)
        .WillRepeatedly(DoAll(SetArgPointee<1>(AMDSMI_PROCESSOR_TYPE_AMD_GPU),
                              Return(AMDSMI_STATUS_SUCCESS)));

    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    auto                                              processors = svc.get_processors();
    EXPECT_EQ(processors.size(), total_processors);

    // Verify processor indices are sequential
    for(size_t i = 0; i < processors.size(); ++i)
    {
        EXPECT_EQ(processors[i]->get_index(), i);
    }
}

TEST_F(ServiceTest, GetProcessorsNoProcessors)
{
    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_version);
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles).Times(2);

    EXPECT_CALL(*g_mock_api_instance, get_processor_handles)
        .WillRepeatedly(DoAll(SetArgPointee<1>(0), Return(AMDSMI_STATUS_SUCCESS)));

    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    auto                                              processors = svc.get_processors();
    EXPECT_EQ(processors.size(), 0);
}

TEST_F(ServiceTest, GetProcessorsNoSockets)
{
    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_version(_));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, nullptr))
        .WillOnce(DoAll(SetArgPointee<0>(0), Return(AMDSMI_STATUS_SUCCESS)));

    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    auto                                              processors = svc.get_processors();
    EXPECT_EQ(processors.size(), 0);
}

TEST_F(ServiceTest, GetProcessorsSocketHandlesSecondCallFail)
{
    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_version(_));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, nullptr))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, testing::NotNull()))
        .WillOnce(Return(AMDSMI_STATUS_INIT_ERROR));

    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    EXPECT_THROW(svc.get_processors(), std::runtime_error);
}

TEST_F(ServiceTest, GetProcessorsProcessorHandlesSecondCallFail)
{
    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_version(_));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, _)).Times(2);
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<1>(1), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, testing::NotNull()))
        .WillOnce(Return(AMDSMI_STATUS_INIT_ERROR));

    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    EXPECT_THROW(svc.get_processors(), std::runtime_error);
}

TEST_F(ServiceTest, GetVersionReturnsReference)
{
    amdsmi_version_t version{ 7, 8, 9, "build789" };
    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_version(_))
        .WillOnce(DoAll(SetArgPointee<0>(version), Return(AMDSMI_STATUS_SUCCESS)));

    rocprofsys::amd_smi::service<mock_driver_factory> svc;

    // Test that we get a reference and can call it multiple times
    const auto& v1 = svc.get_version();
    const auto& v2 = svc.get_version();

    EXPECT_EQ(&v1, &v2);  // Same reference
    EXPECT_EQ(v1.numeric_representation.major, 7);
    EXPECT_EQ(v1.numeric_representation.minor, 8);
    EXPECT_EQ(v1.numeric_representation.release, 9);
    EXPECT_EQ(v1.string_representation, "build789");
}

TEST_F(ServiceTest, GetProcessorsMixedTypes)
{
    const auto number_of_processors{ 3 };

    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_power_info).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_activity).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_memory_usage).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_temperature_metric)
        .Times(2 * number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_metrics_info).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_version(_));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, _)).Times(2);
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, nullptr))
        .WillOnce(
            DoAll(SetArgPointee<1>(number_of_processors), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, testing::NotNull()))
        .WillOnce(
            DoAll(SetArgPointee<1>(number_of_processors), Return(AMDSMI_STATUS_SUCCESS)));

    // Return different processor types
    EXPECT_CALL(*g_mock_api_instance, get_processor_type)
        .Times(3)
        .WillOnce(DoAll(SetArgPointee<1>(AMDSMI_PROCESSOR_TYPE_AMD_GPU),
                        Return(AMDSMI_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<1>(AMDSMI_PROCESSOR_TYPE_AMD_CPU),
                        Return(AMDSMI_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<1>(AMDSMI_PROCESSOR_TYPE_AMD_GPU),
                        Return(AMDSMI_STATUS_SUCCESS)));

    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    auto                                              processors = svc.get_processors();

    EXPECT_EQ(processors.size(), 3);
    EXPECT_EQ(processors[0]->get_processor_type(), AMDSMI_PROCESSOR_TYPE_AMD_GPU);
    EXPECT_EQ(processors[1]->get_processor_type(), AMDSMI_PROCESSOR_TYPE_AMD_CPU);
    EXPECT_EQ(processors[2]->get_processor_type(), AMDSMI_PROCESSOR_TYPE_AMD_GPU);
}

TEST_F(ServiceTest, GetProcessorsFilterReordersProcessors)
{
    const auto number_of_processors{ 3 };

    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_power_info).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_activity).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_memory_usage).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_temperature_metric)
        .Times(2 * number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_metrics_info).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_version(_));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, _)).Times(2);
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, nullptr))
        .WillOnce(
            DoAll(SetArgPointee<1>(number_of_processors), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, testing::NotNull()))
        .WillOnce(
            DoAll(SetArgPointee<1>(number_of_processors), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_processor_type).Times(number_of_processors);

    rocprofsys::amd_smi::service<mock_driver_factory> svc;

    // Filter that reverses the order
    auto reverse_filter = [](auto& processors) {
        std::reverse(processors.begin(), processors.end());
        return processors;
    };

    auto processors = svc.get_processors(reverse_filter);

    EXPECT_EQ(processors.size(), 3);
    // Original indices should be 0, 1, 2, but after reverse: 2, 1, 0
    EXPECT_EQ(processors[0]->get_index(), 2);
    EXPECT_EQ(processors[1]->get_index(), 1);
    EXPECT_EQ(processors[2]->get_index(), 0);
}

TEST_F(ServiceTest, GetProcessorsFilterReturnsEmpty)
{
    const auto number_of_processors{ 2 };

    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_power_info).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_activity).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_memory_usage).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_temperature_metric)
        .Times(2 * number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_metrics_info).Times(number_of_processors);
    EXPECT_CALL(*g_mock_api_instance, get_version(_));
    EXPECT_CALL(*g_mock_api_instance, get_socket_handles(_, _)).Times(2);
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, nullptr))
        .WillOnce(
            DoAll(SetArgPointee<1>(number_of_processors), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_processor_handles(_, _, testing::NotNull()))
        .WillOnce(
            DoAll(SetArgPointee<1>(number_of_processors), Return(AMDSMI_STATUS_SUCCESS)));
    EXPECT_CALL(*g_mock_api_instance, get_processor_type).Times(number_of_processors);

    rocprofsys::amd_smi::service<mock_driver_factory> svc;

    // Filter that returns empty vector
    auto empty_filter = [](auto& processors) {
        processors.clear();
        return processors;
    };

    auto processors = svc.get_processors(empty_filter);
    EXPECT_EQ(processors.size(), 0);
}

TEST_F(ServiceTest, ConstructWithEmptyBuildString)
{
    amdsmi_version_t version{ 1, 2, 3, "" };  // Empty build string
    EXPECT_CALL(*g_mock_api_instance, init);
    EXPECT_CALL(*g_mock_api_instance, get_version(_))
        .WillOnce(DoAll(SetArgPointee<0>(version), Return(AMDSMI_STATUS_SUCCESS)));

    rocprofsys::amd_smi::service<mock_driver_factory> svc;
    auto                                              v = svc.get_version();
    EXPECT_EQ(v.string_representation, "");
}
