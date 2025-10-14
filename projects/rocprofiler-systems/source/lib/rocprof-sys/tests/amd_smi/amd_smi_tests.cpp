#include "library/amd_smi/amd_smi.hpp"
#include "library/amd_smi/common.hpp"

#include "gmock/gmock.h"
#include <algorithm>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <vector>

using namespace rocprofsys::amd_smi;
using testing::_;
using testing::DoAll;
using testing::SaveArg;

namespace
{

struct mock_settings_api
{
    MOCK_METHOD(smi_metric_options, get_enabled_metrics, (), ());
    MOCK_METHOD(device_filter, get_device_filter, (), ());
};

struct mock_processor
{
    MOCK_METHOD(size_t, get_index, (), ());
    MOCK_METHOD(smi_metric_options, get_supported_metrics, (), ());
    MOCK_METHOD(smi_metrics, get_smi_metrics, (), ());
};

struct mock_smi_service
{
    using processor_t        = mock_processor;
    using processor_vector_t = std::vector<std::shared_ptr<mock_processor>>;

    MOCK_METHOD(version, get_version, (), ());
    MOCK_METHOD(processor_vector_t, get_processors,
                (std::function<processor_vector_t(const processor_vector_t&)>), ());
};

std::shared_ptr<mock_smi_service> g_mock_smi_service;

struct mock_smi_service_factory
{
    using smi_service = mock_smi_service;
    static std::shared_ptr<mock_smi_service> create_smi_service()
    {
        return g_mock_smi_service;
    }
};

struct mock_perfetto_api
{
    MOCK_METHOD(smi_metric_options, init_storage, (), ());
    MOCK_METHOD(void, setup_counter_tracks, (const size_t, const smi_metric_options&),
                ());
};

struct mock_rocpd_api
{
    MOCK_METHOD(void, initialize_category_metadata, (), ());
    MOCK_METHOD(void, store_sample,
                (size_t _device_id, const smi_metric_options& _supported_metrics,
                 const smi_metric_options& _enabled_metrics,
                 const smi_metrics& _smi_metrics, unsigned long _timestamp),
                ());
};

std::shared_ptr<testing::StrictMock<mock_perfetto_api>> g_mock_perfetto_api;
std::shared_ptr<testing::StrictMock<mock_rocpd_api>>    g_mock_rocpd_api;
std::shared_ptr<testing::StrictMock<mock_settings_api>> g_mock_settings_api;

struct mock_smi_processor
{};

struct _settings_api
{
    static smi_metric_options get_enabled_metrics()
    {
        return g_mock_settings_api->get_enabled_metrics();
    }

    static device_filter get_device_filter()
    {
        return g_mock_settings_api->get_device_filter();
    }
};

struct _perfetto_api
{
    static void setup_counter_tracks(size_t device_id, const smi_metric_options& metrics)
    {
        g_mock_perfetto_api->setup_counter_tracks(device_id, metrics);
    };

    static void init_storage(size_t) {};

    static void store_sample(size_t /*device_index*/, const smi_metrics& /*_smi_metrics*/,
                             unsigned long /*_timestamp*/) {};

    static void post_process(size_t /*device_index*/,
                             smi_metric_options /*enabled_metrics*/,
                             smi_metric_options /*supported_metrics*/)
    {}
};

struct _rocpd_api
{
    static void initialize_smi_tracks_metadata(size_t /*gpu_id*/) {}

    static void initialize_smi_pmc_metadata(size_t /*gpu_id*/) {}

    static void initialize_category_metadata()
    {
        g_mock_rocpd_api->initialize_category_metadata();
    }

    static void store_sample(size_t                    _device_id,
                             const smi_metric_options& _supported_metrics,
                             const smi_metric_options& _enabled_metrics,
                             const smi_metrics& _smi_metrics, unsigned long _timestamp)
    {
        g_mock_rocpd_api->store_sample(_device_id, _supported_metrics, _enabled_metrics,
                                       _smi_metrics, _timestamp);
    };
};

}  // namespace

class AmdSmiTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_mock_smi_service  = std::make_shared<testing::StrictMock<mock_smi_service>>();
        g_mock_settings_api = std::make_shared<testing::StrictMock<mock_settings_api>>();
        g_mock_perfetto_api = std::make_shared<testing::StrictMock<mock_perfetto_api>>();
        g_mock_rocpd_api    = std::make_shared<testing::StrictMock<mock_rocpd_api>>();
    }

    void TearDown() override
    {
        g_mock_smi_service.reset();
        g_mock_settings_api.reset();
        g_mock_perfetto_api.reset();
        g_mock_rocpd_api.reset();
    }
};

TEST_F(AmdSmiTests, AmdSmiSetup)
{
    amd_smi_impl<mock_smi_service_factory, _settings_api, _perfetto_api, _rocpd_api>
        smi_impl;
    EXPECT_CALL(*g_mock_smi_service, get_version)
        .WillRepeatedly(testing::Return(version{
            .numeric_representation = { .major = 1, .minor = 2, .release = 3 } }));

    EXPECT_CALL(*g_mock_smi_service, get_processors);
    smi_impl.setup();
}

TEST_F(AmdSmiTests, AmdSmiConfig)
{
    amd_smi_impl<mock_smi_service_factory, _settings_api, _perfetto_api, _rocpd_api>
        smi_impl;

    // Test config
    smi_metric_options enabled_metrics;
    EXPECT_CALL(*g_mock_settings_api, get_enabled_metrics())
        .WillOnce(testing::Return(enabled_metrics));
    EXPECT_CALL(*g_mock_rocpd_api, initialize_category_metadata);
    smi_impl.config();
}

TEST_F(AmdSmiTests, AmdSmiSampleSuccess)
{
    amd_smi_impl<mock_smi_service_factory, _settings_api, _perfetto_api, _rocpd_api>
        smi_impl;

    size_t device_id = 0;

    // Setup first
    EXPECT_CALL(*g_mock_smi_service, get_version)
        .WillOnce(testing::Return(version{
            .numeric_representation = { .major = 1, .minor = 2, .release = 3 } }));

    auto _processor_mock = std::make_shared<testing::StrictMock<mock_processor>>();

    EXPECT_CALL(*g_mock_smi_service, get_processors)
        .WillOnce(testing::Return(
            std::vector<std::shared_ptr<mock_processor>>{ _processor_mock }));

    EXPECT_CALL(*_processor_mock, get_index).WillRepeatedly(testing::Return(device_id));

    smi_impl.setup();

    // Test sample
    smi_metric_options enabled_metrics{ .value = 0x5555 };
    smi_metric_options supported_metrics{ .value = 0x5555 };
    smi_metrics        sample_metrics{
               .current_socket_power = 1,
               .average_socket_power = 2,
               .memory_usage         = 3,
               .hotspot_temperature  = 4,
               .edge_temperature     = 5,
               .gfx_activity         = 6,
               .umc_activity         = 7,
               .mm_activity          = 8,
    };

    std::for_each(
        std::begin(sample_metrics.xcp_stats), std::end(sample_metrics.xcp_stats),
        [](auto& xcp_stats) {
            std::for_each(
                std::begin(xcp_stats.vcn_busy), std::end(xcp_stats.vcn_busy),
                [counter = 0](auto& vcn_busy) mutable { vcn_busy = counter++; });
            std::for_each(
                std::begin(xcp_stats.jpeg_busy), std::end(xcp_stats.jpeg_busy),
                [counter = 0](auto& jpeg_busy) mutable { jpeg_busy = counter++; });
        });

    EXPECT_CALL(*g_mock_settings_api, get_enabled_metrics)
        .WillOnce(testing::Return(enabled_metrics));

    EXPECT_CALL(*_processor_mock, get_supported_metrics)
        .WillRepeatedly(testing::Return(supported_metrics));

    EXPECT_CALL(*_processor_mock, get_smi_metrics())
        .WillOnce(testing::Return(sample_metrics));

    size_t             argument_device_id;
    smi_metric_options argument_supported_metrics;
    smi_metric_options argument_enabled_metrics;
    smi_metrics        argument_sample_metrics;
    unsigned long      argument_timestamp;

    EXPECT_CALL(*g_mock_rocpd_api, store_sample)
        .WillOnce(DoAll(
            SaveArg<0>(&argument_device_id), SaveArg<1>(&argument_supported_metrics),
            SaveArg<2>(&argument_enabled_metrics), SaveArg<3>(&argument_sample_metrics),
            SaveArg<4>(&argument_timestamp)));

    smi_impl.sample([] { return 123; });

    EXPECT_EQ(device_id, argument_device_id);
    EXPECT_EQ(supported_metrics.value, argument_supported_metrics.value);
    EXPECT_EQ(enabled_metrics.value, argument_enabled_metrics.value);
    EXPECT_TRUE(
        std::memcmp(&sample_metrics, &argument_sample_metrics, sizeof(smi_metrics)) == 0);
    EXPECT_EQ(123, argument_timestamp);
}

TEST_F(AmdSmiTests, AmdSmiPostProcess)
{
    amd_smi_impl<mock_smi_service_factory, _settings_api, _perfetto_api, _rocpd_api>
        smi_impl;

    EXPECT_CALL(*g_mock_smi_service, get_version)
        .WillOnce(testing::Return(version{
            .numeric_representation = { .major = 1, .minor = 2, .release = 3 } }));

    auto _processor_mock = std::make_shared<testing::StrictMock<mock_processor>>();

    EXPECT_CALL(*_processor_mock, get_index()).WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*_processor_mock, get_supported_metrics)
        .WillRepeatedly(testing::Return(smi_metric_options{ .value = 0xffff }));

    EXPECT_CALL(*g_mock_smi_service, get_processors)
        .WillOnce(testing::Return(
            std::vector<std::shared_ptr<mock_processor>>{ _processor_mock }));

    smi_impl.setup();

    smi_metric_options enabled_metrics;
    smi_metric_options supported_metrics;

    EXPECT_CALL(*g_mock_settings_api, get_enabled_metrics())
        .WillOnce(testing::Return(enabled_metrics));

    EXPECT_CALL(*_processor_mock, get_supported_metrics())
        .WillOnce(testing::Return(supported_metrics));

    smi_impl.post_process();
}

TEST_F(AmdSmiTests, AmdSmiSetupWithDeviceFilter)
{
    amd_smi_impl<mock_smi_service_factory, _settings_api, _perfetto_api, _rocpd_api>
        smi_impl;

    EXPECT_CALL(*g_mock_smi_service, get_version)
        .WillOnce(testing::Return(version{
            .numeric_representation = { .major = 1, .minor = 2, .release = 3 } }));

    auto mock_processor1 = std::make_shared<testing::StrictMock<mock_processor>>();
    auto mock_processor2 = std::make_shared<testing::StrictMock<mock_processor>>();
    auto mock_processor3 = std::make_shared<testing::StrictMock<mock_processor>>();
    auto list_of_mock_processors =
        std::vector<std::shared_ptr<mock_processor>>{ mock_processor1, mock_processor2,
                                                      mock_processor3 };
    EXPECT_CALL(*mock_processor1, get_index()).WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*mock_processor2, get_index()).WillRepeatedly(testing::Return(1));
    EXPECT_CALL(*mock_processor3, get_index()).WillRepeatedly(testing::Return(2));

    // Test with specific device filter
    device_filter filter;
    filter.mode = device_selection_mode::specific;
    filter.indices.insert(0);
    filter.indices.insert(2);

    using filter_callback_t = std::function<mock_smi_service::processor_vector_t(
        mock_smi_service::processor_vector_t&)>;
    filter_callback_t captured_filter_callback;

    EXPECT_CALL(*g_mock_settings_api, get_device_filter)
        .WillOnce(testing::Return(filter));

    EXPECT_CALL(*g_mock_smi_service, get_processors)
        .WillOnce(DoAll(testing::SaveArg<0>(&captured_filter_callback),
                        testing::Return(list_of_mock_processors)));

    smi_impl.setup();

    auto filtered_result = captured_filter_callback(list_of_mock_processors);
    EXPECT_EQ(2, filtered_result.size());
}

TEST_F(AmdSmiTests, AmdSmiSetupWithNoDevices)
{
    amd_smi_impl<mock_smi_service_factory, _settings_api, _perfetto_api, _rocpd_api>
        smi_impl;

    EXPECT_CALL(*g_mock_smi_service, get_version)
        .WillOnce(testing::Return(version{
            .numeric_representation = { .major = 1, .minor = 2, .release = 3 } }));

    EXPECT_CALL(*g_mock_smi_service, get_processors)
        .WillOnce(testing::Return(std::vector<std::shared_ptr<mock_processor>>{}));

    device_filter filter;
    filter.mode = device_selection_mode::none;

    smi_impl.setup();
}
