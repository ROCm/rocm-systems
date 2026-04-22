// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/base/collector.hpp"
#include "library/pmc/collectors/gpu_perf_counter/gpu_perf_counter_traits.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/device_providers/rocprofiler_sdk/drivers/tests/mock_driver.hpp"
#include "library/pmc/device_providers/rocprofiler_sdk/provider.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace rocprofsys::pmc::collectors::gpu_perf_counter;
using MockDriver = rocprofsys::pmc::drivers::rocprofiler_sdk::testing::mock_driver;
using MockDriverFactory =
    rocprofsys::pmc::drivers::rocprofiler_sdk::testing::mock_driver_factory;
using MockProvider =
    rocprofsys::pmc::device_providers::rocprofiler_sdk::provider<MockDriverFactory>;

using ::testing::_;
using ::testing::Return;

namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
{

// ============================================================================
// Test stub policies (replace real cache/perfetto/settings to avoid globals)
// ============================================================================

struct captured_sample
{
    size_t   device_id;
    uint64_t timestamp;
    metrics  values;
};

static std::vector<captured_sample>&
get_captured_samples()
{
    static std::vector<captured_sample> samples;
    return samples;
}

struct test_cache_policy
{
    static void initialize_category_metadata() {}
    static void initialize_tracks_metadata() {}
    static void initialize_pmc_metadata(size_t, const std::vector<counter_metadata>&) {}

    static void store_sample(size_t device_id, const std::string&, const enabled_metrics&,
                             const enabled_metrics&, const metrics& values,
                             uint64_t timestamp)
    {
        if(values.empty()) return;
        get_captured_samples().push_back({ device_id, timestamp, values });
    }
};

struct test_perfetto_policy
{
    using counter_track = void;

    template <typename DeviceEntries>
    static void init_storage(const DeviceEntries&)
    {}
    static void setup_counter_tracks(size_t, const std::vector<counter_metadata>&) {}
    static void store_sample(size_t, const metrics&, uint64_t) {}
    static void post_process(const enabled_metrics&) {}
};

struct test_settings_policy
{
    static enabled_metrics get_gpu_perf_counter_enabled_metrics()
    {
        return enabled_metrics{ { counter_definition{ "SQ_WAVES", 0 },
                                  counter_definition{ "SQ_BUSY_CYCLES", 0 } } };
    }

    static bool get_use_perfetto_legacy_metrics() { return false; }
};

struct test_config
{
    using SettingsApi = test_settings_policy;
    using PerfettoApi = test_perfetto_policy;
    using CacheApi    = test_cache_policy;
};

using test_collector_t =
    base::collector<gpu_perf_counter_traits<MockProvider>, MockProvider, test_config>;

// ============================================================================
// Helper: build a fake agent
// ============================================================================

static std::shared_ptr<rocprofsys::agent>
make_agent(uint64_t handle, size_t device_type_index, const std::string& name)
{
    auto a               = std::make_shared<rocprofsys::agent>();
    a->type              = agent_type::GPU;
    a->handle            = handle;
    a->device_id         = device_type_index;
    a->device_type_index = device_type_index;
    a->name              = name;
    a->product_name      = name;
    a->vendor_name       = "AMD";
    return a;
}

// ============================================================================
// Helper: set up mock expectations for provider construction (per agent)
// ============================================================================

struct counter_setup
{
    rocprofiler_counter_id_t                             id;
    const char*                                          name;
    rocprofiler_counter_record_dimension_instance_info_t dim_instance;
    rocprofiler_counter_dimension_info_t                 dim_info;
};

static void
setup_provider_expectations(std::shared_ptr<MockDriver>& mock, uint64_t agent_handle,
                            std::vector<counter_setup>& counters,
                            uint32_t                    context_handle_out)
{
    // iterate_agent_supported_counters: provide counter IDs to callback
    EXPECT_CALL(
        *mock, iterate_agent_supported_counters(
                   ::testing::Field(&rocprofiler_agent_id_t::handle, agent_handle), _, _))
        .WillOnce([&counters](rocprofiler_agent_id_t,
                              rocprofiler_available_counters_cb_t cb, void* user_data) {
            std::vector<rocprofiler_counter_id_t> ids;
            ids.reserve(counters.size());
            for(auto& c : counters)
                ids.push_back(c.id);
            cb({}, ids.data(), ids.size(), user_data);
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // query_counter_info: called by filter_counters and resolve_counter_details
    for(auto& c : counters)
    {
        c.dim_info = { sizeof(rocprofiler_counter_dimension_info_t), "DIMENSION", 0 };

        c.dim_instance                  = {};
        c.dim_instance.size             = sizeof(c.dim_instance);
        c.dim_instance.instance_id      = c.id.handle;
        c.dim_instance.counter_id       = c.id.handle;
        c.dim_instance.dimensions_count = 0;
        c.dim_instance.dimensions       = nullptr;

        EXPECT_CALL(
            *mock,
            query_counter_info(
                ::testing::Field(&rocprofiler_counter_id_t::handle, c.id.handle), _, _))
            .WillRepeatedly([&c](rocprofiler_counter_id_t,
                                 rocprofiler_counter_info_version_id_t, void* info_out) {
                auto* info        = static_cast<rocprofiler_counter_info_v1_t*>(info_out);
                *info             = {};
                info->id          = c.id;
                info->name        = c.name;
                info->description = "";
                info->block       = "";
                info->expression  = "";
                info->is_constant = 0;
                info->is_derived  = 0;
                info->dimensions_count = 0;
                info->dimensions       = nullptr;

                const rocprofiler_counter_record_dimension_instance_info_t* dim_ptr =
                    &c.dim_instance;
                info->dimensions_instances_count = 1;
                info->dimensions_instances       = &dim_ptr;

                return ROCPROFILER_STATUS_SUCCESS;
            });
    }

    // create_counter_config
    EXPECT_CALL(
        *mock,
        create_counter_config(
            ::testing::Field(&rocprofiler_agent_id_t::handle, agent_handle), _, _, _))
        .WillOnce([](rocprofiler_agent_id_t, rocprofiler_counter_id_t*, size_t,
                     rocprofiler_counter_config_id_t* config_id) {
            config_id->handle = 200;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // create_context (per-agent)
    EXPECT_CALL(*mock, create_context(_))
        .WillOnce([context_handle_out](rocprofiler_context_id_t* ctx) {
            ctx->handle = context_handle_out;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // configure_device_counting_service
    EXPECT_CALL(*mock, configure_device_counting_service(
                           ::testing::Field(&rocprofiler_context_id_t::handle,
                                            context_handle_out),
                           _, _, _, _))
        .WillOnce(Return(ROCPROFILER_STATUS_SUCCESS));

    // start_context
    EXPECT_CALL(*mock, start_context(::testing::Field(&rocprofiler_context_id_t::handle,
                                                      context_handle_out)))
        .WillOnce(Return(ROCPROFILER_STATUS_SUCCESS));

    // stop_context
    EXPECT_CALL(*mock, stop_context(::testing::Field(&rocprofiler_context_id_t::handle,
                                                     context_handle_out)))
        .WillOnce(Return(ROCPROFILER_STATUS_SUCCESS));
}

// ============================================================================
// Fixture
// ============================================================================

class SdkPmcCollectorWorkflowTest : public ::testing::Test
{
protected:
    std::shared_ptr<MockDriver> mock;

    void SetUp() override
    {
        get_captured_samples().clear();
        mock = std::make_shared<MockDriver>();
        MockDriverFactory::set_mock(mock);
    }

    void TearDown() override
    {
        MockDriverFactory::reset();
        get_captured_samples().clear();
    }
};

// ============================================================================
// Tests
// ============================================================================

TEST_F(SdkPmcCollectorWorkflowTest, SingleGpuWorkflow)
{
    auto agent = make_agent(42, 0, "GPU 0");

    std::vector<counter_setup> counters = {
        { rocprofiler_counter_id_t{ 10 }, "SQ_WAVES", {}, {} },
        { rocprofiler_counter_id_t{ 20 }, "SQ_BUSY_CYCLES", {}, {} },
    };

    setup_provider_expectations(mock, 42, counters, /*context=*/100);

    // sample_device_counting_service: return 2 records
    EXPECT_CALL(
        *mock, sample_device_counting_service(
                   ::testing::Field(&rocprofiler_context_id_t::handle, 100u), _, _, _, _))
        .WillRepeatedly([](rocprofiler_context_id_t, rocprofiler_user_data_t,
                           rocprofiler_counter_flag_t, rocprofiler_counter_record_t* out,
                           size_t* count) {
            out[0].id            = 10;
            out[0].counter_value = 42.0;
            out[1].id            = 20;
            out[1].counter_value = 99.0;
            *count               = 2;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // Build real provider + collector
    auto provider = std::make_shared<MockProvider>(
        std::vector<std::shared_ptr<rocprofsys::agent>>{ agent },
        test_settings_policy::get_gpu_perf_counter_enabled_metrics());
    provider->start();

    test_collector_t collector(provider);
    collector.setup();
    collector.config();

    collector.sample(1000);
    collector.sample(2000);
    collector.sample(3000);

    collector.post_process();
    collector.shutdown();

    auto& samples = get_captured_samples();
    ASSERT_EQ(samples.size(), 3U);

    for(size_t i = 0; i < 3; ++i)
    {
        EXPECT_EQ(samples[i].device_id, 0U);
        EXPECT_EQ(samples[i].timestamp, (i + 1) * 1000U);
        ASSERT_EQ(samples[i].values.size(), 2U);
        EXPECT_EQ(samples[i].values[0].counter_id, 10U);
        EXPECT_DOUBLE_EQ(samples[i].values[0].value, 42.0);
        EXPECT_EQ(samples[i].values[1].counter_id, 20U);
        EXPECT_DOUBLE_EQ(samples[i].values[1].value, 99.0);
    }
}

TEST_F(SdkPmcCollectorWorkflowTest, MultiGpuIsolation)
{
    auto agent0 = make_agent(10, 0, "GPU 0");
    auto agent1 = make_agent(11, 1, "GPU 1");

    std::vector<counter_setup> counters0 = {
        { rocprofiler_counter_id_t{ 100 }, "SQ_WAVES", {}, {} },
    };
    std::vector<counter_setup> counters1 = {
        { rocprofiler_counter_id_t{ 200 }, "SQ_WAVES", {}, {} },
    };

    // Expectations must be set in the order the provider processes agents.
    // Provider iterates agent_list sequentially, calling these per-agent.

    // Agent 0: iterate_agent_supported_counters
    EXPECT_CALL(*mock, iterate_agent_supported_counters(
                           ::testing::Field(&rocprofiler_agent_id_t::handle, 10u), _, _))
        .WillOnce([&](rocprofiler_agent_id_t, rocprofiler_available_counters_cb_t cb,
                      void* ud) {
            rocprofiler_counter_id_t id = { 100 };
            cb({}, &id, 1, ud);
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // Agent 1: iterate_agent_supported_counters
    EXPECT_CALL(*mock, iterate_agent_supported_counters(
                           ::testing::Field(&rocprofiler_agent_id_t::handle, 11u), _, _))
        .WillOnce([&](rocprofiler_agent_id_t, rocprofiler_available_counters_cb_t cb,
                      void* ud) {
            rocprofiler_counter_id_t id = { 200 };
            cb({}, &id, 1, ud);
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // query_counter_info for counter 100 and 200
    for(auto& cs : counters0)
    {
        cs.dim_instance                  = {};
        cs.dim_instance.size             = sizeof(cs.dim_instance);
        cs.dim_instance.instance_id      = cs.id.handle;
        cs.dim_instance.counter_id       = cs.id.handle;
        cs.dim_instance.dimensions_count = 0;
        cs.dim_instance.dimensions       = nullptr;
    }
    for(auto& cs : counters1)
    {
        cs.dim_instance                  = {};
        cs.dim_instance.size             = sizeof(cs.dim_instance);
        cs.dim_instance.instance_id      = cs.id.handle;
        cs.dim_instance.counter_id       = cs.id.handle;
        cs.dim_instance.dimensions_count = 0;
        cs.dim_instance.dimensions       = nullptr;
    }

    EXPECT_CALL(*mock,
                query_counter_info(
                    ::testing::Field(&rocprofiler_counter_id_t::handle, 100u), _, _))
        .WillRepeatedly([&counters0](rocprofiler_counter_id_t,
                                     rocprofiler_counter_info_version_id_t,
                                     void* info_out) {
            auto* info          = static_cast<rocprofiler_counter_info_v1_t*>(info_out);
            *info               = {};
            info->id            = counters0[0].id;
            info->name          = counters0[0].name;
            info->description   = "";
            info->block         = "";
            info->expression    = "";
            const auto* dim_ptr = &counters0[0].dim_instance;
            info->dimensions_instances_count = 1;
            info->dimensions_instances       = &dim_ptr;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    EXPECT_CALL(*mock,
                query_counter_info(
                    ::testing::Field(&rocprofiler_counter_id_t::handle, 200u), _, _))
        .WillRepeatedly([&counters1](rocprofiler_counter_id_t,
                                     rocprofiler_counter_info_version_id_t,
                                     void* info_out) {
            auto* info          = static_cast<rocprofiler_counter_info_v1_t*>(info_out);
            *info               = {};
            info->id            = counters1[0].id;
            info->name          = counters1[0].name;
            info->description   = "";
            info->block         = "";
            info->expression    = "";
            const auto* dim_ptr = &counters1[0].dim_instance;
            info->dimensions_instances_count = 1;
            info->dimensions_instances       = &dim_ptr;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // create_counter_config for both agents
    EXPECT_CALL(*mock, create_counter_config(_, _, _, _))
        .Times(2)
        .WillRepeatedly([](rocprofiler_agent_id_t, rocprofiler_counter_id_t*, size_t,
                           rocprofiler_counter_config_id_t* config_id) {
            config_id->handle = 300;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // create_context: agent0 gets context 50, agent1 gets context 51
    EXPECT_CALL(*mock, create_context(_))
        .WillOnce([](rocprofiler_context_id_t* ctx) {
            ctx->handle = 50;
            return ROCPROFILER_STATUS_SUCCESS;
        })
        .WillOnce([](rocprofiler_context_id_t* ctx) {
            ctx->handle = 51;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // configure_device_counting_service for both contexts
    EXPECT_CALL(*mock, configure_device_counting_service(_, _, _, _, _))
        .Times(2)
        .WillRepeatedly(Return(ROCPROFILER_STATUS_SUCCESS));

    // start/stop for both contexts
    EXPECT_CALL(*mock, start_context(_))
        .Times(2)
        .WillRepeatedly(Return(ROCPROFILER_STATUS_SUCCESS));
    EXPECT_CALL(*mock, stop_context(_))
        .Times(2)
        .WillRepeatedly(Return(ROCPROFILER_STATUS_SUCCESS));

    // sample: context 50 returns counter 100, context 51 returns counter 200
    EXPECT_CALL(*mock,
                sample_device_counting_service(
                    ::testing::Field(&rocprofiler_context_id_t::handle, 50u), _, _, _, _))
        .WillRepeatedly([](rocprofiler_context_id_t, rocprofiler_user_data_t,
                           rocprofiler_counter_flag_t, rocprofiler_counter_record_t* out,
                           size_t* count) {
            out[0].id            = 100;
            out[0].counter_value = 1000.0;
            *count               = 1;
            return ROCPROFILER_STATUS_SUCCESS;
        });
    EXPECT_CALL(*mock,
                sample_device_counting_service(
                    ::testing::Field(&rocprofiler_context_id_t::handle, 51u), _, _, _, _))
        .WillRepeatedly([](rocprofiler_context_id_t, rocprofiler_user_data_t,
                           rocprofiler_counter_flag_t, rocprofiler_counter_record_t* out,
                           size_t* count) {
            out[0].id            = 200;
            out[0].counter_value = 2000.0;
            *count               = 1;
            return ROCPROFILER_STATUS_SUCCESS;
        });

    // Build
    auto multi_gpu_enabled = enabled_metrics{ {
        counter_definition{ "SQ_WAVES", 0 },
        counter_definition{ "SQ_WAVES", 1 },
    } };

    auto provider = std::make_shared<MockProvider>(
        std::vector<std::shared_ptr<rocprofsys::agent>>{ agent0, agent1 },
        multi_gpu_enabled);
    provider->start();

    test_collector_t collector(provider);
    collector.setup();
    collector.config();

    collector.sample(5000);

    collector.post_process();
    collector.shutdown();

    auto& samples = get_captured_samples();
    ASSERT_EQ(samples.size(), 2U);

    EXPECT_EQ(samples[0].device_id, 0U);
    ASSERT_EQ(samples[0].values.size(), 1U);
    EXPECT_EQ(samples[0].values[0].counter_id, 100U);
    EXPECT_DOUBLE_EQ(samples[0].values[0].value, 1000.0);

    EXPECT_EQ(samples[1].device_id, 1U);
    ASSERT_EQ(samples[1].values.size(), 1U);
    EXPECT_EQ(samples[1].values[0].counter_id, 200U);
    EXPECT_DOUBLE_EQ(samples[1].values[0].value, 2000.0);
}

TEST_F(SdkPmcCollectorWorkflowTest, SampleFailureProducesEmptyMetrics)
{
    auto agent = make_agent(42, 0, "GPU 0");

    std::vector<counter_setup> counters = {
        { rocprofiler_counter_id_t{ 10 }, "SQ_WAVES", {}, {} },
    };

    setup_provider_expectations(mock, 42, counters, /*context=*/100);

    // sample_device_counting_service returns error → device returns empty metrics
    // but device stays (no exception thrown), cache policy skips empty values
    EXPECT_CALL(*mock, sample_device_counting_service(_, _, _, _, _))
        .WillRepeatedly(Return(ROCPROFILER_STATUS_ERROR));

    auto provider = std::make_shared<MockProvider>(
        std::vector<std::shared_ptr<rocprofsys::agent>>{ agent },
        test_settings_policy::get_gpu_perf_counter_enabled_metrics());
    provider->start();

    test_collector_t collector(provider);
    collector.setup();
    collector.config();

    collector.sample(1000);
    collector.sample(2000);

    EXPECT_EQ(get_captured_samples().size(), 0U);
    EXPECT_EQ(collector.get_device_count(), 1U);

    collector.shutdown();
}

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
