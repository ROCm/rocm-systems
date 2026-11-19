// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/base/collector.hpp"
#include "library/pmc/collectors/gpu_perf_counter/gpu_perf_counter_traits.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/device_providers/rocprofiler_sdk/provider.hpp"
#include "mock_backend.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace rocprofsys::pmc::collectors::gpu_perf_counter;
// NOLINTBEGIN(readability-identifier-naming)
using MockBackend     = rocprofsys::backends::rocprofiler_sdk::testing::mock_backend;
using MockBackendImpl = rocprofsys::backends::rocprofiler_sdk::testing::mock_backend_impl;
// NOLINTEND(readability-identifier-naming)
using MockBackendFactory =
    rocprofsys::backends::rocprofiler_sdk::testing::mock_backend_factory;
using MockProvider =
    rocprofsys::pmc::device_providers::rocprofiler_sdk::provider<MockBackendFactory>;

using ::testing::_;
using ::testing::Return;

namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
{

// ============================================================================
// Test stub policies (replace real cache/settings to avoid globals)
// ============================================================================

struct captured_sample
{
    size_t        device_id;
    std::uint64_t timestamp;
    metrics       values;
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
                             std::uint64_t timestamp)
    {
        if(values.empty())
        {
            return;
        }
        get_captured_samples().push_back(
            { .device_id = device_id, .timestamp = timestamp, .values = values });
    }
};

struct test_settings_policy
{
    static enabled_metrics get_gpu_perf_counter_enabled_metrics()
    {
        return enabled_metrics{
            { counter_definition{ .name = "SQ_WAVES", .device_index = 0 },
              counter_definition{ .name = "SQ_BUSY_CYCLES", .device_index = 0 } }
        };
    }
};

struct test_config
{
    using SettingsApi = test_settings_policy;
    using CacheApi    = test_cache_policy;
};

using test_collector_t =
    base::collector<gpu_perf_counter_traits<MockProvider>, MockProvider, test_config>;

// ============================================================================
// Helper: build a fake agent
// ============================================================================

static std::shared_ptr<rocprofsys::agent>
make_agent(std::uint64_t handle, size_t device_type_index, const std::string& name)
{
    auto a               = std::make_shared<rocprofsys::agent>();
    a->type              = agent_type::gpu;
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
    MockBackend::counter_id_t id;
    const char*               name;
};

static void
setup_provider_expectations(const std::shared_ptr<MockBackendImpl>& mock,
                            std::uint64_t                           agent_handle,
                            std::vector<counter_setup>&             counters,
                            std::uint32_t                           context_handle_out)
{
    EXPECT_CALL(
        *mock,
        iterate_agent_supported_counters(
            ::testing::Field(&MockBackend::agent_id_t::handle, agent_handle), _, _))
        .WillOnce([&counters](MockBackend::agent_id_t,
                              MockBackend::available_counters_cb_t cb, void* user_data) {
            std::vector<MockBackend::counter_id_t> ids;
            ids.reserve(counters.size());
            for(auto const& c : counters)
            {
                ids.push_back(c.id);
            }
            cb({}, ids.data(), ids.size(), user_data);
            return MockBackend::status_success;
        });

    for(auto& c : counters)
    {
        EXPECT_CALL(*mock, query_counter_details(::testing::Field(
                               &MockBackend::counter_id_t::handle, c.id.handle)))
            .WillRepeatedly([&c](MockBackend::counter_id_t) {
                return std::vector<counter_metadata>{ { .counter_id  = c.id.handle,
                                                        .name        = c.name,
                                                        .description = "",
                                                        .block       = "",
                                                        .expression  = "",
                                                        .is_constant = false,
                                                        .is_derived  = false,
                                                        .dimensions  = {} } };
            });
    }

    EXPECT_CALL(
        *mock,
        create_counter_config(
            ::testing::Field(&MockBackend::agent_id_t::handle, agent_handle), _, _, _))
        .WillOnce([](MockBackend::agent_id_t, MockBackend::counter_id_t*, size_t,
                     MockBackend::counter_config_id_t* config_id) {
            config_id->handle = 200;
            return MockBackend::status_success;
        });

    EXPECT_CALL(*mock, create_context(_))
        .WillOnce([context_handle_out](MockBackend::context_id_t* ctx) {
            ctx->handle = context_handle_out;
        });

    EXPECT_CALL(*mock, configure_device_counting_service(
                           ::testing::Field(&MockBackend::context_id_t::handle,
                                            context_handle_out),
                           _, _, _, _))
        .WillOnce(Return(MockBackend::status_success));

    EXPECT_CALL(*mock, start_context(::testing::Field(&MockBackend::context_id_t::handle,
                                                      context_handle_out)))
        .WillOnce(Return());

    EXPECT_CALL(*mock, stop_context(::testing::Field(&MockBackend::context_id_t::handle,
                                                     context_handle_out)))
        .WillOnce(Return());
}

// ============================================================================
// Fixture
// ============================================================================

class SdkPmcCollectorWorkflowTest : public ::testing::Test
{
protected:
    std::shared_ptr<MockBackendImpl> m_mock;

    void SetUp() override
    {
        get_captured_samples().clear();
        m_mock = std::make_shared<MockBackendImpl>();
        MockBackendFactory::set_mock(m_mock);

        ON_CALL(*m_mock, query_record_counter_id(_, _))
            .WillByDefault([](MockBackend::counter_record_t record,
                              MockBackend::counter_id_t*    counter_id) {
                counter_id->handle = record.id;
                return MockBackend::status_success;
            });
        EXPECT_CALL(*m_mock, query_record_counter_id(_, _)).Times(::testing::AnyNumber());
    }

    void TearDown() override
    {
        MockBackendFactory::reset();
        get_captured_samples().clear();
    }
};

// ============================================================================
// Tests
// ============================================================================

TEST_F(SdkPmcCollectorWorkflowTest, SingleGpuWorkflow)
{
    constexpr std::uint64_t k_agent_handle   = 42;
    constexpr std::uint32_t k_context_handle = 100;

    auto const agent = make_agent(k_agent_handle, 0, "GPU 0");

    std::vector<counter_setup> counters = {
        { .id = MockBackend::counter_id_t{ 10 }, .name = "SQ_WAVES" },
        { .id = MockBackend::counter_id_t{ 20 }, .name = "SQ_BUSY_CYCLES" },
    };

    setup_provider_expectations(m_mock, k_agent_handle, counters, k_context_handle);

    auto const sample_call_count = std::make_shared<int>(0);
    EXPECT_CALL(*m_mock, sample_device_counting_service(
                             ::testing::Field(&MockBackend::context_id_t::handle, 100u),
                             _, _, _, _))
        .WillRepeatedly(
            [sample_call_count](MockBackend::context_id_t, MockBackend::user_data_t,
                                MockBackend::counter_flag_t,
                                MockBackend::counter_record_t* out, size_t* count) {
                ++(*sample_call_count);
                out[0].id            = 10;
                out[0].counter_value = 42.0 * (*sample_call_count);
                out[1].id            = 20;
                out[1].counter_value = 99.0 * (*sample_call_count);
                *count               = 2;
                return MockBackend::status_success;
            });

    auto const provider = std::make_shared<MockProvider>(
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
    auto const agent0 = make_agent(10, 0, "GPU 0");
    auto const agent1 = make_agent(11, 1, "GPU 1");

    std::vector<counter_setup> const counters0 = {
        { .id = MockBackend::counter_id_t{ 100 }, .name = "SQ_WAVES" },
    };
    std::vector<counter_setup> const counters1 = {
        { .id = MockBackend::counter_id_t{ 200 }, .name = "SQ_WAVES" },
    };

    // Agent 0
    EXPECT_CALL(*m_mock,
                iterate_agent_supported_counters(
                    ::testing::Field(&MockBackend::agent_id_t::handle, 10u), _, _))
        .WillOnce([&](MockBackend::agent_id_t, MockBackend::available_counters_cb_t cb,
                      void* ud) {
            MockBackend::counter_id_t id = { 100 };
            cb({}, &id, 1, ud);
            return MockBackend::status_success;
        });

    // Agent 1
    EXPECT_CALL(*m_mock,
                iterate_agent_supported_counters(
                    ::testing::Field(&MockBackend::agent_id_t::handle, 11u), _, _))
        .WillOnce([&](MockBackend::agent_id_t, MockBackend::available_counters_cb_t cb,
                      void* ud) {
            MockBackend::counter_id_t id = { 200 };
            cb({}, &id, 1, ud);
            return MockBackend::status_success;
        });

    EXPECT_CALL(*m_mock, query_counter_details(
                             ::testing::Field(&MockBackend::counter_id_t::handle, 100u)))
        .WillRepeatedly([](MockBackend::counter_id_t) {
            return std::vector<counter_metadata>{ { .counter_id  = 100,
                                                    .name        = "SQ_WAVES",
                                                    .description = "",
                                                    .block       = "",
                                                    .expression  = "",
                                                    .is_constant = false,
                                                    .is_derived  = false,
                                                    .dimensions  = {} } };
        });

    EXPECT_CALL(*m_mock, query_counter_details(
                             ::testing::Field(&MockBackend::counter_id_t::handle, 200u)))
        .WillRepeatedly([](MockBackend::counter_id_t) {
            return std::vector<counter_metadata>{ { .counter_id  = 200,
                                                    .name        = "SQ_WAVES",
                                                    .description = "",
                                                    .block       = "",
                                                    .expression  = "",
                                                    .is_constant = false,
                                                    .is_derived  = false,
                                                    .dimensions  = {} } };
        });

    EXPECT_CALL(*m_mock, create_counter_config(_, _, _, _))
        .Times(2)
        .WillRepeatedly([](MockBackend::agent_id_t, MockBackend::counter_id_t*, size_t,
                           MockBackend::counter_config_id_t* config_id) {
            config_id->handle = 300;
            return MockBackend::status_success;
        });

    constexpr std::uint64_t k_context_handle_agent0 = 50;
    constexpr std::uint64_t k_context_handle_agent1 = 51;
    EXPECT_CALL(*m_mock, create_context(_))
        .WillOnce([=](MockBackend::context_id_t* ctx) {
            ctx->handle = k_context_handle_agent0;
        })
        .WillOnce([=](MockBackend::context_id_t* ctx) {
            ctx->handle = k_context_handle_agent1;
        });

    EXPECT_CALL(*m_mock, configure_device_counting_service(_, _, _, _, _))
        .Times(2)
        .WillRepeatedly(Return(MockBackend::status_success));

    EXPECT_CALL(*m_mock, start_context(_)).Times(2).WillRepeatedly(Return());
    EXPECT_CALL(*m_mock, stop_context(_)).Times(2).WillRepeatedly(Return());

    EXPECT_CALL(*m_mock, sample_device_counting_service(
                             ::testing::Field(&MockBackend::context_id_t::handle,
                                              k_context_handle_agent0),
                             _, _, _, _))
        .WillRepeatedly([](MockBackend::context_id_t, MockBackend::user_data_t,
                           MockBackend::counter_flag_t,
                           MockBackend::counter_record_t* out, size_t* count) {
            out[0].id            = 100;
            out[0].counter_value = 1000.0;
            *count               = 1;
            return MockBackend::status_success;
        });
    EXPECT_CALL(*m_mock, sample_device_counting_service(
                             ::testing::Field(&MockBackend::context_id_t::handle,
                                              k_context_handle_agent1),
                             _, _, _, _))
        .WillRepeatedly([](MockBackend::context_id_t, MockBackend::user_data_t,
                           MockBackend::counter_flag_t,
                           MockBackend::counter_record_t* out, size_t* count) {
            out[0].id            = 200;
            out[0].counter_value = 2000.0;
            *count               = 1;
            return MockBackend::status_success;
        });

    auto const multi_gpu_enabled = enabled_metrics{ {
        counter_definition{ .name = "SQ_WAVES", .device_index = 0 },
        counter_definition{ .name = "SQ_WAVES", .device_index = 1 },
    } };

    auto const provider = std::make_shared<MockProvider>(
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
    constexpr std::uint64_t k_agent_handle   = 42;
    constexpr std::uint32_t k_context_handle = 100;

    auto const agent = make_agent(k_agent_handle, 0, "GPU 0");

    std::vector<counter_setup> counters = {
        { .id = MockBackend::counter_id_t{ 10 }, .name = "SQ_WAVES" },
    };

    setup_provider_expectations(m_mock, k_agent_handle, counters, k_context_handle);

    EXPECT_CALL(*m_mock, sample_device_counting_service(_, _, _, _, _))
        .WillRepeatedly(Return(MockBackend::status_error));

    auto const provider = std::make_shared<MockProvider>(
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

// ============================================================================
// Derived counters
// ============================================================================

namespace
{
constexpr std::uint64_t k_derived_agent_handle   = 7;
constexpr std::uint32_t k_derived_context_handle = 300;

constexpr std::uint64_t k_busy_id        = 1;
constexpr std::uint64_t k_active_id      = 2;
constexpr std::uint64_t k_simd_count_id  = 3;
constexpr std::uint64_t k_simd_num_id    = 4;
constexpr std::uint64_t k_mfma_util_id   = 5;
constexpr std::uint64_t k_waves_id       = 6;
constexpr std::uint64_t k_waves_sum_id   = 7;
constexpr std::uint64_t k_lds_level_id   = 8;
constexpr std::uint64_t k_lds_insts_id   = 9;
constexpr std::uint64_t k_lds_latency_id = 10;

counter_metadata
catalog_counter(std::uint64_t counter_id, std::string name, std::string expression = {})
{
    auto meta       = counter_metadata{};
    meta.counter_id = counter_id;
    meta.name       = std::move(name);
    meta.is_derived = !expression.empty();
    meta.expression = std::move(expression);
    return meta;
}

// Counters and definitions as rocprofiler-sdk reports them for gfx942.
std::vector<counter_metadata>
sdk_catalog()
{
    auto simd_count        = catalog_counter(k_simd_count_id, "simd_count");
    simd_count.is_constant = true;
    return {
        catalog_counter(k_busy_id, "SQ_VALU_MFMA_BUSY_CYCLES"),
        catalog_counter(k_active_id, "GRBM_GUI_ACTIVE"),
        simd_count,
        catalog_counter(k_simd_num_id, "SIMD_NUM", "simd_count"),
        catalog_counter(k_mfma_util_id, "MfmaUtil",
                        "reduce(SQ_VALU_MFMA_BUSY_CYCLES,sum)/"
                        "(reduce(GRBM_GUI_ACTIVE,max)*SIMD_NUM)*100"),
        catalog_counter(k_waves_id, "SQ_WAVES"),
        catalog_counter(k_waves_sum_id, "SQ_WAVES_sum", "reduce(SQ_WAVES,sum)"),
        catalog_counter(k_lds_level_id, "SQ_INST_LEVEL_LDS"),
        catalog_counter(k_lds_insts_id, "SQ_INSTS_LDS"),
        catalog_counter(k_lds_latency_id, "LdsLatency",
                        "reduce(accumulate(SQ_INST_LEVEL_LDS, HIGH_RES),sum)/"
                        "reduce(SQ_INSTS_LDS,sum)"),
    };
}

enabled_metrics
enable_on_device_0(std::initializer_list<const char*> names)
{
    auto definitions = std::vector<counter_definition>{};
    for(const char* name : names)
    {
        definitions.push_back({ .name = name, .device_index = 0 });
    }
    return enabled_metrics{ std::move(definitions) };
}

// Expects provider construction for one agent that supports sdk_catalog(). Returns the
// counter ids the provider passes to create_counter_config.
std::shared_ptr<std::vector<std::uint64_t>>
expect_catalog_provider(MockBackendImpl& mock)
{
    auto const catalog = std::make_shared<std::vector<counter_metadata>>(sdk_catalog());

    EXPECT_CALL(mock, iterate_agent_supported_counters(_, _, _))
        .WillOnce([catalog](MockBackend::agent_id_t,
                            MockBackend::available_counters_cb_t callback,
                            void*                                user_data) {
            auto ids = std::vector<MockBackend::counter_id_t>{};
            for(const auto& meta : *catalog)
            {
                ids.push_back({ meta.counter_id });
            }
            callback({}, ids.data(), ids.size(), user_data);
            return MockBackend::status_success;
        });

    EXPECT_CALL(mock, query_counter_details(_))
        .WillRepeatedly([catalog](MockBackend::counter_id_t counter_id) {
            auto const itr = std::ranges::find(*catalog, counter_id.handle,
                                               &counter_metadata::counter_id);
            return std::vector<counter_metadata>{ *itr };
        });

    auto config_ids = std::make_shared<std::vector<std::uint64_t>>();
    EXPECT_CALL(mock, create_counter_config(_, _, _, _))
        .WillOnce([config_ids](MockBackend::agent_id_t, MockBackend::counter_id_t* ids,
                               size_t count, MockBackend::counter_config_id_t* config) {
            for(const auto& counter_id : std::span{ ids, count })
            {
                config_ids->push_back(counter_id.handle);
            }
            config->handle = 1;
            return MockBackend::status_success;
        });

    EXPECT_CALL(mock, create_context(_)).WillOnce([](MockBackend::context_id_t* ctx) {
        ctx->handle = k_derived_context_handle;
    });
    EXPECT_CALL(mock, configure_device_counting_service(_, _, _, _, _))
        .WillOnce(Return(MockBackend::status_success));
    return config_ids;
}

[[nodiscard]] std::vector<std::string>
reported_counter_names(MockProvider& provider)
{
    using device_t = MockProvider::device_t;
    auto names     = std::vector<std::string>{};
    for(const auto& device : provider.get_devices<device_t>(device_type::gpu))
    {
        for(const auto& meta : device->get_counter_metadata())
        {
            names.push_back(meta.name);
        }
    }
    return names;
}
}  // namespace

// Totals since start_context under a steady 87 % MFMA load after a ramp: the SDK's
// MfmaUtil climbs 43.5 -> 65.25 -> 72.5 while the interval value is 43.5, 87, 87.
constexpr std::array<double, 3>       k_busy_cumulative    = { 1740.0, 5220.0, 8700.0 };
constexpr std::array<double, 3>       k_active_cumulative  = { 1000.0, 2000.0, 3000.0 };
constexpr std::array<double, 3>       k_mfma_util_interval = { 43.5, 87.0, 87.0 };
constexpr std::array<std::int64_t, 3> k_mfma_timestamps    = { 1000, 2000, 3000 };
constexpr double                      k_simd_count         = 4.0;

void
expect_mfma_util_input_samples(MockBackendImpl& mock)
{
    auto sample_index = std::make_shared<size_t>(0);
    EXPECT_CALL(mock, sample_device_counting_service(_, _, _, _, _))
        .Times(static_cast<int>(k_mfma_timestamps.size()))
        .WillRepeatedly([sample_index](
                            MockBackend::context_id_t, MockBackend::user_data_t,
                            MockBackend::counter_flag_t,
                            MockBackend::counter_record_t* out, size_t* count) {
            auto const index = *sample_index;
            ++*sample_index;
            out[0] = { .id = k_busy_id, .counter_value = k_busy_cumulative.at(index) };
            out[1] = { .id            = k_active_id,
                       .counter_value = k_active_cumulative.at(index) };
            out[2] = { .id = k_simd_count_id, .counter_value = k_simd_count };
            *count = 3;
            return MockBackend::status_success;
        });
}

void
expect_interval_mfma_util(const std::vector<captured_sample>& samples)
{
    ASSERT_EQ(samples.size(), k_mfma_timestamps.size());
    for(size_t i = 0; i < samples.size(); ++i)
    {
        SCOPED_TRACE(i);
        ASSERT_EQ(samples[i].values.size(), 1U);
        EXPECT_EQ(samples[i].values[0].counter_id, k_mfma_util_id);
        EXPECT_DOUBLE_EQ(samples[i].values[0].value, k_mfma_util_interval.at(i));
    }
}

// MfmaUtil is a ratio of totals since start_context, so the provider collects its
// inputs and the device evaluates it on their interval values (ISS-116).
TEST_F(SdkPmcCollectorWorkflowTest, DerivedRatioCounterIsEvaluatedFromItsInputs)
{
    auto const config_ids = expect_catalog_provider(*m_mock);
    EXPECT_CALL(*m_mock, start_context(_)).WillOnce(Return());
    EXPECT_CALL(*m_mock, stop_context(_)).WillOnce(Return());
    expect_mfma_util_input_samples(*m_mock);

    auto const provider = std::make_shared<MockProvider>(
        std::vector<std::shared_ptr<rocprofsys::agent>>{
            make_agent(k_derived_agent_handle, 0, "GPU 0") },
        enable_on_device_0({ "MfmaUtil" }));

    EXPECT_THAT(*config_ids,
                ::testing::UnorderedElementsAre(k_busy_id, k_active_id, k_simd_count_id));
    EXPECT_THAT(reported_counter_names(*provider), ::testing::ElementsAre("MfmaUtil"));

    provider->start();
    test_collector_t collector(provider);
    collector.setup();
    collector.config();
    std::ranges::for_each(k_mfma_timestamps, [&collector](std::int64_t timestamp) {
        collector.sample(timestamp);
    });
    collector.shutdown();

    expect_interval_mfma_util(get_captured_samples());
}

// A constant or linear derived counter is exact from the SDK, so it is collected itself.
TEST_F(SdkPmcCollectorWorkflowTest,
       ConstantAndLinearDerivedCountersAreCollectedThemselves)
{
    auto const config_ids = expect_catalog_provider(*m_mock);

    auto const provider = std::make_shared<MockProvider>(
        std::vector<std::shared_ptr<rocprofsys::agent>>{
            make_agent(k_derived_agent_handle, 0, "GPU 0") },
        enable_on_device_0({ "SIMD_NUM", "SQ_WAVES_sum" }));

    EXPECT_THAT(*config_ids,
                ::testing::UnorderedElementsAre(k_simd_num_id, k_waves_sum_id));
    EXPECT_THAT(reported_counter_names(*provider),
                ::testing::UnorderedElementsAre("SIMD_NUM", "SQ_WAVES_sum"));
}

// LdsLatency uses accumulate(), which device-wide sampling cannot evaluate; it is
// rejected and the other requested counters are still collected.
TEST_F(SdkPmcCollectorWorkflowTest, UnsupportedDerivedCounterIsNotCollected)
{
    auto const config_ids = expect_catalog_provider(*m_mock);

    auto const provider = std::make_shared<MockProvider>(
        std::vector<std::shared_ptr<rocprofsys::agent>>{
            make_agent(k_derived_agent_handle, 0, "GPU 0") },
        enable_on_device_0({ "LdsLatency", "SQ_WAVES" }));

    EXPECT_THAT(*config_ids, ::testing::ElementsAre(k_waves_id));
    EXPECT_THAT(reported_counter_names(*provider), ::testing::ElementsAre("SQ_WAVES"));
}

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
