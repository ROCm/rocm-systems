// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/gpu_perf_counter/counter_definitions.hpp"
#include "library/pmc/collectors/gpu_perf_counter/derived_expression.hpp"
#include "library/pmc/collectors/gpu_perf_counter/device.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "mock_backend.hpp"
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace rocprofsys::pmc::collectors::gpu_perf_counter;
using ::testing::_;
using ::testing::Return;
using ::testing::StrictMock;

// NOLINTBEGIN(readability-identifier-naming)
using MockBackendImpl = ::testing::StrictMock<
    rocprofsys::backends::rocprofiler_sdk::testing::mock_backend_impl>;
using MockBackend = rocprofsys::backends::rocprofiler_sdk::testing::mock_backend;
// NOLINTEND(readability-identifier-naming)

namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
{

class SdkPmcDeviceTest : public ::testing::Test
{
protected:
    // `m_mock_impl` is the GMock object EXPECT_CALL binds to. `mock_backend` is the
    // stateless handle passed as device<Backend>'s Backend argument; its static
    // methods forward to whichever impl is bound via MockBackend::bind().
    std::shared_ptr<MockBackendImpl>   m_mock_impl;
    std::shared_ptr<MockBackend>       mock_backend;
    std::shared_ptr<rocprofsys::agent> test_agent;
    MockBackend::context_id_t          test_context{};

    void SetUp() override
    {
        m_mock_impl = std::make_shared<MockBackendImpl>();
        MockBackend::bind(m_mock_impl);
        mock_backend        = std::make_shared<MockBackend>();
        test_context.handle = 1;

        test_agent                    = std::make_shared<rocprofsys::agent>();
        test_agent->type              = agent_type::gpu;
        test_agent->handle            = 42;
        test_agent->device_id         = 0;
        test_agent->device_type_index = 0;
        test_agent->name              = "GPU 0";
        test_agent->product_name      = "GPU 0";
        test_agent->vendor_name       = "AMD";
    }

    void TearDown() override { MockBackend::unbind(); }
};

TEST_F(SdkPmcDeviceTest, DeviceProperties)
{
    auto meta = std::vector<counter_metadata>{
        counter_metadata{ .counter_id  = 10,
                          .name        = "SQ_WAVES",
                          .description = "",
                          .block       = "",
                          .expression  = "",
                          .is_constant = false,
                          .is_derived  = false,
                          .dimensions  = {} },
    };

    const device<MockBackend> dev(mock_backend, test_context, test_agent,
                                  { .requested = std::move(meta) });

    EXPECT_EQ(dev.get_index(), 0U);
    EXPECT_EQ(dev.get_name(), "GPU 0");
    EXPECT_EQ(dev.get_vendor_name(), "AMD");
    EXPECT_TRUE(dev.is_supported());
}

TEST_F(SdkPmcDeviceTest, EmptyDeviceNotSupported)
{
    const device<MockBackend> dev(mock_backend, test_context, test_agent,
                                  device_counters{});

    EXPECT_FALSE(dev.is_supported());
}

TEST_F(SdkPmcDeviceTest, DeviceWithIndex3)
{
    auto const agent3         = std::make_shared<rocprofsys::agent>();
    agent3->device_type_index = 3;
    agent3->name              = "GPU 3";
    agent3->product_name      = "GPU 3";
    agent3->vendor_name       = "AMD";

    const device<MockBackend> dev(mock_backend, test_context, agent3, device_counters{});

    EXPECT_EQ(dev.get_index(), 3U);
    EXPECT_EQ(dev.get_name(), "GPU 3");
    EXPECT_EQ(dev.get_product_name(), "GPU 3");
}

TEST_F(SdkPmcDeviceTest, SampleWithScalarCounters)
{
    auto meta = std::vector<counter_metadata>{
        counter_metadata{ .counter_id  = 10,
                          .name        = "SQ_WAVES",
                          .description = "",
                          .block       = "",
                          .expression  = "",
                          .is_constant = false,
                          .is_derived  = false,
                          .dimensions  = {} },
        counter_metadata{ .counter_id  = 20,
                          .name        = "SQ_INSTS_VALU",
                          .description = "",
                          .block       = "",
                          .expression  = "",
                          .is_constant = false,
                          .is_derived  = false,
                          .dimensions  = {} },
    };

    device<MockBackend> dev(mock_backend, test_context, test_agent,
                            { .requested = std::move(meta) });

    MockBackend::counter_record_t records[2];
    records[0].id            = 10;
    records[0].counter_value = 42.0;
    records[1].id            = 20;
    records[1].counter_value = 100.0;

    EXPECT_CALL(*m_mock_impl, start_context(_)).WillOnce(Return());

    EXPECT_CALL(*m_mock_impl, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([&](MockBackend::context_id_t, MockBackend::user_data_t,
                      MockBackend::counter_flag_t, MockBackend::counter_record_t* out,
                      size_t* count) {
            out[0] = records[0];
            out[1] = records[1];
            *count = 2;
            return MockBackend::status_success;
        });

    EXPECT_CALL(*m_mock_impl, query_record_counter_id(_, _))
        .WillOnce(
            [](MockBackend::counter_record_t record, MockBackend::counter_id_t* out) {
                out->handle = 10;
                return MockBackend::status_success;
            })
        .WillOnce(
            [](MockBackend::counter_record_t record, MockBackend::counter_id_t* out) {
                out->handle = 20;
                return MockBackend::status_success;
            });

    const enabled_metrics enabled{ {} };
    auto                  result = dev.sample_metrics(enabled, 1000000);

    ASSERT_EQ(result.size(), 2U);
    EXPECT_EQ(result[0].counter_id, 10U);
    EXPECT_DOUBLE_EQ(result[0].value, 42.0);
    EXPECT_EQ(result[1].counter_id, 20U);
    EXPECT_DOUBLE_EQ(result[1].value, 100.0);
}

TEST_F(SdkPmcDeviceTest, SampleWithMultiDimCounters)
{
    auto meta = std::vector<counter_metadata>{
        counter_metadata{ .counter_id  = 100,
                          .name        = "SQC_ICACHE_HITS",
                          .description = "",
                          .block       = "",
                          .expression  = "",
                          .is_constant = false,
                          .is_derived  = false,
                          .dimensions  = { { .name = "WGP", .position = 0 },
                                           { .name = "SA", .position = 0 },
                                           { .name = "SE", .position = 0 } } },
        counter_metadata{ .counter_id  = 101,
                          .name        = "SQC_ICACHE_HITS",
                          .description = "",
                          .block       = "",
                          .expression  = "",
                          .is_constant = false,
                          .is_derived  = false,
                          .dimensions  = { { .name = "WGP", .position = 1 },
                                           { .name = "SA", .position = 0 },
                                           { .name = "SE", .position = 0 } } },
        counter_metadata{ .counter_id  = 102,
                          .name        = "SQC_ICACHE_HITS",
                          .description = "",
                          .block       = "",
                          .expression  = "",
                          .is_constant = false,
                          .is_derived  = false,
                          .dimensions  = { { .name = "WGP", .position = 2 },
                                           { .name = "SA", .position = 0 },
                                           { .name = "SE", .position = 0 } } },
        counter_metadata{ .counter_id  = 103,
                          .name        = "SQC_ICACHE_HITS",
                          .description = "",
                          .block       = "",
                          .expression  = "",
                          .is_constant = false,
                          .is_derived  = false,
                          .dimensions  = { { .name = "WGP", .position = 3 },
                                           { .name = "SA", .position = 0 },
                                           { .name = "SE", .position = 0 } } },
    };

    device<MockBackend> dev(mock_backend, test_context, test_agent,
                            { .requested = std::move(meta) });

    MockBackend::counter_record_t records[4];
    for(int i = 0; i < 4; ++i)
    {
        records[i].id            = static_cast<std::uint64_t>(100 + i);
        records[i].counter_value = static_cast<double>(10 * (i + 1));
    }

    EXPECT_CALL(*m_mock_impl, start_context(_)).WillOnce(Return());

    EXPECT_CALL(*m_mock_impl, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([&](MockBackend::context_id_t, MockBackend::user_data_t,
                      MockBackend::counter_flag_t, MockBackend::counter_record_t* out,
                      size_t* count) {
            for(int i = 0; i < 4; ++i)
            {
                out[i] = records[i];
            }
            *count = 4;
            return MockBackend::status_success;
        });

    EXPECT_CALL(*m_mock_impl, query_record_counter_id(_, _))
        .Times(4)
        .WillRepeatedly(
            [](MockBackend::counter_record_t record, MockBackend::counter_id_t* out) {
                out->handle = record.id;
                return MockBackend::status_success;
            });

    const enabled_metrics enabled{ {} };
    auto                  result = dev.sample_metrics(enabled, 1000000);

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
// counter handle via query_record_counter_id.
TEST_F(SdkPmcDeviceTest, CounterIdDecodedFromInstanceId)
{
    constexpr std::uint64_t plain_counter_handle = 7;
    constexpr std::uint64_t sdk_instance_id      = 0xDEAD0007ULL;

    auto meta = std::vector<counter_metadata>{
        counter_metadata{ .counter_id  = plain_counter_handle,
                          .name        = "SQ_WAVES",
                          .description = "",
                          .block       = "",
                          .expression  = "",
                          .is_constant = false,
                          .is_derived  = false,
                          .dimensions  = {} },
    };

    device<MockBackend> dev(mock_backend, test_context, test_agent,
                            { .requested = std::move(meta) });

    MockBackend::counter_record_t record{};
    record.id            = sdk_instance_id;
    record.counter_value = 99.0;

    EXPECT_CALL(*m_mock_impl, start_context(_)).WillOnce(Return());

    EXPECT_CALL(*m_mock_impl, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([&](MockBackend::context_id_t, MockBackend::user_data_t,
                      MockBackend::counter_flag_t, MockBackend::counter_record_t* out,
                      size_t* count) {
            out[0] = record;
            *count = 1;
            return MockBackend::status_success;
        });

    EXPECT_CALL(*m_mock_impl, query_record_counter_id(_, _))
        .WillOnce([](MockBackend::counter_record_t, MockBackend::counter_id_t* out) {
            out->handle = plain_counter_handle;
            return MockBackend::status_success;
        });

    const enabled_metrics enabled{ {} };
    auto                  result = dev.sample_metrics(enabled, 1000000);

    ASSERT_EQ(result.size(), 1U);
    EXPECT_EQ(result[0].counter_id, plain_counter_handle);
    EXPECT_DOUBLE_EQ(result[0].value, 99.0);
}

// Verify that calling sample_metrics repeatedly does not grow
// heap unboundedly — the same m_result_cache vector is reused across samples.
TEST_F(SdkPmcDeviceTest, ResultCacheReusedAcrossSamples)
{
    auto meta = std::vector<counter_metadata>{
        counter_metadata{ .counter_id  = 5,
                          .name        = "SQ_WAVES",
                          .description = "",
                          .block       = "",
                          .expression  = "",
                          .is_constant = false,
                          .is_derived  = false,
                          .dimensions  = {} },
    };

    device<MockBackend> dev(mock_backend, test_context, test_agent,
                            { .requested = std::move(meta) });

    MockBackend::counter_record_t record{};
    record.id            = 5;
    record.counter_value = 1.0;

    EXPECT_CALL(*m_mock_impl, start_context(_)).WillOnce(Return());

    // Two successive sample calls; each must return correct data.
    EXPECT_CALL(*m_mock_impl, sample_device_counting_service(_, _, _, _, _))
        .Times(2)
        .WillRepeatedly([&](MockBackend::context_id_t, MockBackend::user_data_t,
                            MockBackend::counter_flag_t,
                            MockBackend::counter_record_t* out, size_t* count) {
            out[0] = record;
            *count = 1;
            return MockBackend::status_success;
        });

    EXPECT_CALL(*m_mock_impl, query_record_counter_id(_, _))
        .Times(2)
        .WillRepeatedly(
            [](MockBackend::counter_record_t, MockBackend::counter_id_t* out) {
                out->handle = 5;
                return MockBackend::status_success;
            });

    const enabled_metrics enabled{ {} };

    auto result1 = dev.sample_metrics(enabled, 1000000);
    ASSERT_EQ(result1.size(), 1U);
    EXPECT_EQ(result1[0].counter_id, 5U);

    auto result2 = dev.sample_metrics(enabled, 2000000);
    ASSERT_EQ(result2.size(), 1U);
    EXPECT_EQ(result2[0].counter_id, 5U);
}

TEST_F(SdkPmcDeviceTest, SampleFailureReturnsEmpty)
{
    device<MockBackend> dev(mock_backend, test_context, test_agent, device_counters{});

    EXPECT_CALL(*m_mock_impl, start_context(_)).WillOnce(Return());

    EXPECT_CALL(*m_mock_impl, sample_device_counting_service(_, _, _, _, _))
        .WillOnce(Return(MockBackend::status_error));

    const enabled_metrics enabled{ {} };
    auto const            result = dev.sample_metrics(enabled, 1000000);

    EXPECT_TRUE(result.empty());
}

TEST_F(SdkPmcDeviceTest, SampleWithZeroRecords)
{
    device<MockBackend> dev(mock_backend, test_context, test_agent, device_counters{});

    EXPECT_CALL(*m_mock_impl, start_context(_)).WillOnce(Return());

    EXPECT_CALL(*m_mock_impl, sample_device_counting_service(_, _, _, _, _))
        .WillOnce([](MockBackend::context_id_t, MockBackend::user_data_t,
                     MockBackend::counter_flag_t, MockBackend::counter_record_t*,
                     size_t* count) {
            *count = 0;
            return MockBackend::status_success;
        });

    const enabled_metrics enabled{ {} };
    auto const            result = dev.sample_metrics(enabled, 1000000);

    EXPECT_TRUE(result.empty());
}

namespace
{
[[nodiscard]] std::optional<double>
find_value(const metrics& result, counter_id_t counter_id)
{
    auto const itr = std::ranges::find(result, counter_id, &counter_value::counter_id);
    if(itr == result.end())
    {
        return std::nullopt;
    }
    return itr->value;
}

void
expect_value(const metrics& result, counter_id_t counter_id, double expected)
{
    auto const value = find_value(result, counter_id);
    ASSERT_TRUE(value.has_value()) << "counter " << counter_id;
    EXPECT_DOUBLE_EQ(*value, expected) << "counter " << counter_id;
}

counter_metadata
raw_counter(counter_id_t counter_id, std::string name)
{
    auto meta       = counter_metadata{};
    meta.counter_id = counter_id;
    meta.name       = std::move(name);
    return meta;
}

counter_metadata
constant_counter(counter_id_t counter_id, std::string name)
{
    auto meta        = raw_counter(counter_id, std::move(name));
    meta.is_constant = true;
    return meta;
}

counter_metadata
derived_counter(counter_id_t counter_id, std::string name, std::string expression)
{
    auto meta       = raw_counter(counter_id, std::move(name));
    meta.expression = std::move(expression);
    meta.is_derived = true;
    return meta;
}

// Resolves counters the way the device provider does: parses every requested derived
// counter. @p definitions are counters its expression names that are not collected.
device_counters
resolve_counters(std::vector<counter_metadata> requested,
                 std::vector<counter_metadata> inputs,
                 std::vector<counter_metadata> definitions)
{
    definitions.insert(definitions.end(), requested.begin(), requested.end());
    definitions.insert(definitions.end(), inputs.begin(), inputs.end());
    auto const names = counter_definitions{ definitions };

    auto counters = device_counters{};
    for(const auto& meta : requested)
    {
        if(meta.is_derived)
        {
            counters.expressions.insert_or_assign(
                meta.name,
                expression_parser::parse(meta.expression, names, counters.slots));
        }
    }
    counters.requested = std::move(requested);
    counters.inputs    = std::move(inputs);
    return counters;
}

MockBackend::counter_record_t
record_of(counter_id_t counter_id, double value)
{
    auto record          = MockBackend::counter_record_t{};
    record.id            = counter_id;
    record.counter_value = value;
    return record;
}

// The SDK device counting service returns counters as totals since start_context.
// This feeds one set of totals per sample and identifies each record by its own id.
constexpr const char* k_mfma_util = "reduce(SQ_VALU_MFMA_BUSY_CYCLES,sum)/"
                                    "(reduce(GRBM_GUI_ACTIVE,max)*SIMD_NUM)*100";

void
expect_cumulative_samples(MockBackendImpl& mock_impl,
                          std::vector<std::vector<MockBackend::counter_record_t>> samples)
{
    EXPECT_CALL(mock_impl, start_context(_)).WillOnce(Return());

    auto sample_index = std::make_shared<size_t>(0);
    EXPECT_CALL(mock_impl, sample_device_counting_service(_, _, _, _, _))
        .Times(static_cast<int>(samples.size()))
        .WillRepeatedly([samples = std::move(samples), sample_index](
                            MockBackend::context_id_t, MockBackend::user_data_t,
                            MockBackend::counter_flag_t,
                            MockBackend::counter_record_t* out, size_t* count) {
            const auto& records = samples.at(*sample_index);
            ++*sample_index;
            std::ranges::copy(records, out);
            *count = records.size();
            return MockBackend::status_success;
        });

    EXPECT_CALL(mock_impl, query_record_counter_id(_, _))
        .WillRepeatedly(
            [](MockBackend::counter_record_t record, MockBackend::counter_id_t* out) {
                out->handle = record.id;
                return MockBackend::status_success;
            });
}
}  // namespace

// A raw hardware counter is cumulative in the SDK; each sample must be the count
// accumulated during that sample interval.
TEST_F(SdkPmcDeviceTest, RawCounterSampleIsIntervalDelta)
{
    constexpr counter_id_t                 k_waves_id         = 10;
    constexpr std::array<double, 3>        k_waves_cumulative = { 100.0, 250.0, 450.0 };
    constexpr std::array<double, 3>        k_waves_interval   = { 100.0, 150.0, 200.0 };
    constexpr std::array<std::uint64_t, 3> k_timestamps       = { 1000, 2000, 3000 };

    device<MockBackend> dev(mock_backend, test_context, test_agent,
                            { .requested = { raw_counter(k_waves_id, "SQ_WAVES") } });

    auto samples = std::vector<std::vector<MockBackend::counter_record_t>>{};
    for(const double total : k_waves_cumulative)
    {
        samples.push_back({ record_of(k_waves_id, total) });
    }
    expect_cumulative_samples(*m_mock_impl, std::move(samples));

    const enabled_metrics enabled{ {} };
    for(size_t i = 0; i < k_waves_interval.size(); ++i)
    {
        auto const value =
            find_value(dev.sample_metrics(enabled, k_timestamps.at(i)), k_waves_id);
        ASSERT_TRUE(value.has_value()) << "sample " << i;
        EXPECT_DOUBLE_EQ(*value, k_waves_interval.at(i)) << "sample " << i;
    }
}

// SIMD_NUM is a derived counter whose expression is the agent constant simd_count. It
// does not accumulate, so every sample must report its value.
TEST_F(SdkPmcDeviceTest, ConstantDerivedCounterSampleIsItsValue)
{
    constexpr counter_id_t                 k_simd_num_id = 30;
    constexpr double                       k_simd_num    = 4.0;
    constexpr std::array<std::uint64_t, 3> k_timestamps  = { 1000, 2000, 3000 };

    device<MockBackend> dev(
        mock_backend, test_context, test_agent,
        resolve_counters({ derived_counter(k_simd_num_id, "SIMD_NUM", "simd_count") }, {},
                         { constant_counter(0, "simd_count") }));

    auto samples = std::vector<std::vector<MockBackend::counter_record_t>>(
        k_timestamps.size(), { record_of(k_simd_num_id, k_simd_num) });
    expect_cumulative_samples(*m_mock_impl, std::move(samples));

    const enabled_metrics enabled{ {} };
    for(size_t i = 0; i < k_timestamps.size(); ++i)
    {
        auto const value =
            find_value(dev.sample_metrics(enabled, k_timestamps.at(i)), k_simd_num_id);
        ASSERT_TRUE(value.has_value()) << "sample " << i;
        EXPECT_DOUBLE_EQ(*value, k_simd_num) << "sample " << i;
    }
}

// Reproduces ISS-116 / the MfmaUtil field report. The SDK evaluates a derived counter on
// totals since start_context, so under a steady 87 % MFMA load its MfmaUtil climbs
// 43.5 -> 65.25 -> 72.5. Subtracting consecutive samples gives 43.5, 21.75, 7.25.
// A sample must be the derived expression evaluated on the interval deltas of its
// inputs, which is what per-kernel collection reports for the same load.
TEST_F(SdkPmcDeviceTest, DerivedRatioCounterSampleIsEvaluatedOnIntervalDeltas)
{
    constexpr counter_id_t k_busy_id       = 1;
    constexpr counter_id_t k_active_id     = 2;
    constexpr counter_id_t k_simd_num_id   = 3;
    constexpr counter_id_t k_simd_count_id = 4;
    constexpr counter_id_t k_mfma_util_id  = 50;
    constexpr double       k_simd_num      = 4.0;
    constexpr double       k_percent       = 100.0;

    constexpr std::array<double, 3> k_busy_cumulative    = { 1740.0, 5220.0, 8700.0 };
    constexpr std::array<double, 3> k_active_cumulative  = { 1000.0, 2000.0, 3000.0 };
    constexpr std::array<double, 3> k_mfma_util_interval = { 43.5, 87.0, 87.0 };
    constexpr std::array<std::uint64_t, 3> k_timestamps  = { 1000, 2000, 3000 };

    device<MockBackend> dev(
        mock_backend, test_context, test_agent,
        resolve_counters({ derived_counter(k_mfma_util_id, "MfmaUtil", k_mfma_util) },
                         { raw_counter(k_busy_id, "SQ_VALU_MFMA_BUSY_CYCLES"),
                           raw_counter(k_active_id, "GRBM_GUI_ACTIVE"),
                           constant_counter(k_simd_count_id, "simd_count") },
                         { derived_counter(k_simd_num_id, "SIMD_NUM", "simd_count") }));

    auto samples = std::vector<std::vector<MockBackend::counter_record_t>>{};
    for(size_t i = 0; i < k_timestamps.size(); ++i)
    {
        const double busy          = k_busy_cumulative.at(i);
        const double active        = k_active_cumulative.at(i);
        const double sdk_mfma_util = k_percent * busy / (active * k_simd_num);
        samples.push_back({ record_of(k_busy_id, busy), record_of(k_active_id, active),
                            record_of(k_simd_count_id, k_simd_num),
                            record_of(k_mfma_util_id, sdk_mfma_util) });
    }
    expect_cumulative_samples(*m_mock_impl, std::move(samples));

    const enabled_metrics enabled{ {} };
    for(size_t i = 0; i < k_timestamps.size(); ++i)
    {
        auto const result = dev.sample_metrics(enabled, k_timestamps.at(i));
        auto const value  = find_value(result, k_mfma_util_id);
        ASSERT_TRUE(value.has_value()) << "sample " << i;
        EXPECT_DOUBLE_EQ(*value, k_mfma_util_interval.at(i)) << "sample " << i;
        EXPECT_EQ(result.size(), 1U) << "inputs must not be reported, sample " << i;
    }
}

// A counter can be requested itself and be an input of a requested derived counter, and
// it can have several instances. Every instance is reported and enters the reduction.
TEST_F(SdkPmcDeviceTest, RequestedInputWithTwoInstancesIsReportedAndReduced)
{
    constexpr counter_id_t k_busy_xcc0_id   = 11;
    constexpr counter_id_t k_busy_xcc1_id   = 12;
    constexpr counter_id_t k_active_xcc0_id = 21;
    constexpr counter_id_t k_active_xcc1_id = 22;
    constexpr counter_id_t k_simd_num_id    = 30;
    constexpr counter_id_t k_simd_count_id  = 31;
    constexpr counter_id_t k_mfma_util_id   = 50;
    constexpr double       k_simd_num       = 4.0;

    constexpr std::array<double, 2>        k_busy_xcc0_cumulative   = { 1000.0, 3000.0 };
    constexpr std::array<double, 2>        k_busy_xcc1_cumulative   = { 740.0, 2220.0 };
    constexpr std::array<double, 2>        k_active_xcc0_cumulative = { 1000.0, 2000.0 };
    constexpr std::array<double, 2>        k_active_xcc1_cumulative = { 990.0, 1980.0 };
    constexpr std::array<double, 2>        k_mfma_util_interval     = { 43.5, 87.0 };
    constexpr double                       k_active_xcc0_interval   = 1000.0;
    constexpr double                       k_active_xcc1_interval   = 990.0;
    constexpr std::array<std::uint64_t, 2> k_timestamps             = { 1000, 2000 };
    constexpr size_t                       k_reported_count         = 3;

    auto const active_instances =
        std::vector{ raw_counter(k_active_xcc0_id, "GRBM_GUI_ACTIVE"),
                     raw_counter(k_active_xcc1_id, "GRBM_GUI_ACTIVE") };
    device<MockBackend> dev(
        mock_backend, test_context, test_agent,
        resolve_counters({ derived_counter(k_mfma_util_id, "MfmaUtil", k_mfma_util),
                           active_instances.at(0), active_instances.at(1) },
                         { raw_counter(k_busy_xcc0_id, "SQ_VALU_MFMA_BUSY_CYCLES"),
                           raw_counter(k_busy_xcc1_id, "SQ_VALU_MFMA_BUSY_CYCLES"),
                           active_instances.at(0), active_instances.at(1),
                           constant_counter(k_simd_count_id, "simd_count") },
                         { derived_counter(k_simd_num_id, "SIMD_NUM", "simd_count") }));

    auto samples = std::vector<std::vector<MockBackend::counter_record_t>>{};
    for(size_t i = 0; i < k_timestamps.size(); ++i)
    {
        samples.push_back({ record_of(k_busy_xcc0_id, k_busy_xcc0_cumulative.at(i)),
                            record_of(k_busy_xcc1_id, k_busy_xcc1_cumulative.at(i)),
                            record_of(k_active_xcc0_id, k_active_xcc0_cumulative.at(i)),
                            record_of(k_active_xcc1_id, k_active_xcc1_cumulative.at(i)),
                            record_of(k_simd_count_id, k_simd_num) });
    }
    expect_cumulative_samples(*m_mock_impl, std::move(samples));

    const enabled_metrics enabled{ {} };
    for(size_t i = 0; i < k_timestamps.size(); ++i)
    {
        SCOPED_TRACE(i);
        auto const result = dev.sample_metrics(enabled, k_timestamps.at(i));
        expect_value(result, k_mfma_util_id, k_mfma_util_interval.at(i));
        expect_value(result, k_active_xcc0_id, k_active_xcc0_interval);
        expect_value(result, k_active_xcc1_id, k_active_xcc1_interval);
        EXPECT_EQ(result.size(), k_reported_count);
    }
}

// The SDK returns records only for the counters in the counter config, so a record of
// any other counter is a broken invariant, not a value to drop.
TEST_F(SdkPmcDeviceTest, RecordOfCounterOutsideTheConfigThrows)
{
    constexpr counter_id_t  k_waves_id     = 10;
    constexpr counter_id_t  k_unknown_id   = 99;
    constexpr double        k_record_value = 5.0;
    constexpr std::uint64_t k_timestamp    = 1000;

    device<MockBackend> dev(mock_backend, test_context, test_agent,
                            { .requested = { raw_counter(k_waves_id, "SQ_WAVES") } });
    expect_cumulative_samples(*m_mock_impl,
                              { { record_of(k_unknown_id, k_record_value) } });

    const enabled_metrics enabled{ {} };
    auto const            sample = [&dev, &enabled] {
        [[maybe_unused]] const auto& result = dev.sample_metrics(enabled, k_timestamp);
    };
    EXPECT_THROW(sample(), std::logic_error);
}

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
