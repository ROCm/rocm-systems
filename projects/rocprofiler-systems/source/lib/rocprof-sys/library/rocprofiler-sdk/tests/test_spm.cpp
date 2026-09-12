// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_vars.hpp"
#include "core/config.hpp"
#include "core/timemory.hpp"
#include "rocprof-sys/library/rocprofiler-sdk/spm_internal.hpp"
#include "rocprof-sys/library/rocprofiler-sdk/spm_record_batch.hpp"
#include "rocprof-sys/library/rocprofiler-sdk/spm_sample.hpp"

#include <gtest/gtest.h>
#if ROCPROFSYS_USE_SPM
#    include <rocprofiler-sdk/fwd.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
constexpr auto k_valid_sample_interval = std::uint64_t{ 8192 };
constexpr auto k_counter_id_offset     = std::uint64_t{ 100 };
constexpr auto k_failed_counter_id     = std::uint64_t{ 9 };
constexpr auto k_failed_record_count   = std::size_t{ 2 };
constexpr auto k_transient_counter_id  = std::uint64_t{ 7 };
constexpr auto k_test_serialized_limit = std::uint32_t{ 2 };

using rocprofsys::rocprofiler_sdk::spm::configuration;
using rocprofsys::rocprofiler_sdk::spm::configure_runtime;
using rocprofsys::rocprofiler_sdk::spm::counter_info;
using rocprofsys::rocprofiler_sdk::spm::is_config_valid;
namespace spm_detail = rocprofsys::rocprofiler_sdk::spm::detail;

struct fake_spm_record
{
    std::uint64_t id        = 0;
    std::uint64_t timestamp = 0;
    double        value     = 0.0;
};

class scripted_counter_decoder
{
public:
    explicit scripted_counter_decoder(
        std::unordered_map<std::uint64_t, std::size_t> failures = {})
    : m_failures{ std::move(failures) }
    {}

    [[nodiscard]] std::optional<counter_info> operator()(std::uint64_t instance_id)
    {
        ++m_calls[instance_id];
        if(auto itr = m_failures.find(instance_id);
           itr != m_failures.end() && itr->second > 0)
        {
            --itr->second;
            return std::nullopt;
        }
        return counter_info{ .counter_id          = instance_id + k_counter_id_offset,
                             .counter_instance_id = instance_id };
    }

    [[nodiscard]] std::size_t call_count(std::uint64_t instance_id) const
    {
        if(const auto itr = m_calls.find(instance_id); itr != m_calls.end())
        {
            return itr->second;
        }
        return 0;
    }

private:
    std::unordered_map<std::uint64_t, std::size_t> m_failures;
    std::unordered_map<std::uint64_t, std::size_t> m_calls;
};

template <std::uint32_t MaxSerializedCount = std::numeric_limits<std::uint32_t>::max(),
          typename DecoderT>
[[nodiscard]]
spm_detail::decoded_record_batch
build_test_batch(const std::vector<std::optional<fake_spm_record>>& records,
                 DecoderT                                           decoder)
{
    auto record_ptrs = std::vector<const fake_spm_record*>{};
    record_ptrs.reserve(records.size());
    std::ranges::transform(
        records, std::back_inserter(record_ptrs),
        [](const auto& record) { return record ? &*record : nullptr; });

    return spm_detail::build_record_batch<MaxSerializedCount>(
        std::span<const fake_spm_record* const>{ record_ptrs.data(), record_ptrs.size() },
        std::move(decoder));
}

// Numeric literals below are intentionally compact, table-shaped SPM test data.
// NOLINTBEGIN(readability-magic-numbers)
[[nodiscard]] std::vector<std::optional<fake_spm_record>>
make_dense_test_records()
{
    return {
        fake_spm_record{ .id = 2, .timestamp = 20, .value = 2.0 },
        fake_spm_record{ .id = 1, .timestamp = 10, .value = 1.0 },
        fake_spm_record{ .id = 1, .timestamp = 20, .value = 3.0 },
        fake_spm_record{ .id = 2, .timestamp = 10, .value = 4.0 },
    };
}
// NOLINTEND(readability-magic-numbers)

void
ensure_spm_settings_registered()
{
    auto settings            = rocprofsys::settings::shared_instance();
    auto register_if_missing = [&settings](const char* env_name, auto initial_value) {
        using value_type  = decltype(initial_value);
        auto setting_name = std::string{ env_name };
        if(settings->find(setting_name) == settings->end())
        {
            (void) settings->insert<value_type, value_type>(
                setting_name, setting_name, "SPM unit-test setting",
                value_type{ initial_value }, std::set<std::string>{ "spm" });
        }
    };

    register_if_missing(rocprofsys::env_vars::ROCM_SPM_EVENTS, std::string{});
    register_if_missing(rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL,
                        std::uint64_t{ 0 });
}

// GoogleTest fixtures inherit an abstract TestBody contract but are not interfaces.
// NOLINTNEXTLINE(readability-identifier-naming)
class spm_settings_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ensure_spm_settings_registered();
        m_previous_events = rocprofsys::config::get_setting_value<std::string>(
            std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS });
        m_previous_sample_interval = rocprofsys::config::get_setting_value<std::uint64_t>(
            std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL });
    }

    void TearDown() override
    {
        rocprofsys::config::set_setting_value(
            std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS },
            m_previous_events.value_or(std::string{}));
        rocprofsys::config::set_setting_value(
            std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL },
            m_previous_sample_interval.value_or(std::uint64_t{ 0 }));
    }

    std::optional<std::string>   m_previous_events          = std::nullopt;
    std::optional<std::uint64_t> m_previous_sample_interval = std::nullopt;
};

configuration
make_valid_requested_spm_config()
{
    return configuration{ .counter_events  = { "SQ_WAVES" },
                          .sample_interval = k_valid_sample_interval };
}

void
expect_requested_counter(const spm_detail::requested_counter& counter,
                         const std::string&                   expected_name,
                         std::optional<std::uint64_t>         expected_device_id)
{
    EXPECT_EQ(counter.name, expected_name);
    EXPECT_EQ(counter.device_id, expected_device_id);
}
}  // namespace

TEST_F(spm_settings_test, accessors_reflect_configured_spm_settings)
{
    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS },
        std::string{ "SQ_WAVES,TD_TD_BUSY" }));
    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL },
        k_valid_sample_interval));

    const auto events = rocprofsys::rocprofiler_sdk::spm::get_events();

    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(events.at(0), "SQ_WAVES");
    EXPECT_EQ(events.at(1), "TD_TD_BUSY");
    EXPECT_EQ(rocprofsys::rocprofiler_sdk::spm::get_sample_interval(),
              k_valid_sample_interval);
}

TEST(spm_configuration, requested_reflects_events)
{
    EXPECT_FALSE(configuration{}.requested());

    auto event_config           = configuration{};
    event_config.counter_events = { "SQ_WAVES" };
    EXPECT_TRUE(event_config.requested());
}

TEST(spm_configuration_parsing, parse_device_id_accepts_only_complete_unsigned_values)
{
    EXPECT_EQ(spm_detail::parse_device_id("0"), std::optional<std::uint64_t>{ 0 });
    EXPECT_EQ(spm_detail::parse_device_id("42"), std::optional<std::uint64_t>{ 42 });
    EXPECT_EQ(spm_detail::parse_device_id(""), std::nullopt);
    EXPECT_EQ(spm_detail::parse_device_id("abc"), std::nullopt);
    EXPECT_EQ(spm_detail::parse_device_id("1abc"), std::nullopt);
    EXPECT_EQ(spm_detail::parse_device_id("-1"), std::nullopt);
}

TEST(spm_configuration_parsing, parse_counter_name_trims_and_removes_device_qualifier)
{
    EXPECT_EQ(spm_detail::parse_counter_name(" SQ_WAVES "), "SQ_WAVES");
    EXPECT_EQ(spm_detail::parse_counter_name(" SQ_WAVES:device=0 "), "SQ_WAVES");
    EXPECT_EQ(spm_detail::parse_counter_name(":device=0"), "");
}

TEST(spm_configuration_parsing, parse_requested_counters_skips_empty_and_invalid_entries)
{
    const auto parsed = spm_detail::parse_requested_counters(
        configuration{ .counter_events  = { " SQ_WAVES:device=0 ", "", "TD_TD_BUSY",
                                            "BAD:device=abc", ":device=1" },
                       .sample_interval = k_valid_sample_interval });

    ASSERT_EQ(parsed.size(), 2);
    expect_requested_counter(parsed.at(0), "SQ_WAVES", std::uint64_t{ 0 });
    expect_requested_counter(parsed.at(1), "TD_TD_BUSY", std::nullopt);
}

TEST(spm_configuration_parsing,
     requested_counters_for_device_keeps_unqualified_and_matching)
{
    const auto parsed = spm_detail::parse_requested_counters(configuration{
        .counter_events  = { "SQ_WAVES:device=0", "TD_TD_BUSY:device=1", "TCC_HIT" },
        .sample_interval = k_valid_sample_interval });

    const auto device_zero = spm_detail::requested_counters_for_device(parsed, 0);
    ASSERT_EQ(device_zero.size(), 2);
    expect_requested_counter(device_zero.at(0), "SQ_WAVES", std::uint64_t{ 0 });
    expect_requested_counter(device_zero.at(1), "TCC_HIT", std::nullopt);

    const auto device_one = spm_detail::requested_counters_for_device(parsed, 1);
    ASSERT_EQ(device_one.size(), 2);
    expect_requested_counter(device_one.at(0), "TD_TD_BUSY", std::uint64_t{ 1 });
    expect_requested_counter(device_one.at(1), "TCC_HIT", std::nullopt);
}

TEST(spm_configuration_parsing, requested_counter_names_deduplicates_parsed_names)
{
    const auto parsed = spm_detail::parse_requested_counters(configuration{
        .counter_events  = { "SQ_WAVES:device=0", "SQ_WAVES:device=1", "TD_TD_BUSY" },
        .sample_interval = k_valid_sample_interval });

    const auto names = spm_detail::requested_counter_names(parsed);
    EXPECT_EQ(names, (std::unordered_set<std::string>{ "SQ_WAVES", "TD_TD_BUSY" }));
}

// Numeric literals below are intentionally compact, table-shaped SPM test data.
// NOLINTBEGIN(readability-magic-numbers)
TEST(spm_build_record_batch, decodes_each_dense_counter_once)
{
    auto       decoder = scripted_counter_decoder{};
    const auto batch   = build_test_batch(make_dense_test_records(), std::ref(decoder));

    ASSERT_EQ(batch.counters.size(), 2);
    EXPECT_EQ(decoder.call_count(1), 1);
    EXPECT_EQ(decoder.call_count(2), 1);
    EXPECT_EQ(batch.counters.at(0).counter_id, 2 + k_counter_id_offset);
    EXPECT_EQ(batch.counters.at(0).counter_instance_id, 2);
    EXPECT_EQ(batch.counters.at(1).counter_id, 1 + k_counter_id_offset);
    EXPECT_EQ(batch.counters.at(1).counter_instance_id, 1);
}

TEST(spm_build_record_batch, preserves_dense_sample_and_value_order)
{
    const auto batch =
        build_test_batch(make_dense_test_records(), scripted_counter_decoder{});

    ASSERT_EQ(batch.samples.size(), 2);
    EXPECT_EQ(batch.samples.at(0).timestamp, 20);
    ASSERT_EQ(batch.samples.at(0).values.size(), 2);
    EXPECT_EQ(batch.samples.at(0).values.at(0).counter_info_index, 0);
    EXPECT_DOUBLE_EQ(batch.samples.at(0).values.at(0).value, 2.0);
    EXPECT_EQ(batch.samples.at(0).values.at(1).counter_info_index, 1);
    EXPECT_DOUBLE_EQ(batch.samples.at(0).values.at(1).value, 3.0);

    EXPECT_EQ(batch.samples.at(1).timestamp, 10);
    ASSERT_EQ(batch.samples.at(1).values.size(), 2);
    EXPECT_EQ(batch.samples.at(1).values.at(0).counter_info_index, 1);
    EXPECT_DOUBLE_EQ(batch.samples.at(1).values.at(0).value, 1.0);
    EXPECT_EQ(batch.samples.at(1).values.at(1).counter_info_index, 0);
    EXPECT_DOUBLE_EQ(batch.samples.at(1).values.at(1).value, 4.0);
}

// GTest assertion macros inflate the measured complexity of this table-shaped check.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(spm_build_record_batch, groups_sparse_records_without_changing_order)
{
    const auto batch = build_test_batch(
        {
            fake_spm_record{ .id = 1, .timestamp = 10, .value = 1.0 },
            fake_spm_record{ .id = 2, .timestamp = 20, .value = 2.0 },
            fake_spm_record{ .id = 1, .timestamp = 30, .value = 3.0 },
            fake_spm_record{ .id = 2, .timestamp = 40, .value = 4.0 },
        },
        scripted_counter_decoder{});

    ASSERT_EQ(batch.counters.size(), 2);
    ASSERT_EQ(batch.samples.size(), 4);
    EXPECT_EQ(batch.samples.at(0).timestamp, 10);
    EXPECT_EQ(batch.samples.at(1).timestamp, 20);
    EXPECT_EQ(batch.samples.at(2).timestamp, 30);
    EXPECT_EQ(batch.samples.at(3).timestamp, 40);
    const auto expected_counter_indices = std::array<std::uint32_t, 4>{ 0, 1, 0, 1 };
    const auto expected_values          = std::array<double, 4>{ 1.0, 2.0, 3.0, 4.0 };
    for(std::size_t index = 0; index < batch.samples.size(); ++index)
    {
        const auto& values = batch.samples.at(index).values;
        ASSERT_EQ(values.size(), 1);
        EXPECT_EQ(values.front().counter_info_index, expected_counter_indices.at(index));
        EXPECT_DOUBLE_EQ(values.front().value, expected_values.at(index));
    }
}

TEST(spm_build_record_batch, returns_empty_for_empty_input)
{
    const auto batch = build_test_batch({}, scripted_counter_decoder{});
    EXPECT_TRUE(batch.counters.empty());
    EXPECT_TRUE(batch.samples.empty());
}

TEST(spm_build_record_batch, returns_empty_for_all_null_input)
{
    const auto batch =
        build_test_batch({ std::nullopt, std::nullopt }, scripted_counter_decoder{});
    EXPECT_TRUE(batch.counters.empty());
    EXPECT_TRUE(batch.samples.empty());
}

TEST(spm_build_record_batch, skips_interspersed_null_records)
{
    const auto batch = build_test_batch(
        {
            std::nullopt,
            fake_spm_record{ .id = 1, .timestamp = 10, .value = 1.0 },
            std::nullopt,
            fake_spm_record{ .id = 2, .timestamp = 10, .value = 2.0 },
        },
        scripted_counter_decoder{});

    ASSERT_EQ(batch.counters.size(), 2);
    ASSERT_EQ(batch.samples.size(), 1);
    ASSERT_EQ(batch.samples.front().values.size(), 2);
    EXPECT_EQ(batch.samples.front().values.at(0).counter_info_index, 0);
    EXPECT_DOUBLE_EQ(batch.samples.front().values.at(0).value, 1.0);
    EXPECT_EQ(batch.samples.front().values.at(1).counter_info_index, 1);
    EXPECT_DOUBLE_EQ(batch.samples.front().values.at(1).value, 2.0);
}

TEST(spm_build_record_batch, skips_records_when_counter_decode_fails)
{
    auto decoder =
        scripted_counter_decoder{ { { k_failed_counter_id, k_failed_record_count } } };
    const auto batch = build_test_batch(
        {
            fake_spm_record{ .id = 1, .timestamp = 10, .value = 1.0 },
            fake_spm_record{ .id = k_failed_counter_id, .timestamp = 20, .value = 9.0 },
            fake_spm_record{ .id = k_failed_counter_id, .timestamp = 25, .value = 10.0 },
            fake_spm_record{ .id = 2, .timestamp = 30, .value = 2.0 },
        },
        std::ref(decoder));

    EXPECT_EQ(decoder.call_count(k_failed_counter_id), k_failed_record_count);
    ASSERT_EQ(batch.counters.size(), 2);
    EXPECT_EQ(batch.counters.at(0).counter_instance_id, 1);
    EXPECT_EQ(batch.counters.at(1).counter_instance_id, 2);
    ASSERT_EQ(batch.samples.size(), 2);
    EXPECT_EQ(batch.samples.at(0).timestamp, 10);
    ASSERT_EQ(batch.samples.at(0).values.size(), 1);
    EXPECT_EQ(batch.samples.at(0).values.front().counter_info_index, 0);
    EXPECT_DOUBLE_EQ(batch.samples.at(0).values.front().value, 1.0);
    EXPECT_EQ(batch.samples.at(1).timestamp, 30);
    ASSERT_EQ(batch.samples.at(1).values.size(), 1);
    EXPECT_EQ(batch.samples.at(1).values.front().counter_info_index, 1);
    EXPECT_DOUBLE_EQ(batch.samples.at(1).values.front().value, 2.0);
}

TEST(spm_build_record_batch, retries_transient_decode_without_reordering)
{
    auto       decoder = scripted_counter_decoder{ { { k_transient_counter_id, 1 } } };
    const auto batch   = build_test_batch(
        {
            fake_spm_record{
                  .id = k_transient_counter_id, .timestamp = 10, .value = 1.0 },
            fake_spm_record{ .id = 1, .timestamp = 20, .value = 2.0 },
            fake_spm_record{
                  .id = k_transient_counter_id, .timestamp = 30, .value = 3.0 },
        },
        std::ref(decoder));

    EXPECT_EQ(decoder.call_count(k_transient_counter_id), 2);
    ASSERT_EQ(batch.counters.size(), 2);
    EXPECT_EQ(batch.counters.at(0).counter_instance_id, 1);
    EXPECT_EQ(batch.counters.at(1).counter_instance_id, k_transient_counter_id);

    ASSERT_EQ(batch.samples.size(), 2);
    EXPECT_EQ(batch.samples.at(0).timestamp, 20);
    ASSERT_EQ(batch.samples.at(0).values.size(), 1);
    EXPECT_EQ(batch.samples.at(0).values.at(0).counter_info_index, 0);
    EXPECT_DOUBLE_EQ(batch.samples.at(0).values.at(0).value, 2.0);
    EXPECT_EQ(batch.samples.at(1).timestamp, 30);
    ASSERT_EQ(batch.samples.at(1).values.size(), 1);
    EXPECT_EQ(batch.samples.at(1).values.at(0).counter_info_index, 1);
    EXPECT_DOUBLE_EQ(batch.samples.at(1).values.at(0).value, 3.0);
}

TEST(spm_build_record_batch, returns_empty_when_all_counter_decodes_fail)
{
    auto decoder =
        scripted_counter_decoder{ { { k_failed_counter_id, k_failed_record_count } } };
    const auto batch = build_test_batch(
        {
            fake_spm_record{ .id = k_failed_counter_id, .timestamp = 10, .value = 1.0 },
            fake_spm_record{ .id = k_failed_counter_id, .timestamp = 20, .value = 2.0 },
        },
        std::ref(decoder));

    EXPECT_EQ(decoder.call_count(k_failed_counter_id), k_failed_record_count);
    EXPECT_TRUE(batch.counters.empty());
    EXPECT_TRUE(batch.samples.empty());
}

TEST(spm_build_record_batch, preserves_duplicate_timestamp_counter_values)
{
    const auto batch = build_test_batch(
        {
            fake_spm_record{ .id = 1, .timestamp = 10, .value = 1.0 },
            fake_spm_record{ .id = 1, .timestamp = 10, .value = 2.0 },
        },
        scripted_counter_decoder{});

    ASSERT_EQ(batch.counters.size(), 1);
    ASSERT_EQ(batch.samples.size(), 1);
    ASSERT_EQ(batch.samples.front().values.size(), 2);
    EXPECT_EQ(batch.samples.front().values.at(0).counter_info_index, 0);
    EXPECT_DOUBLE_EQ(batch.samples.front().values.at(0).value, 1.0);
    EXPECT_EQ(batch.samples.front().values.at(1).counter_info_index, 0);
    EXPECT_DOUBLE_EQ(batch.samples.front().values.at(1).value, 2.0);
}

TEST(spm_build_record_batch, propagates_decoder_exception)
{
    const auto throwing_decoder =
        []([[maybe_unused]] std::uint64_t instance_id) -> std::optional<counter_info> {
        throw std::runtime_error{ "expected decoder failure" };
    };

    EXPECT_THROW(static_cast<void>(build_test_batch(
                     { fake_spm_record{ .id = 1, .timestamp = 10, .value = 1.0 } },
                     throwing_decoder)),
                 std::runtime_error);
}

TEST(spm_build_record_batch, rejects_counter_count_above_serialized_limit)
{
    EXPECT_THROW(static_cast<void>(build_test_batch<k_test_serialized_limit>(
                     {
                         fake_spm_record{ .id = 1, .timestamp = 10, .value = 1.0 },
                         fake_spm_record{ .id = 2, .timestamp = 10, .value = 2.0 },
                         fake_spm_record{ .id = 3, .timestamp = 20, .value = 3.0 },
                     },
                     scripted_counter_decoder{})),
                 std::length_error);
}

TEST(spm_build_record_batch, rejects_sample_count_above_serialized_limit)
{
    EXPECT_THROW(static_cast<void>(build_test_batch<k_test_serialized_limit>(
                     {
                         fake_spm_record{ .id = 1, .timestamp = 10, .value = 1.0 },
                         fake_spm_record{ .id = 1, .timestamp = 20, .value = 2.0 },
                         fake_spm_record{ .id = 1, .timestamp = 30, .value = 3.0 },
                     },
                     scripted_counter_decoder{})),
                 std::length_error);
}

TEST(spm_build_record_batch, rejects_value_count_above_serialized_limit)
{
    EXPECT_THROW(static_cast<void>(build_test_batch<k_test_serialized_limit>(
                     {
                         fake_spm_record{ .id = 1, .timestamp = 10, .value = 1.0 },
                         fake_spm_record{ .id = 1, .timestamp = 10, .value = 2.0 },
                         fake_spm_record{ .id = 1, .timestamp = 10, .value = 3.0 },
                     },
                     scripted_counter_decoder{})),
                 std::length_error);
}

TEST(spm_build_record_batch, accepts_counts_at_serialized_limit)
{
    const auto batch = build_test_batch<k_test_serialized_limit>(
        {
            fake_spm_record{ .id = 1, .timestamp = 10, .value = 1.0 },
            fake_spm_record{ .id = 2, .timestamp = 10, .value = 2.0 },
            fake_spm_record{ .id = 1, .timestamp = 20, .value = 3.0 },
        },
        scripted_counter_decoder{});

    ASSERT_EQ(batch.counters.size(), k_test_serialized_limit);
    ASSERT_EQ(batch.samples.size(), k_test_serialized_limit);
    EXPECT_EQ(batch.samples.at(0).values.size(), k_test_serialized_limit);
    EXPECT_EQ(batch.samples.at(1).values.size(), 1);
}
// NOLINTEND(readability-magic-numbers)

TEST_F(spm_settings_test, events_request_spm_but_default_interval_is_invalid)
{
    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS }, std::string{ "SQ_WAVES" }));
    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL },
        std::uint64_t{ 0 }));

    const auto events = rocprofsys::rocprofiler_sdk::spm::get_events();

    EXPECT_EQ(events, std::vector<std::string>{ "SQ_WAVES" });
    EXPECT_EQ(rocprofsys::rocprofiler_sdk::spm::get_sample_interval(), 0);
    EXPECT_FALSE(
        is_config_valid(configuration{ .counter_events = events, .sample_interval = 0 }));
}

TEST(spm_config_validation, accepts_when_spm_is_not_requested)
{
    EXPECT_TRUE(is_config_valid(configuration{}));
}

TEST(spm_config_validation, accepts_sample_interval_without_events)
{
    EXPECT_TRUE(is_config_valid(configuration{
        .counter_events = {}, .sample_interval = k_valid_sample_interval }));
}

TEST(spm_config_validation, rejects_zero_sample_interval)
{
    auto requested_config            = make_valid_requested_spm_config();
    requested_config.sample_interval = 0;

    EXPECT_FALSE(is_config_valid(requested_config));
}

TEST(spm_config_validation, accepts_valid_requested_spm_configuration)
{
    const auto requested_config = make_valid_requested_spm_config();

    EXPECT_TRUE(is_config_valid(requested_config));
}

#if ROCPROFSYS_USE_SPM
TEST(spm_runtime_configuration, classifies_only_context_conflict_as_fatal)
{
    using result_type = spm_detail::RuntimeConfigurationResult;

    EXPECT_EQ(
        spm_detail::classify_runtime_configuration_status(ROCPROFILER_STATUS_SUCCESS),
        result_type::Configured);
    EXPECT_EQ(spm_detail::classify_runtime_configuration_status(ROCPROFILER_STATUS_ERROR),
              result_type::Unavailable);
    EXPECT_EQ(spm_detail::classify_runtime_configuration_status(
                  ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT),
              result_type::FatalError);
}
#endif

TEST(spm_runtime_configuration, accepts_when_spm_is_not_requested)
{
    EXPECT_TRUE(configure_runtime(nullptr, configuration{}));
}

TEST(spm_runtime_configuration, accepts_sample_interval_without_events)
{
    EXPECT_TRUE(configure_runtime(
        nullptr, configuration{ .counter_events  = {},
                                .sample_interval = k_valid_sample_interval }));
}

// The unit-test target disables the SDK runtime, so a valid request without client
// data exercises the non-fatal unavailable-service fallback.
TEST(spm_runtime_configuration, valid_request_without_client_data_is_non_fatal)
{
    EXPECT_TRUE(configure_runtime(nullptr, make_valid_requested_spm_config()));
}

TEST(spm_runtime_configuration, rejects_zero_sample_interval)
{
    auto requested_config            = make_valid_requested_spm_config();
    requested_config.sample_interval = 0;

    EXPECT_FALSE(configure_runtime(nullptr, requested_config));
}
