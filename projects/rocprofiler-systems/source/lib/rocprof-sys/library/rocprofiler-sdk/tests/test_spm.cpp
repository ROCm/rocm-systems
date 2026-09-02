// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_vars.hpp"
#include "core/config.hpp"
#include "core/timemory.hpp"
#include "rocprof-sys/library/rocprofiler-sdk/spm_internal.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{
constexpr auto valid_sample_interval = std::uint64_t{ 8192 };

using rocprofsys::rocprofiler_sdk::spm::configuration;
using rocprofsys::rocprofiler_sdk::spm::configure_runtime;
using rocprofsys::rocprofiler_sdk::spm::is_config_valid;
namespace spm_detail = rocprofsys::rocprofiler_sdk::spm::detail;

void
ensure_spm_settings_registered()
{
    auto settings = rocprofsys::settings::shared_instance();
    auto register_if_missing = [&settings](const char* env_name, auto initial_value) {
        using value_type = decltype(initial_value);
        auto setting_name = std::string{ env_name };
        if(settings->find(setting_name) == settings->end())
        {
            (void) settings->insert<value_type, value_type>(
                setting_name, setting_name, "SPM unit-test setting",
                value_type{ initial_value }, std::set<std::string>{ "spm" });
        }
    };

    register_if_missing(rocprofsys::env_vars::ROCM_EVENTS, std::string{});
    register_if_missing(rocprofsys::env_vars::ROCM_SPM_EVENTS, std::string{});
    register_if_missing(rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL,
                        std::uint64_t{ 0 });
}

class spm_settings_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ensure_spm_settings_registered();
        previous_events = rocprofsys::config::get_setting_value<std::string>(
            std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS });
        previous_sample_interval = rocprofsys::config::get_setting_value<std::uint64_t>(
            std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL });
        previous_dispatch_events = rocprofsys::config::get_setting_value<std::string>(
            std::string{ rocprofsys::env_vars::ROCM_EVENTS });
    }

    void TearDown() override
    {
        rocprofsys::config::set_setting_value(
            std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS },
            previous_events.value_or(std::string{}));
        rocprofsys::config::set_setting_value(
            std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL },
            previous_sample_interval.value_or(std::uint64_t{ 0 }));
        rocprofsys::config::set_setting_value(
            std::string{ rocprofsys::env_vars::ROCM_EVENTS },
            previous_dispatch_events.value_or(std::string{}));
    }

    std::optional<std::string>   previous_events          = std::nullopt;
    std::optional<std::uint64_t> previous_sample_interval = std::nullopt;
    std::optional<std::string>   previous_dispatch_events = std::nullopt;
};

configuration
make_valid_requested_spm_config()
{
    return configuration{ .counter_events  = { "SQ_WAVES" },
                          .sample_interval = valid_sample_interval };
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
        valid_sample_interval));

    const auto events = rocprofsys::rocprofiler_sdk::spm::get_events();

    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(events.at(0), "SQ_WAVES");
    EXPECT_EQ(events.at(1), "TD_TD_BUSY");
    EXPECT_EQ(rocprofsys::rocprofiler_sdk::spm::get_sample_interval(),
              valid_sample_interval);
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
                       .sample_interval = valid_sample_interval });

    ASSERT_EQ(parsed.size(), 2);
    expect_requested_counter(parsed.at(0), "SQ_WAVES", std::uint64_t{ 0 });
    expect_requested_counter(parsed.at(1), "TD_TD_BUSY", std::nullopt);
}

TEST(spm_configuration_parsing,
     requested_counters_for_device_keeps_unqualified_and_matching)
{
    const auto parsed = spm_detail::parse_requested_counters(configuration{
        .counter_events  = { "SQ_WAVES:device=0", "TD_TD_BUSY:device=1", "TCC_HIT" },
        .sample_interval = valid_sample_interval });

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
        .sample_interval = valid_sample_interval });

    const auto names = spm_detail::requested_counter_names(parsed);
    EXPECT_EQ(names, (std::unordered_set<std::string>{ "SQ_WAVES", "TD_TD_BUSY" }));
}

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
    EXPECT_FALSE(is_config_valid(
        configuration{ .counter_events = events, .sample_interval = 0 }, {}, {}));
}

TEST(spm_config_validation, accepts_when_spm_is_not_requested)
{
    EXPECT_TRUE(is_config_valid(configuration{}, {}, {}));
}

TEST(spm_config_validation, accepts_sample_interval_without_events)
{
    EXPECT_TRUE(is_config_valid(
        configuration{ .counter_events = {}, .sample_interval = valid_sample_interval },
        {}, {}));
}

TEST(spm_config_validation, rejects_rocm_dispatch_counter_conflict)
{
    const auto requested_config = make_valid_requested_spm_config();

    EXPECT_FALSE(is_config_valid(requested_config, { "SQ_WAVES" }, {}));
}

TEST(spm_config_validation, rejects_gpu_perf_counter_conflict)
{
    const auto requested_config = make_valid_requested_spm_config();

    EXPECT_FALSE(is_config_valid(requested_config, {}, "SQ_WAVES"));
}

TEST(spm_config_validation, rejects_zero_sample_interval)
{
    auto requested_config            = make_valid_requested_spm_config();
    requested_config.sample_interval = 0;

    EXPECT_FALSE(is_config_valid(requested_config, {}, {}));
}

TEST(spm_config_validation, accepts_valid_requested_spm_configuration)
{
    const auto requested_config = make_valid_requested_spm_config();

    EXPECT_TRUE(is_config_valid(requested_config, {}, {}));
}

TEST(spm_runtime_configuration, accepts_when_spm_is_not_requested)
{
    EXPECT_TRUE(configure_runtime(nullptr, configuration{}, {}, {}));
}

TEST(spm_runtime_configuration, accepts_sample_interval_without_events)
{
    EXPECT_TRUE(configure_runtime(
        nullptr,
        configuration{ .counter_events = {}, .sample_interval = valid_sample_interval },
        {}, {}));
}

// Invalid user configuration is the only case that fails tool initialization, so it
// must be rejected before any client data is touched.
TEST(spm_runtime_configuration, rejects_rocm_dispatch_counter_conflict)
{
    EXPECT_FALSE(configure_runtime(nullptr, make_valid_requested_spm_config(),
                                   { "SQ_WAVES" }, {}));
}

TEST(spm_runtime_configuration, rejects_zero_sample_interval)
{
    auto requested_config            = make_valid_requested_spm_config();
    requested_config.sample_interval = 0;

    EXPECT_FALSE(configure_runtime(nullptr, requested_config, {}, {}));
}

TEST(spm_runtime_configuration, rejects_counter_conflict_with_zero_interval)
{
    auto requested_config            = make_valid_requested_spm_config();
    requested_config.sample_interval = 0;

    EXPECT_FALSE(configure_runtime(nullptr, requested_config, { "SQ_WAVES" }, {}));
}
