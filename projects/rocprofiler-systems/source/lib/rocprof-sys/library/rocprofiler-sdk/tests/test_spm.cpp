// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_vars.hpp"
#include "common/rocm_spm.hpp"
#include "core/config.hpp"
#include "core/rocprofiler-sdk.hpp"
#include "rocprof-sys/library/rocprofiler-sdk/spm.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace
{
using rocprofsys::rocprofiler_sdk::spm::beta_request;
using rocprofsys::rocprofiler_sdk::spm::get_request;
using rocprofsys::rocprofiler_sdk::spm::validate_beta_request;

beta_request
make_valid_requested_spm_request()
{
    const auto unit =
        std::string{ rocprofsys::common::rocm_spm_sample_interval_unit_sclk_cycles };
    return beta_request{ true, { "SQ_WAVES" }, 4200, unit };
}
}  // namespace

TEST(spm_beta_request, get_request_reflects_configured_spm_settings)
{
    const auto unit =
        std::string{ rocprofsys::common::rocm_spm_sample_interval_unit_sclk_cycles };
    auto settings = rocprofsys::settings::shared_instance();
    rocprofsys::rocprofiler_sdk::config_settings(settings);

    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_ENABLED }, true));
    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS },
        std::string{ "SQ_WAVES,TD_TD_BUSY" }));
    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL },
        std::uint64_t{ 4200 }));

    const auto request = get_request();

    EXPECT_TRUE(request.enabled);
    ASSERT_EQ(request.events.size(), 2);
    EXPECT_EQ(request.events.at(0), "SQ_WAVES");
    EXPECT_EQ(request.events.at(1), "TD_TD_BUSY");
    EXPECT_EQ(request.sample_interval, 4200);
    EXPECT_EQ(request.sample_interval_unit, unit);

    EXPECT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_ENABLED }, false));
    EXPECT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS }, std::string{}));
    EXPECT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL },
        std::uint64_t{ 0 }));
}

TEST(spm_beta_request, requested_reflects_enabled_flag_or_events)
{
    EXPECT_FALSE(beta_request{}.requested());

    auto enabled_request    = beta_request{};
    enabled_request.enabled = true;
    EXPECT_TRUE(enabled_request.requested());

    auto event_request   = beta_request{};
    event_request.events = { "SQ_WAVES" };
    EXPECT_TRUE(event_request.requested());
}

TEST(spm_beta_validation, accepts_when_spm_is_not_requested)
{
    EXPECT_TRUE(validate_beta_request(beta_request{}, {}, {}));
}

TEST(spm_beta_validation, rejects_enabled_request_without_events)
{
    auto request    = beta_request{};
    request.enabled = true;

    EXPECT_FALSE(validate_beta_request(request, {}, {}));
}

TEST(spm_beta_validation, rejects_rocm_dispatch_counter_conflict)
{
    const auto request = make_valid_requested_spm_request();

    EXPECT_FALSE(validate_beta_request(request, { "SQ_WAVES" }, {}));
}

TEST(spm_beta_validation, rejects_gpu_perf_counter_conflict)
{
    const auto request = make_valid_requested_spm_request();

    EXPECT_FALSE(validate_beta_request(request, {}, "SQ_WAVES"));
}

TEST(spm_beta_validation, rejects_zero_sample_interval)
{
    auto request            = make_valid_requested_spm_request();
    request.sample_interval = 0;

    EXPECT_FALSE(validate_beta_request(request, {}, {}));
}

TEST(spm_beta_validation, rejects_unsupported_sample_interval_unit)
{
    auto request                 = make_valid_requested_spm_request();
    request.sample_interval_unit = "ns";

    EXPECT_FALSE(validate_beta_request(request, {}, {}));
}

TEST(spm_beta_validation, accepts_valid_requested_spm_request)
{
    const auto request = make_valid_requested_spm_request();

    EXPECT_TRUE(validate_beta_request(request, {}, {}));
}
