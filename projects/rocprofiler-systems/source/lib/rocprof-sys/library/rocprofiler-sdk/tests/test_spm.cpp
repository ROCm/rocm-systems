// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/rocm_spm.hpp"
#include "rocprof-sys/library/rocprofiler-sdk/spm.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace
{
using rocprofsys::rocprofiler_sdk::spm::beta_request;
using rocprofsys::rocprofiler_sdk::spm::validate_beta_request;

beta_request
make_valid_requested_spm_request()
{
    const auto unit =
        std::string{ rocprofsys::common::rocm_spm_sample_interval_unit_sclk_cycles };
    return beta_request{ true, true, { "SQ_WAVES" }, 4200, unit };
}
}  // namespace

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
    EXPECT_TRUE(validate_beta_request(beta_request{}, {}));
}

TEST(spm_beta_validation, rejects_requested_spm_without_sdk_header)
{
    auto request                 = make_valid_requested_spm_request();
    request.sdk_header_available = false;

    EXPECT_FALSE(validate_beta_request(request, {}));
}

TEST(spm_beta_validation, rejects_requested_spm_until_runtime_collection_lands)
{
    const auto request = make_valid_requested_spm_request();

    EXPECT_FALSE(validate_beta_request(request, {}));
}
