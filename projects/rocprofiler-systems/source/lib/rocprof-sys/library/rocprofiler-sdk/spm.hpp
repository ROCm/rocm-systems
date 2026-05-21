// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace spm
{
struct beta_request
{
    bool                     enabled              = false;
    bool                     sdk_header_available = false;
    std::vector<std::string> events               = {};
    std::uint64_t            sample_interval      = 0;
    std::string              sample_interval_unit = {};

    bool requested() const;
};

beta_request
get_request();

bool
validate_beta_request(const beta_request&             request,
                      const std::vector<std::string>& dispatch_counter_events);

void
prepare_beta_environment(const beta_request& request);
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
