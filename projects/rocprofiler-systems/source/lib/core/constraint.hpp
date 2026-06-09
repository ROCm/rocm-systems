// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// This provides generic functionality for constraining data collection within
/// a windows of time. E.g., delay, delay + duration, (delay + duration) * nrepeat
///
/// @todo Migrate delay/duration for sampling, process sampling, and causal profiling
/// to use this
///

#include "common/defines.h"

#include <cstdint>
#include <ctime>
#include <set>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace constraint
{
struct clock_identifier
{
    int              value    = -1;
    std::string_view raw_name = {};
    std::string      name     = {};

    clock_identifier() = default;
    clock_identifier(std::string_view, int);

    clock_identifier(const clock_identifier&)                = default;
    clock_identifier(clock_identifier&&) noexcept            = default;
    clock_identifier& operator=(const clock_identifier&)     = default;
    clock_identifier& operator=(clock_identifier&&) noexcept = default;

    std::string as_string() const;

    bool operator<(const clock_identifier& _rhs) const;
    bool operator==(const clock_identifier& _rhs) const;
    bool operator==(int _rhs) const;
    bool operator==(std::string _rhs) const;

    friend std::ostream& operator<<(std::ostream& _os, const clock_identifier& _v)
    {
        return (_os << _v.as_string());
    }
};

struct spec
{
    double           delay    = 0.0;
    double           duration = 0.0;
    std::uint64_t    repeat   = 1;
    clock_identifier clock_id = {};
};

const std::set<clock_identifier>&
get_valid_clock_ids();

std::vector<spec>
get_trace_specs();
}  // namespace constraint
}  // namespace rocprofsys
