// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

/// @file
/// This provides generic functionality for constraining data collection within
/// a windows of time. E.g., delay, delay + duration, (delay + duration) * nrepeat
///
/// @todo Migrate delay/duration for sampling, process sampling, and causal profiling
/// to use this
///

#include "defines.hpp"

#include <cstdint>
#include <ctime>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace constraint
{
struct spec;

struct stages
{
    using functor_t = std::function<bool(const spec&)>;

    stages();

    ROCPROFSYS_DEFAULT_COPY_MOVE(stages)

    functor_t init    = [](const spec&) { return true; };
    functor_t wait    = [](const spec&) { return true; };
    functor_t start   = [](const spec&) { return true; };
    functor_t collect = [](const spec&) { return true; };
    functor_t stop    = [](const spec&) { return true; };
};

struct clock_identifier
{
    int              value    = -1;
    std::string_view raw_name = {};
    std::string      name     = {};

    clock_identifier();
    clock_identifier(std::string_view, int);

    ROCPROFSYS_DEFAULT_COPY_MOVE(clock_identifier)

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
    spec(int, double, double, uint64_t = 0, uint64_t = 1);
    spec(clock_identifier, double, double, uint64_t = 0, uint64_t = 1);
    spec(const std::string&, double, double, uint64_t = 0, uint64_t = 1);
    spec(const std::string&);

    ROCPROFSYS_DEFAULT_COPY_MOVE(spec)

    void operator()(const stages&) const;

    double           delay    = 0.0;
    double           duration = 0.0;
    uint64_t         count    = 0;
    uint64_t         repeat   = 1;
    clock_identifier clock_id = {};
};

const std::set<clock_identifier>&
get_valid_clock_ids();

std::vector<spec>
get_trace_specs();

stages
get_trace_stages();
}  // namespace constraint
}  // namespace rocprofsys
