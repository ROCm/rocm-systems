// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/defines.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace rocprofiler
{
namespace common
{
struct defer_start
{};

struct simple_timer
{
    using duration_t = std::chrono::duration<double>;

    explicit simple_timer(std::string&& label);
    explicit simple_timer(std::string&& label, defer_start);
    ~simple_timer();

    void             start();
    void             stop();
    double           get() const;
    size_t           get_nsec() const;
    std::string_view label() const { return std::string_view{m_label}; }
    void             set_quiet(bool v) const { m_quiet = v; }

    friend std::ostream& operator<<(std::ostream& _os, const simple_timer& _val);

private:
    using clock_type   = std::chrono::steady_clock;
    using time_point_t = std::chrono::time_point<clock_type, std::chrono::nanoseconds>;

    std::string  m_label = {};
    time_point_t m_beg   = {};
    time_point_t m_end   = {};
    mutable bool m_quiet = false;
};
}  // namespace common
}  // namespace rocprofiler
