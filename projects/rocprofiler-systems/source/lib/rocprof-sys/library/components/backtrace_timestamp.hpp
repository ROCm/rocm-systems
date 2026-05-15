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

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"

#include <timemory/components/base.hpp>
#include <timemory/macros/language.hpp>
#include <timemory/mpl/concepts.hpp>

#include <chrono>
#include <cstdint>

namespace rocprofsys
{
namespace component
{
struct backtrace_timestamp : comp::empty_base
{
    using value_type = void;

    static std::string label() { return "backtrace_timestamp"; }
    static std::string description() { return "Timestamp for backtrace"; }

    backtrace_timestamp()                               = default;
    ~backtrace_timestamp()                              = default;
    backtrace_timestamp(const backtrace_timestamp&)     = default;
    backtrace_timestamp(backtrace_timestamp&&) noexcept = default;

    backtrace_timestamp& operator=(const backtrace_timestamp&)     = default;
    backtrace_timestamp& operator=(backtrace_timestamp&&) noexcept = default;

    bool operator<(const backtrace_timestamp& rhs) const;

    static void start() {}
    static void stop() {}

    void sample(int = -1);

    auto get_tid() const { return m_tid; }
    auto get_timestamp() const { return m_real; }
    bool is_valid() const;

private:
    int64_t  m_tid  = 0;
    uint64_t m_real = 0;
};
}  // namespace component
}  // namespace rocprofsys
