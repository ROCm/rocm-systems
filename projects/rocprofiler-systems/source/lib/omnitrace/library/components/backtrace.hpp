// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include "library/thread_data.hpp"

#include <timemory/components/base.hpp>
#include <timemory/macros/language.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/utility/unwind.hpp>
#include <timemory/variadic/types.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace omnitrace
{
namespace component
{
struct backtrace
: tim::component::empty_base
, tim::concepts::component
{
    static constexpr size_t stack_depth = OMNITRACE_MAX_UNWIND_DEPTH;

    using data_t            = tim::unwind::stack<stack_depth>;
    using cache_type        = typename data_t::cache_type;
    using entry_type        = tim::unwind::processed_entry;
    using clock_type        = std::chrono::steady_clock;
    using value_type        = void;
    using system_clock      = std::chrono::system_clock;
    using system_time_point = typename system_clock::time_point;

    static std::string label();
    static std::string description();

    backtrace()                     = default;
    ~backtrace()                    = default;
    backtrace(const backtrace&)     = default;
    backtrace(backtrace&&) noexcept = default;

    backtrace& operator=(const backtrace&) = default;
    backtrace& operator=(backtrace&&) noexcept = default;

    static std::vector<entry_type> filter_and_patch(const std::vector<entry_type>&);

    static void start();
    static void stop();

    void                    sample(int = -1);
    bool                    empty() const;
    size_t                  size() const;
    std::vector<entry_type> get() const;
    data_t                  get_data() const { return m_data; }

private:
    data_t m_data = {};
};
}  // namespace component
}  // namespace omnitrace
