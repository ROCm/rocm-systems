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

#include "library/common.hpp"
#include "library/components/fwd.hpp"
#include "library/defines.hpp"
#include "library/thread_data.hpp"
#include "library/timemory.hpp"

#include <timemory/components/base.hpp>
#include <timemory/components/papi/papi_array.hpp>
#include <timemory/macros/language.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/sampling/sampler.hpp>
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
    static constexpr size_t num_hw_counters = 8;

    using data_t            = std::array<char[512], 128>;
    using clock_type        = std::chrono::steady_clock;
    using time_point_type   = typename clock_type::time_point;
    using value_type        = void;
    using hw_counters       = tim::component::papi_array<num_hw_counters>;
    using hw_counter_data_t = typename hw_counters::value_type;
    using system_clock      = std::chrono::system_clock;
    using system_time_point = typename system_clock::time_point;

    static void        preinit();
    static std::string label();
    static std::string description();

    backtrace()                 = default;
    ~backtrace()                = default;
    backtrace(backtrace&&)      = default;
    backtrace(const backtrace&) = default;

    backtrace& operator=(const backtrace&) = default;
    backtrace& operator=(backtrace&&) = default;

    bool operator<(const backtrace& rhs) const;

    static std::set<int>      configure(bool, int64_t _tid = threading::get_id());
    static void               post_process(int64_t _tid = threading::get_id());
    static hw_counter_data_t& get_last_hwcounters();

    static void              start();
    static void              stop();
    void                     sample(int = -1);
    bool                     empty() const;
    size_t                   size() const;
    std::vector<std::string> get() const;
    time_point_type          get_timestamp() const;
    int64_t                  get_thread_cpu_timestamp() const;

private:
    int64_t           m_tid        = 0;
    int64_t           m_thr_cpu_ts = 0;
    int64_t           m_mem_peak   = 0;
    size_t            m_size       = 0;
    time_point_type   m_ts         = {};
    data_t            m_data       = {};
    hw_counter_data_t m_hw_counter = {};
};
}  // namespace component
}  // namespace omnitrace

#if !defined(OMNITRACE_EXTERN_COMPONENTS) ||                                             \
    (defined(OMNITRACE_EXTERN_COMPONENTS) && OMNITRACE_EXTERN_COMPONENTS > 0)

#    include <timemory/operations.hpp>

TIMEMORY_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_wall_clock>), true,
    double)

TIMEMORY_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_cpu_clock>), true,
    double)

TIMEMORY_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_fraction>), true,
    double)

#endif
