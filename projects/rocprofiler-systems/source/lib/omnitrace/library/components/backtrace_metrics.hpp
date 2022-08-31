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
#include "library/components/backtrace.hpp"
#include "library/components/fwd.hpp"
#include "library/defines.hpp"
#include "library/thread_data.hpp"
#include "library/timemory.hpp"

#include <timemory/components/base.hpp>
#include <timemory/components/papi/papi_array.hpp>
#include <timemory/components/papi/types.hpp>
#include <timemory/macros/language.hpp>
#include <timemory/mpl/concepts.hpp>
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
struct backtrace_metrics
: tim::component::empty_base
, tim::concepts::component
{
    static constexpr size_t num_hw_counters = TIMEMORY_PAPI_ARRAY_SIZE;

    using clock_type        = std::chrono::steady_clock;
    using value_type        = void;
    using hw_counters       = tim::component::papi_array<num_hw_counters>;
    using hw_counter_data_t = typename hw_counters::value_type;
    using system_clock      = std::chrono::system_clock;
    using system_time_point = typename system_clock::time_point;

    static std::string label();
    static std::string description();

    backtrace_metrics()                             = default;
    ~backtrace_metrics()                            = default;
    backtrace_metrics(const backtrace_metrics&)     = default;
    backtrace_metrics(backtrace_metrics&&) noexcept = default;

    backtrace_metrics& operator=(const backtrace_metrics&) = default;
    backtrace_metrics& operator=(backtrace_metrics&&) noexcept = default;

    static void configure(bool, int64_t _tid = threading::get_id());
    static void init_perfetto(int64_t _tid);
    static void fini_perfetto(int64_t _tid);

    static void start();
    static void stop();
    void        sample(int = -1);
    void        post_process(int64_t _tid, const backtrace* _bt,
                             const backtrace_metrics* _last) const;

    auto        get_cpu_timestamp() const { return m_cpu; }
    auto        get_peak_memory() const { return m_mem_peak; }
    auto        get_context_switches() const { return m_ctx_swch; }
    auto        get_page_faults() const { return m_page_flt; }
    const auto& get_hw_counters() const { return m_hw_counter; }

    void post_process_perfetto(int64_t _tid, uint64_t _ts) const;

private:
    int64_t           m_cpu        = 0;
    int64_t           m_mem_peak   = 0;
    int64_t           m_ctx_swch   = 0;
    int64_t           m_page_flt   = 0;
    hw_counter_data_t m_hw_counter = {};
};
}  // namespace component
}  // namespace omnitrace

#if !defined(OMNITRACE_EXTERN_COMPONENTS) ||                                             \
    (defined(OMNITRACE_EXTERN_COMPONENTS) && OMNITRACE_EXTERN_COMPONENTS > 0)

#    include <timemory/operations.hpp>

OMNITRACE_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_wall_clock>), true,
    double)

OMNITRACE_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_cpu_clock>), true,
    double)

OMNITRACE_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_fraction>), true,
    double)

#endif
