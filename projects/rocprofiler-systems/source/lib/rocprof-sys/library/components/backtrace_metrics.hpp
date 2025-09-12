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
#include "library/components/backtrace.hpp"
#include "library/thread_data.hpp"

#include <timemory/components/base.hpp>
#include <timemory/components/papi/papi_array.hpp>
#include <timemory/components/papi/types.hpp>
#include <timemory/macros/language.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/utility/type_list.hpp>
#include <timemory/variadic/types.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace rocprofsys
{
template <typename... Tp>
using type_list = ::tim::type_list<Tp...>;

namespace component
{
struct backtrace_metrics : comp::empty_base
{
    static constexpr size_t num_hw_counters = TIMEMORY_PAPI_ARRAY_SIZE;

    using clock_type        = std::chrono::steady_clock;
    using value_type        = void;
    using hw_counters       = tim::component::papi_array<num_hw_counters>;
    using hw_counter_data_t = typename hw_counters::value_type;
    using system_clock      = std::chrono::system_clock;
    using system_time_point = typename system_clock::time_point;

    using categories_t =
        type_list<category::thread_cpu_time, category::thread_peak_memory,
                  category::thread_context_switch, category::thread_page_fault,
                  category::thread_hardware_counter, hw_counters>;
    static constexpr size_t num_categories = std::tuple_size<categories_t>::value;
    using valid_array_t                    = std::bitset<num_categories>;

    static std::string label();
    static std::string description();

    backtrace_metrics()                             = default;
    ~backtrace_metrics()                            = default;
    backtrace_metrics(const backtrace_metrics&)     = default;
    backtrace_metrics(backtrace_metrics&&) noexcept = default;

    backtrace_metrics& operator=(const backtrace_metrics&)     = default;
    backtrace_metrics& operator=(backtrace_metrics&&) noexcept = default;

    static void                     configure(bool, int64_t _tid = threading::get_id());
    static void                     init_perfetto(int64_t _tid, valid_array_t);
    static void                     fini_perfetto(int64_t _tid, valid_array_t);
    static void                     init_rocpd(int64_t _tid, valid_array_t);
    static void                     fini_rocpd(int64_t _tid, valid_array_t);
    static std::vector<std::string> get_hw_counter_labels(int64_t);

    template <typename Tp>
    static bool get_valid(Tp, valid_array_t);

    template <typename Tp>
    static bool get_valid(type_list<Tp>, valid_array_t);

    static void start();
    static void stop();
    void        sample(int = -1);
    void        post_process(int64_t _tid, const backtrace* _bt,
                             const backtrace_metrics* _last) const;

    explicit operator bool() const { return m_valid.any(); }

    template <typename Tp>
    bool operator()(Tp) const;

    template <typename Tp>
    bool operator()(type_list<Tp>) const;

    auto        get_valid() const { return m_valid; }
    auto        get_cpu_timestamp() const { return m_cpu; }
    auto        get_peak_memory() const { return m_mem_peak; }
    auto        get_context_switches() const { return m_ctx_swch; }
    auto        get_page_faults() const { return m_page_flt; }
    const auto& get_hw_counters() const { return m_hw_counter; }

    void post_process_perfetto(int64_t _tid, uint64_t _ts) const;
    void post_process_rocpd(int64_t _tid, uint64_t _ts) const;

    backtrace_metrics& operator-=(const backtrace_metrics&);

    friend backtrace_metrics operator-(backtrace_metrics        _lhs,
                                       const backtrace_metrics& _rhs)
    {
        return (_lhs -= _rhs);
    }

private:
    valid_array_t     m_valid      = {};
    int64_t           m_cpu        = 0;
    int64_t           m_mem_peak   = 0;
    int64_t           m_ctx_swch   = 0;
    int64_t           m_page_flt   = 0;
    hw_counter_data_t m_hw_counter = {};
};

template <typename Tp>
bool
backtrace_metrics::get_valid(type_list<Tp>, valid_array_t _valid)
{
    constexpr auto idx = tim::index_of<Tp, categories_t>::value;
    return _valid.test(idx);
}

template <typename Tp>
bool
backtrace_metrics::operator()(type_list<Tp>) const
{
    static_assert(!concepts::is_type_listing<Tp>::value,
                  "Error! invalid call with tuple");

    constexpr auto idx = tim::index_of<Tp, categories_t>::value;
    return m_valid.test(idx);
}

template <typename Tp>
bool
backtrace_metrics::get_valid(Tp, valid_array_t _valid)
{
    return get_valid(type_list<Tp>{}, _valid);
}

template <typename Tp>
bool
backtrace_metrics::operator()(Tp) const
{
    return (*this)(type_list<Tp>{});
}
}  // namespace component
}  // namespace rocprofsys

#if !defined(ROCPROFSYS_EXTERN_COMPONENTS) ||                                            \
    (defined(ROCPROFSYS_EXTERN_COMPONENTS) && ROCPROFSYS_EXTERN_COMPONENTS > 0)

#    include <timemory/operations.hpp>

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_wall_clock>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_cpu_clock>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_fraction>), true,
    double)

#endif
