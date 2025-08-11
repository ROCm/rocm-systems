// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "lib/common/logging.hpp"

#include <sys/syscall.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <functional>
#include <mutex>
#include <ratio>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace rocprofiler
{
namespace common
{
template <typename... Tp>
void
consume_args(Tp&&...)
{}

uint64_t
get_clock_period_ns_impl(clockid_t _clk_id);

inline uint64_t
get_tid()
{
    // system calls are expensive so store this in a thread-local
    static thread_local uint64_t _v = ::syscall(__NR_gettid);
    return _v;
}

inline uint64_t
get_ticks(clockid_t clk_id_v) noexcept
{
    constexpr auto nanosec = std::nano::den;
    auto&&         ts      = timespec{};
    auto           ret     = clock_gettime(clk_id_v, &ts);

    if(ROCPROFILER_UNLIKELY(ret != 0))
    {
        auto _err = errno;
        ROCP_FATAL << "clock_gettime failed: " << strerror(_err);
    }

    return (static_cast<uint64_t>(ts.tv_sec) * nanosec) + static_cast<uint64_t>(ts.tv_nsec);
}

static constexpr int default_clock_id = CLOCK_BOOTTIME;

// CLOCK_MONOTONIC_RAW equates to HSA-runtime library implementation of os::ReadAccurateClock()
// CLOCK_BOOTTIME equates to HSA-runtime library implementation of os::ReadSystemClock()
template <int ClockT = default_clock_id>
inline uint64_t
timestamp_ns()
{
    constexpr auto _clk        = ClockT;
    static auto    _clk_period = get_clock_period_ns_impl(_clk);

    if(ROCPROFILER_LIKELY(_clk_period == 1)) return get_ticks(_clk);
    return get_ticks(_clk) / _clk_period;
}

// returns the process start time (in CLOCK_BOOTTIME nanoseconds) via /proc/<pid>/stat
uint64_t
get_process_start_time_ns(pid_t _pid);

std::vector<std::string>
read_command_line(pid_t _pid);

template <class Container, typename Key = typename Container::key_type>
const auto*
get_val(const Container& map, const Key& key)
{
    auto pos = map.find(key);
    return (pos != map.end() ? &pos->second : nullptr);
}

template <class Container, typename Key = typename Container::key_type>
auto*
get_val(Container& map, const Key& key)
{
    auto pos = map.find(key);
    return (pos != map.end() ? &pos->second : nullptr);
}

template <typename Tp>
constexpr void
assert_public_data_type_properties()
{
    static_assert(std::is_standard_layout<Tp>::value,
                  "public data type struct should have a standard layout");
    static_assert(std::is_trivial<Tp>::value, "public data type should be trivial");
    static_assert(std::is_default_constructible<Tp>::value,
                  "public data type struct should be default constructible");
    static_assert(std::is_trivially_copy_constructible<Tp>::value,
                  "public data type struct should be trivially copy constructible");
    static_assert(std::is_trivially_move_constructible<Tp>::value,
                  "public data type struct should be trivially move constructible");
    static_assert(std::is_trivially_copy_assignable<Tp>::value,
                  "public data type struct should be trivially move assignable");
    static_assert(std::is_trivially_move_assignable<Tp>::value,
                  "public data type struct should be trivially move assignable");
    static_assert(std::is_trivially_copyable<Tp>::value,
                  "public data type struct should be trivially move assignable");
}

template <typename Tp>
constexpr void
assert_public_api_struct_properties()
{
    assert_public_data_type_properties<Tp>();
    static_assert(std::is_class<Tp>::value, "this is not a public API struct");
    static_assert(offsetof(Tp, size) == 0, "public API struct should have a size field first");
    static_assert(sizeof(std::declval<Tp>().size) == sizeof(uint64_t),
                  "public API struct size field should be 64 bits");
}

// used to set the "size" field to the offset of the "reserved_padding" field.
// The reserved_padding field is extra unused bytes added to the a struct to
// avoid an ABI break if/when new fields are added. This is only done
// for fields which are regularly passed by value
template <typename Tp, typename Up = Tp>
constexpr auto
compute_runtime_sizeof(int) -> decltype(std::declval<Up>().reserved_padding, size_t{})
{
    return offsetof(Tp, reserved_padding);
}

template <typename Tp, typename Up = Tp>
constexpr auto
compute_runtime_sizeof(long)
{
    return sizeof(Tp);
}

template <typename Tp>
constexpr auto
compute_runtime_sizeof()
{
    return compute_runtime_sizeof<Tp>(0);
}

template <typename Tp, typename... Args>
decltype(auto)
init_public_api_struct(Tp&& val, Args&&... args)
{
    assert_public_api_struct_properties<Tp>();

    ::memset(&val, 0, sizeof(Tp));

    if constexpr(sizeof...(Args) == 0)
        val.size = compute_runtime_sizeof<Tp>();
    else
        val = {compute_runtime_sizeof<Tp>(), std::forward<Args>(args)...};

    return std::forward<Tp>(val);
}

template <typename Tp, typename... Args>
Tp&
init_public_api_struct(Tp& val, Args&&... args)
{
    assert_public_api_struct_properties<Tp>();

    ::memset(&val, 0, sizeof(Tp));

    if constexpr(sizeof...(Args) == 0)
        val.size = compute_runtime_sizeof<Tp>();
    else
        val = {compute_runtime_sizeof<Tp>(), std::forward<Args>(args)...};

    return val;
}

/**
 * A simple wrapper that will call a function when the
 * wrapper is being destroyed. This is primarily useful
 * for static variables where we want to run some destruction
 * operations when the program exits.
 */
template <typename Tp>
class static_cleanup_wrapper
{
public:
    using data_type    = Tp;
    using functor_type = std::function<void(Tp&)>;

    static_cleanup_wrapper(data_type&& data, functor_type&& destroy_func)
    : m_data(std::move(data))
    , m_destroy_func(std::move(destroy_func))
    {}

    static_cleanup_wrapper(functor_type&& destroy_func)
    : m_destroy_func(std::move(destroy_func))
    {}

    ~static_cleanup_wrapper() { m_destroy_func(m_data); }

    void destroy() { m_destroy_func(m_data); }

    data_type&       get() { return m_data; }
    const data_type& get() const { return m_data; }

private:
    data_type    m_data         = {};
    functor_type m_destroy_func = {};
};

template <typename Tp = long, typename RatioT = std::ratio<1, 1000>>
void
yield(std::chrono::duration<Tp, RatioT> duration = std::chrono::milliseconds{10})
{
    std::this_thread::yield();
    std::this_thread::sleep_for(duration);
}

template <typename PredicateT, typename Tp = long, typename RatioT = std::ratio<1, 1000>>
bool
yield(PredicateT&&                      predicate,
      std::chrono::duration<Tp, RatioT> max_yield_time,
      std::chrono::duration<Tp, RatioT> query_interval = std::chrono::milliseconds{10})
{
    auto now    = []() { return std::chrono::steady_clock::now(); };
    auto start  = now();
    auto result = false;
    while(!(result = predicate()))
    {
        yield(query_interval);
        if((now() - start) > max_yield_time)
        {
            break;
        }
    }

    // return the result of the last predicate query
    return result;
}

class assert_single_threaded
{
public:
    assert_single_threaded(std::atomic<bool>& lock)
    : m_is_initialized(lock)
    {
        bool expected = false;
        if(!m_is_initialized.compare_exchange_strong(expected, true))
        {
            ROCP_FATAL << "This code must be run in a single thread!!!";
        }
    }

    ~assert_single_threaded() { m_is_initialized.store(false, std::memory_order_release); }

private:
    std::atomic<bool>& m_is_initialized;
};

}  // namespace common
}  // namespace rocprofiler

extern "C" {
void
rocprofiler_debugger_block();
void
rocprofiler_debugger_continue();
}
