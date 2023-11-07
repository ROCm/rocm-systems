// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include <glog/logging.h>

#include <sys/syscall.h>
#include <sys/utsname.h>
#include <unistd.h>
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
#include <type_traits>
#include <vector>

namespace rocprofiler
{
namespace common
{
clockid_t
get_accurate_clock_id_impl();

uint64_t
get_clock_freq_ns_impl(clockid_t _clk_id);

inline uint64_t
get_tid()
{
    // system calls are expensive so store this in a thread-local
    static thread_local uint64_t _v = ::syscall(__NR_gettid);
    return _v;
}

inline clockid_t
get_accurate_clock_id()
{
    static auto clk_id = get_accurate_clock_id_impl();
    return clk_id;
}

inline uint64_t
get_accurate_clock_freq_ns()
{
    static auto clk_freq = get_clock_freq_ns_impl(get_accurate_clock_id());
    return clk_freq;
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
        LOG(FATAL) << "clock_gettime failed: " << strerror(_err);
    }

    return (static_cast<uint64_t>(ts.tv_sec) * nanosec) + static_cast<uint64_t>(ts.tv_nsec);
}

// this equates to HSA-runtime library implementation of os::ReadAccurateClock()
inline uint64_t
timestamp_ns()
{
    return get_ticks(get_accurate_clock_id()) * get_accurate_clock_freq_ns();
}

// this equates to HSA-runtime library implementation of os::ReadSystemClock()
inline uint64_t
system_timestamp_ns()
{
    constexpr auto boottime_clk      = CLOCK_BOOTTIME;
    static auto    boottime_clk_freq = get_clock_freq_ns_impl(boottime_clk);

    return get_ticks(boottime_clk) * boottime_clk_freq;
}

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
assert_public_api_struct_properties()
{
    static_assert(std::is_class<Tp>::value, "this is not a public API struct");
    static_assert(std::is_standard_layout<Tp>::value,
                  "public API struct should have a standard layout");
    static_assert(std::is_trivially_default_constructible<Tp>::value,
                  "public API struct should be trivially default constructible");
    static_assert(std::is_trivially_copy_constructible<Tp>::value,
                  "public API struct should be trivially copy constructible");
    static_assert(std::is_trivially_move_constructible<Tp>::value,
                  "public API struct should be trivially move constructible");
    static_assert(std::is_trivially_copy_assignable<Tp>::value,
                  "public API struct should be trivially move assignable");
    static_assert(std::is_trivially_move_assignable<Tp>::value,
                  "public API struct should be trivially move assignable");
    static_assert(std::is_trivially_copyable<Tp>::value,
                  "public API struct should be trivially move assignable");
    static_assert(std::is_trivial<Tp>::value, "public API struct should be trivial");
    static_assert(offsetof(Tp, size) == 0, "public API struct should have a size field first");
    static_assert(sizeof(std::declval<Tp>().size) == sizeof(uint64_t),
                  "public API struct size field should be 64 bits");
}

template <typename Tp>
decltype(auto)
init_public_api_struct(Tp&& val)
{
    assert_public_api_struct_properties<Tp>();

    ::memset(&val, 0, sizeof(Tp));
    val.size = sizeof(Tp);
    return std::forward<Tp>(val);
}

template <typename Tp>
Tp&
init_public_api_struct(Tp& val)
{
    assert_public_api_struct_properties<Tp>();

    ::memset(&val, 0, sizeof(Tp));
    val.size = sizeof(Tp);
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

/**
 * Limits the number of active items to those set in capacity.
 * If capacity is reached, will block until another caller
 * removes active capacity.
 */
class active_capacity_gate
{
public:
    active_capacity_gate(size_t capacity);

    void add_active(size_t size);
    void remove_active(size_t size);

private:
    size_t                  _count{0};
    size_t                  _capacity{0};
    size_t                  _waiters{0};
    std::mutex              _m;
    std::condition_variable _cv;
};

inline active_capacity_gate::active_capacity_gate(size_t capacity)
: _capacity(capacity)
{}

inline void
active_capacity_gate::add_active(size_t size)
{
    if(size >= _capacity)
    {
        throw std::runtime_error("Size exceeds gate capacity");
    }

    std::unique_lock lock(_m);
    if(_count + size < _capacity)
    {
        _count += size;
        return;
    }
    _waiters++;
    _cv.wait(lock, [&]() { return _count + size < _capacity; });
    _waiters--;
    _count += size;
}

inline void
active_capacity_gate::remove_active(size_t size)
{
    std::unique_lock lock(_m);
    if(_count > size)
        _count -= size;
    else
        _count = 0;

    if(_waiters > 0)
    {
        _cv.notify_all();
    }
}

}  // namespace common
}  // namespace rocprofiler
