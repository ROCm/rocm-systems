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

#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace common
{
inline uint64_t
get_tid()
{
    // system calls are expensive so store this in a thread-local
    static thread_local uint64_t _v = ::syscall(__NR_gettid);
    return _v;
}

inline uint64_t
timestamp_ns()
{
    // TODO(jrmadsen): this should be updated to the HSA method
    return std::chrono::steady_clock::now().time_since_epoch().count();
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

/**
 * A simple wrapper that will call a function when the
 * wrapper is being destroyed. This is primarily useful
 * for static variables where we want to run some destruction
 * operations when the program exits.
 */
template <typename T, typename L>
class static_cleanup_wrapper
{
public:
    static_cleanup_wrapper(T&& data, L&& destroy_func)
    : _data(std::move(data))
    , _destroy_func(destroy_func)
    {}

    static_cleanup_wrapper(L&& destroy_func)
    : _destroy_func(destroy_func)
    {}

    ~static_cleanup_wrapper() { _destroy_func(_data); }

    void destroy() { _destroy_func(_data); }

    T& get() { return _data; }

private:
    T _data;
    L _destroy_func;
};

/**
 * Limits the number of active items to those set in capacity.
 * If capacity is reached, will block until another caller
 * removes active capacity.
 */
class active_capacity_gate
{
public:
    active_capacity_gate(size_t capacity)
    : _capacity(capacity)
    {}
    void add_active(size_t size)
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

    void remove_active(size_t size)
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

private:
    size_t                  _count{0};
    size_t                  _capacity{0};
    size_t                  _waiters{0};
    std::mutex              _m;
    std::condition_variable _cv;
};

}  // namespace common
}  // namespace rocprofiler
