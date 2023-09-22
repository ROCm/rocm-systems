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
#include <cstdint>

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
}  // namespace common
}  // namespace rocprofiler
