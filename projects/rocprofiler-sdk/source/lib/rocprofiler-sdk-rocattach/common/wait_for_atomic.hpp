// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <type_traits>

namespace rocprofiler
{
namespace rocattach
{
// Blocks until predicate(flag) == true or timeout_ms milliseconds have elapsed.
// Returns true if predicate(flag) was true
// Returns false if timeout occurred
template <typename Tp, typename PredicateT>
bool
wait_for(std::atomic<Tp>& flag, size_t timeout_ms, PredicateT&& predicate)
{
    static_assert(std::is_invocable<PredicateT, std::atomic<Tp>&>::value, "Invalid predicate");
    using predicate_return_type = typename std::invoke_result<PredicateT, std::atomic<Tp>&>::type;
    static_assert(std::is_same<predicate_return_type, bool>::value,
                  "Predicate must return boolean");

    auto start_time       = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::milliseconds(timeout_ms);
    auto end_time         = start_time + timeout_duration;
    while(std::chrono::steady_clock::now() < end_time)
    {
        if(std::invoke(std::forward<PredicateT>(predicate), std::forward<std::atomic<Tp>&>(flag)))
        {
            return true;
        }
        std::this_thread::yield();
    }
    // Last chance check in case we were scheduled after timeout
    return std::invoke(std::forward<PredicateT>(predicate), std::forward<std::atomic<Tp>&>(flag));
}
// Blocks until flag is NOT equal to value or timeout_ms milliseconds have elapsed.
// Returns true if the flag is not equal
// Returns false if timeout occurred
template <typename T>
bool
wait_for_ne(std::atomic<T>& flag, T value, size_t timeout_ms)
{
    auto predicate = [value](std::atomic<T>& a) { return a.load() != value; };
    return wait_for(flag, timeout_ms, predicate);
}
// Blocks until flag is equal to value or timeout_ms milliseconds have elapsed.
// Returns true if the flag is equal
// Returns false if timeout occurred
template <typename T>
bool
wait_for_eq(std::atomic<T>& flag, T value, size_t timeout_ms)
{
    auto predicate = [value](std::atomic<T>& a) { return a.load() == value; };
    return wait_for(flag, timeout_ms, predicate);
}
}  // namespace rocattach
}  // namespace rocprofiler
