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
#include <thread>

namespace rocprofiler
{
namespace rocattach
{
template <typename T>
bool
wait_for(std::atomic<T>& flag, T condition, size_t timeout_ms, bool equal)
{
    auto cond_check = [&]() {
        if(equal) return flag.load() == condition;
        return flag.load() != condition;
    };
    auto start_time       = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::milliseconds(timeout_ms);
    auto end_time         = start_time + timeout_duration;
    while(std::chrono::steady_clock::now() < end_time)
    {
        if(cond_check())
        {
            return true;
        }
        std::this_thread::yield();
    }
    // Last chance check in case we were scheduled after timeout
    return cond_check();
}
// Blocks until flag is NOT equal to condition or timeout_ms milliseconds have elapsed.
// Returns true if the flag is not equal
// Returns false if timeout occurred
template <typename T>
bool
wait_for_ne(std::atomic<T>& flag, T condition, size_t timeout_ms)
{
    return wait_for(flag, condition, timeout_ms, false);
}
// Blocks until flag is equal to condition or timeout_ms milliseconds have elapsed.
// Returns true if the flag is equal
// Returns false if timeout occurred
template <typename T>
bool
wait_for_eq(std::atomic<T>& flag, T condition, size_t timeout_ms)
{
    return wait_for(flag, condition, timeout_ms, true);
}

}  // namespace rocattach
}  // namespace rocprofiler
