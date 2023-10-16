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

#include <cstddef>
#include <functional>
#include <mutex>
#include <shared_mutex>

namespace rocprofiler
{
namespace common
{
/**
 * Sychronized is a wrapper that adds lock based write/read
 * protection around a datatype. The protected data is accessed
 * only by rlock/wlock. rlock(lambda) gets a reader lock of the
 * protected value, passing the protected value to the lambda as a
 * const. wlock(lambda) gets a writer lock on the protective value
 * and does the same. The reason for this class is to make it less
 * error prone to access shared data and more obvious when a lock
 * is being held.
 *
 * Example usage:
 *
 * Synchronized<int> x(9);
 * x.rlock([](const auto& data){
 *  // data = 9
 * });
 *
 * x.wlock([](auto& data){
 *  // set data to new value
 * });
 */
template <typename LockedType>
class Synchronized
{
public:
    Synchronized() = default;
    Synchronized(LockedType&& data)
    : data_(std::move(data))
    {}
    // Do not allow this data structure to be copied, std::move only.
    Synchronized(const Synchronized&) = delete;

    void rlock(std::function<void(const LockedType&)> lambda) const
    {
        std::shared_lock lock(mutex_);
        lambda(data_);
    }

    void wlock(std::function<void(LockedType&)> lambda)
    {
        std::unique_lock lock(mutex_);
        lambda(data_);
    }

    // Upgradable lock. If read returns false, write will be called with a unique_lock.
    // Essentially a helper function that does .rlock() followed by .wlock().
    void ulock(std::function<bool(const LockedType&)> read, std::function<bool(LockedType&)> write)
    {
        {
            std::shared_lock lock(mutex_);
            if(read(data_)) return;
        }

        std::unique_lock lock(mutex_);
        write(data_);
    }

private:
    mutable std::shared_mutex mutex_;
    LockedType                data_;
};
}  // namespace common
}  // namespace rocprofiler
