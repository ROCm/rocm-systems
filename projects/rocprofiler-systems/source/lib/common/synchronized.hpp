// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include <type_traits>

namespace rocprofsys
{
inline namespace common
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
 * synchronized<int> x(9);
 * x.rlock([](const auto& data){
 *  // data = 9
 * });
 *
 * x.wlock([](auto& data){
 *  // set data to new value
 * });
 */
template <typename LockedType, bool IsMappedTypeV = false>
class synchronized
{
public:
    using value_type = LockedType;
    using this_type  = synchronized<value_type, IsMappedTypeV>;

    synchronized()  = default;
    ~synchronized() = default;

    explicit synchronized(value_type&& data)
    : m_data{ std::move(data) }
    {}

    synchronized(synchronized&& data) noexcept            = default;
    synchronized& operator=(synchronized&& data) noexcept = default;

    // Do not allow this data structure to be copied, std::move only.
    synchronized(const synchronized&) = delete;

    template <typename FuncT, typename... Args>
    decltype(auto) rlock(FuncT&& lambda, Args&&... args) const;

    template <typename FuncT, typename... Args>
    decltype(auto) wlock(FuncT&& lambda, Args&&... args);

    // This overload to wlock allows a synchronized map whose keys map to synchronized
    // data to use a read lock on the key data and then a write lock on the mapped data.
    template <typename FuncT, typename... Args, bool EnableForMappedType = IsMappedTypeV,
              std::enable_if_t<EnableForMappedType, int> = 0>
    decltype(auto) wlock(FuncT&& lambda, Args&&... args) const;

    // Upgradable lock. If read returns false, write will be called with a unique_lock.
    // Essentially a helper function that does .rlock() followed by .wlock().
    template <typename ReadFuncT, typename WriteFuncT, typename... Args>
    bool ulock(ReadFuncT&& read, WriteFuncT&& write, Args&&... args);

private:
    mutable std::shared_mutex m_mutex = {};
    value_type                m_data  = {};
};

//
//      member definitions
//
template <typename LockedType, bool IsMappedTypeV>
template <typename FuncT, typename... Args>
decltype(auto)
synchronized<LockedType, IsMappedTypeV>::rlock(FuncT&& lambda, Args&&... args) const
{
    static_assert(std::is_invocable<FuncT, const value_type&, Args...>::value,
                  "function must accept const reference to locked type");

    auto lock = std::shared_lock{ m_mutex };
    return std::forward<FuncT>(lambda)(m_data, std::forward<Args>(args)...);
}

template <typename LockedType, bool IsMappedTypeV>
template <typename FuncT, typename... Args>
decltype(auto)
synchronized<LockedType, IsMappedTypeV>::wlock(FuncT&& lambda, Args&&... args)
{
    static_assert(std::is_invocable<FuncT, value_type&, Args...>::value,
                  "function must accept reference to locked type");

    auto lock = std::unique_lock{ m_mutex };
    return std::forward<FuncT>(lambda)(m_data, std::forward<Args>(args)...);
}

// This overload to wlock allows a synchronized map whose keys map to synchronized data to
// use a read lock on the key data and then a write lock on the mapped data.
template <typename LockedType, bool IsMappedTypeV>
template <typename FuncT, typename... Args, bool EnableForMappedType,
          std::enable_if_t<EnableForMappedType, int>>
decltype(auto)
synchronized<LockedType, IsMappedTypeV>::wlock(FuncT&& lambda, Args&&... args) const
{
    return const_cast<this_type*>(this)->wlock(std::forward<FuncT>(lambda),
                                               std::forward<Args>(args)...);
}

// Upgradable lock. If read returns false, write will be called with a unique_lock.
// Essentially a helper function that does .rlock() followed by .wlock().
template <typename LockedType, bool IsMappedTypeV>
template <typename ReadFuncT, typename WriteFuncT, typename... Args>
bool
synchronized<LockedType, IsMappedTypeV>::ulock(ReadFuncT&& read, WriteFuncT&& write,
                                               Args&&... args)
{
    static_assert(std::is_invocable<ReadFuncT, const value_type&, Args...>::value,
                  "read function must accept const reference to locked type");
    static_assert(std::is_invocable<WriteFuncT, value_type&, Args...>::value,
                  "write function must accept reference to locked type");

    using read_return_type  = std::invoke_result_t<ReadFuncT, const value_type&, Args...>;
    using write_return_type = std::invoke_result_t<WriteFuncT, value_type&, Args...>;

    static_assert(std::is_same<read_return_type, write_return_type>::value,
                  "read and write functions must return same type");
    static_assert(std::is_same<read_return_type, bool>::value,
                  "read/write functions must return bool");

    {
        auto lock = std::shared_lock{ m_mutex };
        if(read(m_data, std::forward<Args>(args)...)) return true;
    }

    auto lock = std::unique_lock{ m_mutex };
    return write(m_data, std::forward<Args>(args)...);
}
}  // namespace common
}  // namespace rocprofsys
