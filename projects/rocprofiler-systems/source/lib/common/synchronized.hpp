// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
        requires std::is_invocable_v<FuncT, const LockedType&, Args...>
    decltype(auto) rlock(FuncT&& lambda, Args&&... args) const;

    template <typename FuncT, typename... Args>
        requires std::is_invocable_v<FuncT, LockedType&, Args...>
    decltype(auto) wlock(FuncT&& lambda, Args&&... args);

    // This overload to wlock allows a synchronized map whose keys map to synchronized
    // data to use a read lock on the key data and then a write lock on the mapped data.
    template <typename FuncT, typename... Args>
        requires(IsMappedTypeV)
    decltype(auto) wlock(FuncT&& lambda, Args&&... args) const;

    // Upgradable lock. If read returns false, write will be called with a unique_lock.
    // Essentially a helper function that does .rlock() followed by .wlock().
    template <typename ReadFuncT, typename WriteFuncT, typename... Args>
        requires(std::is_invocable_v<ReadFuncT, const LockedType&, Args...> &&
                 std::is_invocable_v<WriteFuncT, LockedType&, Args...>)
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
    requires std::is_invocable_v<FuncT, const LockedType&, Args...>
decltype(auto)
synchronized<LockedType, IsMappedTypeV>::rlock(FuncT&& lambda, Args&&... args) const
{
    auto lock = std::shared_lock{ m_mutex };
    return std::forward<FuncT>(lambda)(m_data, std::forward<Args>(args)...);
}

template <typename LockedType, bool IsMappedTypeV>
template <typename FuncT, typename... Args>
    requires std::is_invocable_v<FuncT, LockedType&, Args...>
decltype(auto)
synchronized<LockedType, IsMappedTypeV>::wlock(FuncT&& lambda, Args&&... args)
{
    auto lock = std::unique_lock{ m_mutex };
    return std::forward<FuncT>(lambda)(m_data, std::forward<Args>(args)...);
}

// This overload to wlock allows a synchronized map whose keys map to synchronized data to
// use a read lock on the key data and then a write lock on the mapped data.
template <typename LockedType, bool IsMappedTypeV>
template <typename FuncT, typename... Args>
    requires(IsMappedTypeV)
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
    requires(std::is_invocable_v<ReadFuncT, const LockedType&, Args...> &&
             std::is_invocable_v<WriteFuncT, LockedType&, Args...>)
bool
synchronized<LockedType, IsMappedTypeV>::ulock(ReadFuncT&& read, WriteFuncT&& write,
                                               Args&&... args)
{
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
