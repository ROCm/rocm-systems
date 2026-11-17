// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/concepts.hpp"
#include "core/containers/stable_vector.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

namespace rocprofsys
{
using grow_functor_t = std::int64_t (*)(std::int64_t);

namespace detail
{
inline auto&
grow_functors()
{
    static auto s_functors = container::stable_vector<grow_functor_t>{};
    return s_functors;
}

inline std::atomic_flag&
grow_functors_lock()
{
    static std::atomic_flag s_lock;
    return s_lock;
}

// A std::mutex is not usable here: std::mutex::lock calls pthread_mutex_lock, which the
// pthread_mutex_gotcha wraps. The wrapper lazily constructs a thread_data whose
// constructor registers a grow functor, re-entering the lock and its static-init guard.
class [[nodiscard]] grow_functors_guard
{
public:
    grow_functors_guard()
    {
        while(grow_functors_lock().test_and_set(std::memory_order_acquire))
        {
        }
    }

    ~grow_functors_guard() { grow_functors_lock().clear(std::memory_order_release); }

    grow_functors_guard(const grow_functors_guard&)            = delete;
    grow_functors_guard& operator=(const grow_functors_guard&) = delete;
    grow_functors_guard(grow_functors_guard&&)                 = delete;
    grow_functors_guard& operator=(grow_functors_guard&&)      = delete;
};
}  // namespace detail

/// Registers a callback that grows one thread_data container. Thread-safe: distinct
/// thread_data specializations register from their own function-local static
/// initializers, which can run concurrently on different threads.
inline void
register_grow_functor(grow_functor_t callback)
{
    const detail::grow_functors_guard guard;
    detail::grow_functors().emplace_back(callback);
}

/// Returns a copy of the registered callbacks. Callers invoke them without the registry
/// lock held, since a callback may initialize a thread_data and register another one.
inline std::vector<grow_functor_t>
get_grow_functors()
{
    const detail::grow_functors_guard guard;
    auto                              callbacks = std::vector<grow_functor_t>{};
    for(auto itr : detail::grow_functors())
    {
        callbacks.push_back(itr);
    }
    return callbacks;
}

}  // namespace rocprofsys
