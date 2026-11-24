// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/spin_mutex.hpp"
#include "core/concepts.hpp"
#include "core/containers/stable_vector.hpp"

#include <cstdint>
#include <mutex>
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

// A std::mutex is not usable here: std::mutex::lock calls pthread_mutex_lock, which the
// pthread_mutex_gotcha wraps. The wrapper lazily constructs a thread_data whose
// constructor registers a grow functor, re-entering the lock and its static-init guard.
inline spin_mutex&
grow_functors_mutex()
{
    static spin_mutex s_mutex;
    return s_mutex;
}
}  // namespace detail

/// Registers a callback that grows one thread_data container. Thread-safe: distinct
/// thread_data specializations register from their own function-local static
/// initializers, which can run concurrently on different threads.
inline void
register_grow_functor(grow_functor_t callback)
{
    const std::lock_guard guard{ detail::grow_functors_mutex() };
    detail::grow_functors().emplace_back(callback);
}

/// Returns a copy of the registered callbacks. Callers invoke them without the registry
/// lock held, since a callback may initialize a thread_data and register another one.
inline std::vector<grow_functor_t>
get_grow_functors()
{
    const std::lock_guard guard{ detail::grow_functors_mutex() };
    auto                  callbacks = std::vector<grow_functor_t>{};
    for(auto itr : detail::grow_functors())
    {
        callbacks.push_back(itr);
    }
    return callbacks;
}

}  // namespace rocprofsys
