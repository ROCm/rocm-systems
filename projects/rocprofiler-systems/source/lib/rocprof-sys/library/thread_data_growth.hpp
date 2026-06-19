// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/concepts.hpp"
#include "core/containers/stable_vector.hpp"

#include <cstdint>
#include <functional>

namespace rocprofsys
{
using grow_functor_t = std::int64_t (*)(std::int64_t);

inline auto&
grow_functors()
{
    static auto _v = container::stable_vector<grow_functor_t>{};
    return _v;
}

inline auto&
get_peak_num_threads_callback()
{
    static std::function<std::int64_t()> _v = []() -> std::int64_t {
        return static_cast<std::int64_t>(max_supported_threads);
    };
    return _v;
}

inline std::int64_t
get_current_peak_num_threads()
{
    return get_peak_num_threads_callback()();
}

inline void
set_peak_num_threads_callback(std::function<std::int64_t()> _cb)
{
    get_peak_num_threads_callback() = std::move(_cb);
}

}  // namespace rocprofsys
