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

}  // namespace rocprofsys
