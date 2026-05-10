// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/timemory.hpp"

#include <timemory/components/gotcha/backends.hpp>
#include <timemory/mpl/macros.hpp>

#include <array>
#include <cstddef>
#include <string>

namespace rocprofsys
{
namespace causal
{
namespace component
{
struct unblocking_gotcha : comp::base<unblocking_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 9;

    enum indexes
    {
        pthread_barrier_wait_idx = 7,
        kill_idx                 = 8,
        indexes_max              = gotcha_capacity,
    };

    template <size_t Idx>
    using gotcha_index = std::integral_constant<size_t, Idx>;

    // string id for component
    static std::string label();
    static std::string description();
    static void        preinit();

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    template <size_t Idx, typename Ret, typename... Args>
    std::enable_if_t<(Idx < kill_idx), Ret> operator()(gotcha_index<Idx>,
                                                       Ret (*)(Args...),
                                                       Args...) const noexcept;

    int operator()(gotcha_index<kill_idx>, int (*)(pid_t, int), pid_t,
                   int) const noexcept;
};

using unblocking_gotcha_t =
    comp::gotcha<unblocking_gotcha::gotcha_capacity, tim::type_list<>, unblocking_gotcha>;
}  // namespace component
}  // namespace causal
}  // namespace rocprofsys

ROCPROFSYS_DEFINE_CONCRETE_TRAIT(prevent_reentry, causal::component::unblocking_gotcha_t,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(static_data, causal::component::unblocking_gotcha_t,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(fast_gotcha, causal::component::unblocking_gotcha_t,
                                 true_type)
