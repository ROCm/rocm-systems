// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>

namespace rocprofsys
{
namespace type_traits
{
// Structural concept, deliberately independent of any concrete state type
// (e.g. rocprofsys::state::thread) so that lower-level code depending on this
// header never needs to include a higher-level state module.
template <typename T>
concept thread_state_policy = requires {
    typename T::State;
    { T::Internal } -> std::convertible_to<typename T::State>;
    { T::scoped(std::declval<typename T::State>()) } -> std::destructible;
};
}  // namespace type_traits
}  // namespace rocprofsys
