// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <optional>
#include <string>
#include <string_view>

namespace rocprofsys::policies::tracing_config
{

/// @brief External dependencies required by rocprofsys::rocprofiler_sdk::tracing_config:
/// process-state transitions, feature toggles, and settings lookup. Production code
/// satisfies this via rocprofsys::rocprofiler_sdk::default_externals; tests substitute
/// a mock.
template <typename Externals>
concept externals = requires(std::string_view setting_name) {
    typename Externals::ProcessState;
    typename Externals::ProcessState::State;
    {
        Externals::ProcessState::Finalized
    } -> std::convertible_to<typename Externals::ProcessState::State>;
    { Externals::ProcessState::set(Externals::ProcessState::Finalized) };
    { Externals::get_use_rcclp() } -> std::convertible_to<bool>;
    { Externals::get_use_ompt() } -> std::convertible_to<bool>;
    { Externals::get_use_unified_memory_profiling() } -> std::convertible_to<bool>;
    { Externals::get_rocm_domains() } -> std::convertible_to<std::string>;
    {
        Externals::get_setting_value(setting_name)
    } -> std::same_as<std::optional<std::string>>;
};

}  // namespace rocprofsys::policies::tracing_config
