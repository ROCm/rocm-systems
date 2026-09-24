// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace rocprofsys::avail
{

/// Listing-only Externals for tracing_config. Avoids core/config.hpp so the catalog
/// can enumerate domains and operations without configure_settings().
struct listing_externals
{
    // Names must match policies::tracing_config::externals and state::process.
    struct ProcessState  // NOLINT(readability-identifier-naming)
    {
        enum class State  // NOLINT(readability-identifier-naming)
        {
            finalized
        };

        // NOLINTNEXTLINE(readability-identifier-naming)
        static constexpr State Finalized = State::finalized;

        static void set(State) {}
    };

    static bool        get_use_rcclp() { return false; }
    static bool        get_use_ompt() { return false; }
    static bool        get_use_unified_memory_profiling() { return false; }
    static std::string get_rocm_domains() { return {}; }

    static std::optional<std::string> get_setting_value(std::string_view)
    {
        return std::nullopt;
    }
};

}  // namespace rocprofsys::avail
