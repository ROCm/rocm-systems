// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/tool_runner.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace rocprofsys::rocsys
{

/// @brief Stable identity of a rocsys subcommand. Help order follows @ref k_verbs.
enum class verb_id : std::uint8_t
{
    profile = 0,
    avail,
    attach,
    causal,
    instrument,
    rewrite,
    python,
};

/// @brief What a verb accepts and how it is launched. Options later read these bits.
enum class capability : std::uint32_t
{
    needs_operand        = 1U << 0,  ///< At least one token after the verb.
    takes_app            = 1U << 1,  ///< That operand is a target application.
    honours_rocm_tracing = 1U << 2,  ///< May receive ROCm and runtime trace env ops.
    in_process           = 1U << 3,  ///< Stay in this process and call run_tool.
    embeds_env_in_binary = 1U << 4,  ///< Bake env into the rewritten binary.
    query_only           = 1U << 5,  ///< No target application.
};

using capability_set = std::uint32_t;

/// @brief One routed subcommand. The first row of @ref k_verbs is the implicit default.
struct verb_spec
{
    verb_id                                id          = verb_id::profile;
    std::string_view                       name        = {};
    std::string_view                       description = {};
    std::string_view                       binary_name = {};
    std::optional<common_utils::tool_mode> mode        = {};
    capability_set                         caps        = {};
};

[[nodiscard]] constexpr capability_set
operator|(capability lhs, capability rhs) noexcept
{
    return static_cast<capability_set>(lhs) | static_cast<capability_set>(rhs);
}

[[nodiscard]] constexpr capability_set
operator|(capability_set lhs, capability rhs) noexcept
{
    return lhs | static_cast<capability_set>(rhs);
}

/// @brief True when @p caps includes @p bit.
[[nodiscard]] constexpr bool
has_capability(capability_set caps, capability bit) noexcept
{
    return (caps & static_cast<capability_set>(bit)) != 0;
}

/// @brief Subcommands in help order. `profile` is first and is the implicit default.
inline constexpr auto k_verbs = std::to_array<verb_spec>({
    {
        .id          = verb_id::profile,
        .name        = "profile",
        .description = "Full trace profile (default)",
        .binary_name = "rocprof-sys-run",
        .mode        = common_utils::tool_mode::run,
        .caps        = capability::needs_operand | capability::takes_app |
                capability::honours_rocm_tracing | capability::in_process,
    },
    {
        .id          = verb_id::avail,
        .name        = "avail",
        .description = "Query available counters and settings",
        .binary_name = "rocprof-sys-avail",
        .caps        = static_cast<capability_set>(capability::query_only),
    },
    {
        .id          = verb_id::attach,
        .name        = "attach",
        .description = "Attach to a running process",
        .binary_name = "rocprof-sys-attach",
        .caps        = capability::needs_operand | capability::honours_rocm_tracing,
    },
    {
        .id          = verb_id::causal,
        .name        = "causal",
        .description = "Causal profiling",
        .binary_name = "rocprof-sys-causal",
        .caps        = capability::needs_operand | capability::takes_app,
    },
    {
        .id          = verb_id::instrument,
        .name        = "instrument",
        .description = "Runtime instrumentation profiling",
        .binary_name = "rocprof-sys-instrument",
        .caps        = capability::needs_operand | capability::takes_app |
                capability::honours_rocm_tracing,
    },
    {
        .id          = verb_id::rewrite,
        .name        = "rewrite",
        .description = "Binary rewrite application",
        .binary_name = "rocprof-sys-instrument",
        .caps        = capability::needs_operand | capability::takes_app |
                capability::honours_rocm_tracing | capability::embeds_env_in_binary,
    },
    {
        .id          = verb_id::python,
        .name        = "python",
        .description = "Python application profiling",
        .binary_name = "rocprof-sys-python",
        .caps        = capability::needs_operand | capability::takes_app |
                capability::honours_rocm_tracing,
    },
});

}  // namespace rocprofsys::rocsys
