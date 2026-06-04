// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "profiler-hub/writer_types.hpp"

#include "spdlog/fmt/bundled/core.h"

#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace profiler_hub
{

template <typename T>
[[nodiscard]] std::string
to_error_string(const std::optional<T>& opt)
{
    return opt.has_value() ? std::to_string(opt.value()) : "[NULL]";
}

[[nodiscard]] inline bool
is_valid_string_view(std::string_view str) noexcept
{
    return !str.empty();
}

[[nodiscard]] inline bool
is_valid_optional_string_view(std::optional<std::string_view> str) noexcept
{
    return str.has_value() && !str->empty();
}

// ============================================================================
// get_key() overloads
// ============================================================================

[[nodiscard]] inline auto
get_key(const writer_types::node_info_t& e) noexcept
{
    return e.node_id;
}

[[nodiscard]] inline auto
get_key(const writer_types::process_info_t& e) noexcept
{
    return e.pid;
}

[[nodiscard]] inline auto
get_key(const writer_types::thread_info_t& e) noexcept
{
    return e.thread_id;
}

[[nodiscard]] inline const writer_types::agent_unique_id_t&
get_key(const writer_types::agent_info_t& e) noexcept
{
    return e.unique_id;
}

[[nodiscard]] inline const writer_types::pmc_info_unique_id_t&
get_key(const writer_types::pmc_info_t& e) noexcept
{
    return e.unique_id;
}

[[nodiscard]] inline auto
get_key(const writer_types::stream_info_t& e) noexcept
{
    return e.stream_id;
}

[[nodiscard]] inline auto
get_key(const writer_types::queue_info_t& e) noexcept
{
    return e.queue_id;
}

[[nodiscard]] inline auto
get_key(const writer_types::code_object_info_t& e) noexcept
{
    return e.id;
}

[[nodiscard]] inline auto
get_key(const writer_types::kernel_symbol_info_t& e) noexcept
{
    return e.id;
}

[[nodiscard]] inline const writer_types::track_info_t&
get_key(const writer_types::track_info_t& e) noexcept
{
    return e;
}

// ============================================================================
// to_string()
// ============================================================================

template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
[[nodiscard]] inline std::string
to_string(T value)
{
    return std::to_string(value);
}

[[nodiscard]] inline std::string
to_string(const writer_types::agent_unique_id_t& id)
{
    return fmt::format(
        "agent_type={}, type_index={}", id.agent_type.value_or("[NULL]"), id.type_index);
}

[[nodiscard]] inline std::string
to_string(const std::optional<writer_types::agent_unique_id_t>& id)
{
    return id.has_value() ? to_string(*id) : "[NULL]";
}

[[nodiscard]] inline std::string
to_string(const writer_types::pmc_info_unique_id_t& id)
{
    if(id.agent_id.has_value())
    {
        return fmt::format(
            "[name={}, agent_id={}]", id.name, to_string(id.agent_id.value()));
    }
    return fmt::format("[name={}]", id.name);
}

[[nodiscard]] inline std::string
to_string(const writer_types::node_info_t& e)
{
    return fmt::format("[node_info] node_id: {}", e.node_id);
}

[[nodiscard]] inline std::string
to_string(const writer_types::process_info_t& e)
{
    return fmt::format("[process_info] pid: {}", e.pid);
}

[[nodiscard]] inline std::string
to_string(const writer_types::thread_info_t& e)
{
    return fmt::format("[thread_info] thread_id: {}", e.thread_id);
}

[[nodiscard]] inline std::string
to_string(const writer_types::agent_info_t& e)
{
    return fmt::format(
        "[agent_info]  name: {}, {}", e.name.value_or(""), to_string(e.unique_id));
}

[[nodiscard]] inline std::string
to_string(const writer_types::pmc_info_t& e)
{
    return fmt::format(
        "[pmc_info] name: {}, {}", e.unique_id.name, to_string(e.unique_id.agent_id));
}

[[nodiscard]] inline std::string
to_string(const writer_types::stream_info_t& e)
{
    return fmt::format("[stream_info] stream_id: {}", e.stream_id);
}

[[nodiscard]] inline std::string
to_string(const writer_types::queue_info_t& e)
{
    return fmt::format("[queue_info] queue_id: {}", e.queue_id);
}

[[nodiscard]] inline std::string
to_string(const writer_types::code_object_info_t& e)
{
    return fmt::format("[code_object_info] id: {}", e.id);
}

[[nodiscard]] inline std::string
to_string(const writer_types::kernel_symbol_info_t& e)
{
    return fmt::format("[kernel_symbol_info] id: {}", e.id);
}

[[nodiscard]] inline std::string
to_string(const writer_types::track_info_t& e)
{
    return fmt::format(
        "[track_info] node_id: {}, process_id: {}, thread_id: {}, name: {}",
        e.node_id,
        to_error_string(e.process_id),
        to_error_string(e.thread_id),
        e.name.value_or("NULL"));
}

// Handle optional types - must be after all specific overloads due to two-phase lookup
template <typename T>
[[nodiscard]] inline std::string
to_string(const std::optional<T>& opt)
{
    return opt.has_value() ? to_string(opt.value()) : "[NULL]";
}

}  // namespace profiler_hub
