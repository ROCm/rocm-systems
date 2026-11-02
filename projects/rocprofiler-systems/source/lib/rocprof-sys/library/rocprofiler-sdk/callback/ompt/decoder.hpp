// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common_types.hpp"

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace rocprofsys::domains::callback::ompt
{

/**
 * One bit of an OMPT flags field and its textual form. Masks are unsigned because
 * the OMPT spec defines bit 31 (e.g. ompt_task_merged) and the payload stores flags
 * in an `int`.
 */
struct flag_bit
{
    std::uint32_t    mask;
    std::string_view value;
};

using flag_table = std::span<const flag_bit>;

inline constexpr std::uint32_t k_task_initial = 0x00000001;

[[nodiscard]] constexpr bool
has_flag(int flags, std::uint32_t mask) noexcept
{
    return (static_cast<std::uint32_t>(flags) & mask) != 0;
}

// Values follow the ompt_parallel_flag_t / ompt_task_flag_t / ompt_cancel_flag_t
// definitions in the OpenMP 5.x specification (omp-tools.h).
inline constexpr std::array k_parallel_invoker{
    flag_bit{ .mask = 0x00000001, .value = "program" },
    flag_bit{ .mask = 0x00000002, .value = "runtime" },
};

inline constexpr std::array k_parallel_cause{
    flag_bit{ .mask = 0x40000000, .value = "teams_construct" },
    flag_bit{ .mask = 0x80000000, .value = "parallel_construct" },
};

inline constexpr std::array k_task_classification{
    flag_bit{ .mask = k_task_initial, .value = "initial" },
    flag_bit{ .mask = 0x00000002, .value = "implicit" },
    flag_bit{ .mask = 0x00000004, .value = "explicit" },
    flag_bit{ .mask = 0x00000008, .value = "target" },
};

inline constexpr std::array k_task_properties{
    flag_bit{ .mask = 0x08000000, .value = "undeferred" },
    flag_bit{ .mask = 0x10000000, .value = "untied" },
    flag_bit{ .mask = 0x20000000, .value = "final" },
    flag_bit{ .mask = 0x40000000, .value = "mergeable" },
    flag_bit{ .mask = 0x80000000, .value = "merged" },
};

inline constexpr std::array k_implicit_task_kind{
    flag_bit{ .mask = 0x00000002, .value = "implicit" },
};

inline constexpr std::array k_cancel_construct{
    flag_bit{ .mask = 0x01, .value = "parallel" },
    flag_bit{ .mask = 0x02, .value = "sections" },
    flag_bit{ .mask = 0x04, .value = "loop" },
    flag_bit{ .mask = 0x08, .value = "taskgroup" },
};

inline constexpr std::array k_cancel_state{
    flag_bit{ .mask = 0x10, .value = "activated" },
    flag_bit{ .mask = 0x20, .value = "detected" },
    flag_bit{ .mask = 0x40, .value = "discarded_task" },
};

[[nodiscard]] constexpr auto
is_set(int flags) noexcept
{
    return [flags](const flag_bit& bit) { return has_flag(flags, bit.mask); };
}

// Precedence follows table order, so the first matching bit wins.
[[nodiscard]] inline std::optional<std::string_view>
first_match(int flags, flag_table table) noexcept
{
    const auto match = std::ranges::find_if(table, is_set(flags));
    return match == table.end() ? std::nullopt
                                : std::optional<std::string_view>{ match->value };
}

// Every matching bit, ", "-joined; "none" when no bit is set.
[[nodiscard]] inline std::string
all_matches(int flags, flag_table table)
{
    auto values = table | std::views::filter(is_set(flags)) |
                  std::views::transform(&flag_bit::value);

    const auto joined = fmt::format("{}", fmt::join(values, ", "));
    return joined.empty() ? "none" : joined;
}

// Appends decoded flag arguments of one flags field, all sharing `type`.
class flag_arg_writer
{
public:
    flag_arg_writer(function_args_t& args, int flags, std::string_view type) noexcept
    : m_args{ args }
    , m_flags{ flags }
    , m_type{ type }
    {}

    // Emits nothing when no bit of `table` is set.
    void first_match(std::string_view name, flag_table table) const
    {
        if(const auto value = ompt::first_match(m_flags, table))
        {
            append(name, std::string{ *value });
        }
    }

    void all_matches(std::string_view name, flag_table table) const
    {
        append(name, ompt::all_matches(m_flags, table));
    }

private:
    void append(std::string_view name, std::string value) const
    {
        m_args.emplace_back(
            argument_info{ .arg_number = static_cast<std::uint32_t>(m_args.size()),
                           .arg_type   = std::string{ m_type },
                           .arg_name   = std::string{ name },
                           .arg_value  = std::move(value) });
    }

    function_args_t& m_args;
    int              m_flags;
    std::string_view m_type;
};

/**
 * Appends the decoded flag arguments of `operation` to `args`. Operations without a
 * flags field leave `args` untouched.
 */
template <policies::domain_service::backend SdkBackend>
void
append_flag_args(function_args_t& args, typename SdkBackend::ompt_operation_t operation,
                 const typename SdkBackend::callback_tracing_ompt_data_t& payload)
{
    constexpr std::string_view k_parallel_type = "ompt_parallel_flag_t";
    constexpr std::string_view k_task_type     = "ompt_task_flag_t";
    constexpr std::string_view k_cancel_type   = "ompt_cancel_flag_t";

    const auto append_parallel = [&](int flags) {
        const flag_arg_writer writer{ args, flags, k_parallel_type };
        writer.first_match("invoker", k_parallel_invoker);
        writer.first_match("invoker_cause", k_parallel_cause);
    };

    switch(operation)
    {
        case SdkBackend::OMPT_ID_parallel_begin:
            append_parallel(payload.args.parallel_begin.flags);
            break;

        case SdkBackend::OMPT_ID_parallel_end:
            append_parallel(payload.args.parallel_end.flags);
            break;

        case SdkBackend::OMPT_ID_task_create:
        {
            const flag_arg_writer writer{ args, payload.args.task_create.flags,
                                          k_task_type };
            writer.first_match("classification", k_task_classification);
            writer.all_matches("properties", k_task_properties);
            break;
        }

        case SdkBackend::OMPT_ID_implicit_task:
        {
            const flag_arg_writer writer{ args, payload.args.implicit_task.flags,
                                          "flags" };
            writer.first_match("kind", k_implicit_task_kind);
            break;
        }

        case SdkBackend::OMPT_ID_cancel:
        {
            const flag_arg_writer writer{ args, payload.args.cancel.flags,
                                          k_cancel_type };
            writer.first_match("construct", k_cancel_construct);
            writer.first_match("state", k_cancel_state);
            break;
        }

        default: break;
    }
}

}  // namespace rocprofsys::domains::callback::ompt
