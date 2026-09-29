// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/callback/ompt/rules.hpp"

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rocprofsys::domains::callback::ompt
{

template <typename EnumT>
constexpr auto
to_underlying(EnumT value)
{
    if constexpr(std::is_enum_v<EnumT>)
    {
        return static_cast<std::underlying_type_t<EnumT>>(value);
    }
    else
    {
        return value;
    }
}

template <typename FlagT>
constexpr bool
has_flag(int flags_val, FlagT flag)
{
    return (flags_val & to_underlying(flag)) != 0;
}

/**
 * A single decoded flag argument, ready to be surfaced to callers as a
 * type/key/value triple (e.g. for `argument_info`).
 */
struct decoded_flag_arg
{
    std::string_view type;
    std::string_view key;
    std::string      value;
};

/**
 * Normalized flag data and its associated decoding rules.
 *
 * `FlagT` is the raw flag storage type.
 *
 * `rules` is a non-owning view; callers must ensure it refers to storage that
 * outlives this object. Static constexpr rule arrays are the intended source.
 */
template <typename FlagT = int>
class flag_representation
{
public:
    using flag_type = FlagT;
    using rule_type = rules::flag_rule<flag_type>;

    constexpr flag_representation(flag_type                  flags,
                                  std::span<const rule_type> rules) noexcept
    : m_flags{ flags }
    , m_rules{ rules }
    {}

    /**
     * Decodes the raw flag bits into one `decoded_flag_arg` per decoded
     * output argument.
     */
    [[nodiscard]] std::vector<decoded_flag_arg> decode() const
    {
        std::vector<decoded_flag_arg> result;

        for(auto group : groups())
        {
            if(auto decoded = decode_group(group))
            {
                result.push_back(*std::move(decoded));
            }
        }

        return result;
    }

private:
    using group_type = std::span<const rule_type>;

    [[nodiscard]] static bool same_group(const rule_type& lhs,
                                         const rule_type& rhs) noexcept
    {
        return lhs.type == rhs.type && lhs.key == rhs.key && lhs.mode == rhs.mode;
    }

    // Splits `m_rules` into consecutive runs sharing {type, key, mode}; each
    // run yields at most one decoded output argument.
    [[nodiscard]] std::vector<group_type> groups() const
    {
        std::vector<group_type> result;

        for(auto remaining = group_type{ m_rules }; !remaining.empty();)
        {
            const auto last = std::ranges::find_if_not(remaining, [&](const auto& rule) {
                return same_group(rule, remaining.front());
            });
            const auto size = static_cast<std::size_t>(last - remaining.begin());

            result.push_back(remaining.first(size));
            remaining = remaining.subspan(size);
        }

        return result;
    }

    [[nodiscard]] bool matches(const rule_type& rule) const
    {
        return has_flag(m_flags, rule.mask);
    }

    [[nodiscard]] std::optional<decoded_flag_arg> decode_first_match(
        group_type group) const
    {
        const auto match = std::ranges::find_if(
            group, [this](const auto& rule) { return matches(rule); });

        if(match == group.end())
        {
            return std::nullopt;
        }

        return decoded_flag_arg{ match->type, match->key, std::string{ match->value } };
    }

    [[nodiscard]] std::optional<decoded_flag_arg> decode_all_matches(
        group_type group) const
    {
        auto matched_values =
            group |
            std::views::filter([this](const auto& rule) { return matches(rule); }) |
            std::views::transform(&rule_type::value);

        std::string values;
        for(std::string_view value : matched_values)
        {
            if(!values.empty())
            {
                values += ", ";
            }
            values += value;
        }

        const auto& first_rule = group.front();

        if(!values.empty())
        {
            return decoded_flag_arg{ first_rule.type, first_rule.key, std::move(values) };
        }

        if(first_rule.on_no_match == rules::no_match_behavior::emit_none)
        {
            return decoded_flag_arg{ first_rule.type, first_rule.key, "none" };
        }

        return std::nullopt;
    }

    [[nodiscard]] std::optional<decoded_flag_arg> decode_group(group_type group) const
    {
        return group.front().mode == rules::flag_decode_mode::first_match
                   ? decode_first_match(group)
                   : decode_all_matches(group);
    }

    flag_type                  m_flags;
    std::span<const rule_type> m_rules;
};

using ompt_flag_representation = flag_representation<int>;

/**
 * Extracts and normalizes flags from the operation-specific OMPT payload union.
 *
 * A missing result means that `operation` has no supported flags field.
 */
template <policies::domain_service::backend SdkBackend>
[[nodiscard]] std::optional<ompt_flag_representation>
make_ompt_flag_representation(
    typename SdkBackend::ompt_operation_t                    operation,
    const typename SdkBackend::callback_tracing_ompt_data_t& payload)
{
    switch(operation)
    {
        case SdkBackend::OMPT_ID_parallel_begin:
            return ompt_flag_representation{ payload.args.parallel_begin.flags,
                                             rules::k_parallel };

        case SdkBackend::OMPT_ID_parallel_end:
            return ompt_flag_representation{ payload.args.parallel_end.flags,
                                             rules::k_parallel };

        case SdkBackend::OMPT_ID_task_create:
            return ompt_flag_representation{ payload.args.task_create.flags,
                                             rules::k_task_create };

        case SdkBackend::OMPT_ID_implicit_task:
            return ompt_flag_representation{ payload.args.implicit_task.flags,
                                             rules::k_implicit_task };

        case SdkBackend::OMPT_ID_cancel:
            return ompt_flag_representation{ payload.args.cancel.flags, rules::k_cancel };

        default: return std::nullopt;
    }
}

}  // namespace rocprofsys::domains::callback::ompt
