// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/env_op.hpp"
#include "diagnostic.hpp"
#include "options/context.hpp"
#include "verb.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::rocsys::options
{

/// @brief How many values a rocsys option takes.
enum class value_kind : std::uint8_t
{
    flag = 0,    ///< `--name` with no value.
    single,      ///< `--name value` or `--name=value`.
    token_list,  ///< Values until the next flag or `--`. Commas split one token.
};

/// @brief True when a passthrough token for @p binary conflicts with an option.
///        A null @ref matches_value means the flag conflicts with any value, including
///        no value.
struct legacy_conflict
{
    std::string_view                  binary    = {};
    std::span<const std::string_view> spellings = {};
    bool (*matches_value)(std::string_view)     = nullptr;
    std::string_view reason                     = {};
};

using accept_fn = bool (*)(std::string_view token, const context& ctx);

using choices_fn = std::vector<std::string> (*)(const context& ctx);

/// @brief Append env ops for one option. Empty means success.
using resolve_fn = std::optional<diagnostic> (*)(std::span<const std::string> values,
                                                 const verb_spec&             verb,
                                                 const context& ctx, env_plan& out);

/// @brief One rocsys option. Rows live in the option table; callbacks are non-owning.
struct option_spec
{
    std::string_view                  name          = {};
    value_kind                        kind          = value_kind::flag;
    capability_set                    required_caps = {};
    std::string_view                  group         = {};
    std::string_view                  summary       = {};
    std::string_view                  description   = {};
    std::span<const std::string_view> env_keys      = {};
    std::span<const legacy_conflict>  conflicts     = {};
    std::uint8_t                      min_values    = {};
    accept_fn                         accepts       = nullptr;
    choices_fn                        choices       = nullptr;
    resolve_fn                        resolve       = nullptr;
};

}  // namespace rocprofsys::rocsys::options
