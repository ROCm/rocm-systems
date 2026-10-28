// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::rocsys
{

/// @brief How one @ref env_op combines with an existing variable.
///        prepend, append, and unset arrive with the first option that needs them.
enum class env_mode : std::uint8_t
{
    replace = 0,  ///< Overwrite the value. An empty value is a valid replace.
};

/// @brief One environment change. The launcher applies non-embedded ops; delivery
///        turns embedded ops into `--env K=V` tokens.
struct env_op
{
    std::string key    = {};
    std::string value  = {};
    env_mode    mode   = env_mode::replace;
    std::string origin = {};  ///< The rocsys option that produced this op.
    bool embed = {};  ///< Rewrite bakes the value into the binary instead of inheriting
                      ///< it.
};

/// @brief The ordered environment changes for one launch. Phase 2 has a single layer.
using env_plan = std::vector<env_op>;

/// @brief True when @p entry is `key=...` and not a longer key that shares a prefix.
[[nodiscard]] inline bool
env_entry_has_key(std::string_view entry, std::string_view key)
{
    return entry.size() > key.size() && entry.starts_with(key) &&
           entry[key.size()] == '=';
}

/// @brief Copy @p base and apply each replace op. Later ops for the same key win.
[[nodiscard]] inline std::vector<std::string>
merge_environ(std::vector<std::string> base, const env_plan& ops)
{
    for(const auto& op : ops)
    {
        std::erase_if(base, [&op](const std::string& entry) {
            return env_entry_has_key(entry, op.key);
        });
        base.push_back(op.key + "=" + op.value);
    }
    return base;
}

}  // namespace rocprofsys::rocsys
