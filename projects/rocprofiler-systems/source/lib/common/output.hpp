// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "common/defines.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
inline namespace common
{
namespace output
{

inline std::string
build_command_string(const std::vector<char*>& _argv)
{
    std::string _result;
    for(const auto& arg : _argv)
    {
        if(arg == nullptr) continue;
        if(!_result.empty()) _result += " ";
        _result += arg;
    }
    return _result;
}

inline void
print_command(const std::vector<char*>& _argv, int _verbose,
              std::string_view _prefix = {}) ROCPROFSYS_INTERNAL_API;

inline void
print_command(const std::vector<char*>& _argv, int _verbose, std::string_view _prefix)
{
    if(_verbose < 1) return;

    auto _cmd = build_command_string(_argv);
    std::cout << _prefix << "Executing '" << _cmd << "'...\n";
    std::cout << std::flush;
}

namespace detail
{
/// Environment variable prefix for rocprofiler-systems
inline constexpr std::string_view rocprofsys_prefix = "ROCPROFSYS";

/// @brief Check if environment variable name starts with "ROCPROFSYS"
/// @param _entry Environment variable string to check
/// @return true if starts with "ROCPROFSYS", false otherwise
inline bool
starts_with_rocprofsys(std::string_view _entry) noexcept
{
    return _entry.find(rocprofsys_prefix) == 0;
}

/// @brief Check if environment variable was updated
/// @tparam UpdatedEnvsT Container type (e.g., std::unordered_set)
/// @param _entry Environment variable to check
/// @param _updated_envs Set of updated environment variable prefixes
/// @return true if _entry starts with any prefix in _updated_envs
template <typename UpdatedEnvsT>
inline bool
was_updated(std::string_view _entry, const UpdatedEnvsT& _updated_envs)
{
    return std::any_of(_updated_envs.begin(), _updated_envs.end(),
                       [_entry](const auto& _key) { return _entry.find(_key) == 0; });
}

/// @brief Sort environment variables alphabetically, handling nulls
/// @param _env Vector of C-strings to sort (modified in-place)
/// @note Null pointers are sorted to the end
inline void
sort_environment(std::vector<char*>& _env)
{
    std::sort(_env.begin(), _env.end(),
              [](const char* const _lhs, const char* const _rhs) {
                  if(!_lhs) return false;
                  if(!_rhs) return true;
                  return std::string_view{ _lhs } < std::string_view{ _rhs };
              });
}

/// @brief Print environment variables to stderr with optional prefix
/// @param _vars Environment variables to print
/// @param _prefix Prefix to add before each variable
inline void
print_env_vars(const std::vector<std::string_view>& _vars, std::string_view _prefix)
{
    for(const auto& _var : _vars)
        std::cerr << _prefix << _var << "\n";
}
}  // namespace detail

template <typename UpdatedEnvsT>
inline void
print_updated_environment(const std::vector<char*>& _env,
                          const UpdatedEnvsT& _updated_envs, int _verbose,
                          std::string_view _prefix = {}) ROCPROFSYS_INTERNAL_API;

template <typename UpdatedEnvsT>
inline void
print_updated_environment(const std::vector<char*>& _env,
                          const UpdatedEnvsT& _updated_envs, int _verbose,
                          std::string_view _prefix)
{
    if(_verbose < 0) return;

    auto _env_sorted = _env;
    detail::sort_environment(_env_sorted);

    std::vector<std::string_view> _updated_vars;
    std::vector<std::string_view> _general_vars;
    _updated_vars.reserve(_env_sorted.size());
    _general_vars.reserve(_env_sorted.size());

    for(const auto* _entry_ptr : _env_sorted)
    {
        if(_entry_ptr == nullptr) continue;

        auto _entry = std::string_view{ _entry_ptr };

        if(detail::was_updated(_entry, _updated_envs))
            _updated_vars.emplace_back(_entry);
        else if(_verbose >= 1 && detail::starts_with_rocprofsys(_entry))
            _general_vars.emplace_back(_entry);
    }

    if(_general_vars.empty() && _updated_vars.empty()) return;

    std::cerr << '\n';
    detail::print_env_vars(_general_vars, _prefix);
    detail::print_env_vars(_updated_vars, _prefix);
    std::cerr << std::flush;
}

}  // namespace output
}  // namespace common
}  // namespace rocprofsys
