// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace rocprofsys
{
inline namespace common
{
namespace output
{

[[nodiscard]] constexpr bool
matches_env_key(std::string_view _entry, std::string_view _key) noexcept
{
    return _entry.size() > _key.size() && _entry[_key.size()] == '=' &&
           _entry.substr(0, _key.size()) == _key;
}

[[nodiscard]] constexpr bool
starts_with_rocprofsys(std::string_view _entry) noexcept
{
    constexpr std::string_view _prefix = "ROCPROFSYS";
    return _entry.size() >= _prefix.size() && _entry.substr(0, _prefix.size()) == _prefix;
}

[[nodiscard]] inline std::string
build_command_string(const std::vector<char*>& _argv)
{
    std::string _result;
    size_t      _estimated_size = 0;
    for(const auto* arg : _argv)
    {
        if(arg == nullptr) continue;
        _estimated_size += std::string_view{ arg }.size() + 1;
    }
    _result.reserve(_estimated_size);
    for(const auto* arg : _argv)
    {
        if(arg == nullptr) continue;
        if(!_result.empty()) _result += ' ';
        _result += arg;
    }
    return _result;
}

inline void
print_command(const std::vector<char*>& _argv, std::string_view _prefix = {})
{
    if(_argv.empty()) return;
    const auto _cmd = build_command_string(_argv);
    if(_cmd.empty()) return;
    std::cout << _prefix << "Executing '" << _cmd << "'...\n" << std::flush;
}

inline void
print_environment(const std::vector<char*>&                   _env,
                  const std::unordered_set<std::string_view>& _updated_envs,
                  bool _include_general_vars = false, std::string_view _prefix = {})
{
    auto _env_sorted = _env;

    const auto _valid_end = std::remove(_env_sorted.begin(), _env_sorted.end(), nullptr);
    std::sort(_env_sorted.begin(), _valid_end,
              [](const char* const _lhs, const char* const _rhs) {
                  return std::string_view{ _lhs } < std::string_view{ _rhs };
              });

    const auto _partition_point =
        std::stable_partition(_env_sorted.begin(), _valid_end, [&](const char* _entry) {
            auto       _sv     = std::string_view{ _entry };
            const auto _eq_pos = _sv.find('=');
            if(_eq_pos == std::string_view::npos) return false;
            return _updated_envs.count(_sv.substr(0, _eq_pos)) > 0;
        });

    const std::vector<std::string_view> _updated_vars(_env_sorted.begin(),
                                                      _partition_point);

    std::vector<std::string_view> _general_vars;
    if(_include_general_vars)
    {
        std::copy_if(_partition_point, _valid_end, std::back_inserter(_general_vars),
                     [](const char* _entry) {
                         return starts_with_rocprofsys(std::string_view{ _entry });
                     });
    }

    if(_general_vars.empty() && _updated_vars.empty()) return;

    std::cerr << '\n';
    for(const auto& _var : _general_vars)
        std::cerr << _prefix << _var << "\n";
    for(const auto& _var : _updated_vars)
        std::cerr << _prefix << _var << "\n";
    std::cerr << std::flush;
}

}  // namespace output
}  // namespace common
}  // namespace rocprofsys
