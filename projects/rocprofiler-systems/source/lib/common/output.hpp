// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
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
print_command(const std::vector<char*>& _argv, int _verbose,
              std::string_view _prefix = {})
{
    if(_verbose < 1) return;

    auto _cmd = build_command_string(_argv);
    std::cout << _prefix << "Executing '" << _cmd << "'..." << std::endl;
}

template <typename UpdatedEnvsT>
inline void
print_updated_environment(const std::vector<char*>& _env,
                          const UpdatedEnvsT& _updated_envs, int _verbose,
                          std::string_view _prefix = {})
{
    if(_verbose < 0) return;

    auto _env_sorted = _env;
    std::sort(_env_sorted.begin(), _env_sorted.end(),
              [](const char* const _lhs, const char* const _rhs) {
                  if(!_lhs) return false;
                  if(!_rhs) return true;
                  return std::string_view{ _lhs } < std::string_view{ _rhs };
              });

    auto valid_end = std::remove(_env_sorted.begin(), _env_sorted.end(), nullptr);

    auto is_updated = [&](const char* entry) {
        auto sv = std::string_view{ entry };
        return std::any_of(_updated_envs.begin(), _updated_envs.end(),
                           [sv](const auto& key) { return sv.find(key) == 0; });
    };

    auto partition_point =
        std::stable_partition(_env_sorted.begin(), valid_end, is_updated);

    std::vector<std::string_view> _updated_vars(_env_sorted.begin(), partition_point);

    std::vector<std::string_view> _general_vars;
    if(_verbose >= 1)
    {
        constexpr std::string_view rocprofsys_prefix = "ROCPROFSYS";
        std::copy_if(partition_point, valid_end, std::back_inserter(_general_vars),
                     [rocprofsys_prefix](const char* entry) {
                         return std::string_view{ entry }.find(rocprofsys_prefix) == 0;
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
