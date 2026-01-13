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
    for(size_t i = 0; i < _argv.size(); ++i)
    {
        if(_argv[i] == nullptr) continue;
        if(!_result.empty()) _result += " ";
        _result += _argv[i];
    }
    return _result;
}

inline void
print_command(const std::vector<char*>& _argv, int _verbose,
              std::string_view _prefix = {}) ROCPROFSYS_INTERNAL_API;

inline void
print_command(const std::vector<char*>& _argv, int _verbose, std::string_view _prefix)
{
    if(_verbose >= 1)
    {
        auto _cmd = build_command_string(_argv);
        std::cout << _prefix << "Executing '" << _cmd << "'...\n";
        std::cout << std::flush;
    }
}

template <typename UpdatedEnvsT>
inline void
print_updated_environment(std::vector<char*> _env, const UpdatedEnvsT& _updated_envs,
                          int              _verbose,
                          std::string_view _prefix = {}) ROCPROFSYS_INTERNAL_API;

template <typename UpdatedEnvsT>
inline void
print_updated_environment(std::vector<char*> _env, const UpdatedEnvsT& _updated_envs,
                          int _verbose, std::string_view _prefix)
{
    if(_verbose < 0) return;

    std::sort(_env.begin(), _env.end(), [](auto* _lhs, auto* _rhs) {
        if(!_lhs) return false;
        if(!_rhs) return true;
        return std::string_view{ _lhs } < std::string_view{ _rhs };
    });

    std::vector<std::string_view> _updates = {};
    std::vector<std::string_view> _general = {};

    for(auto* itr : _env)
    {
        if(itr == nullptr) continue;

        auto _is_omni = (std::string_view{ itr }.find("ROCPROFSYS") == 0);
        auto _updated = false;
        for(const auto& vitr : _updated_envs)
        {
            if(std::string_view{ itr }.find(vitr) == 0)
            {
                _updated = true;
                break;
            }
        }

        if(_updated)
            _updates.emplace_back(itr);
        else if(_verbose >= 1 && _is_omni)
            _general.emplace_back(itr);
    }

    if(_general.size() + _updates.size() == 0 || _verbose < 0) return;

    std::cerr << std::endl;

    for(const auto& itr : _general)
        std::cerr << _prefix << itr << "\n";
    for(const auto& itr : _updates)
        std::cerr << _prefix << itr << "\n";

    std::cerr << std::flush;
}

}  // namespace output
}  // namespace common
}  // namespace rocprofsys
