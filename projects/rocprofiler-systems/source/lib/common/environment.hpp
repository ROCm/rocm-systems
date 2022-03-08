// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace omnitrace
{
inline namespace common
{
namespace
{
inline std::string
get_env(const char* env_id, const char* _default)
{
    if(strlen(env_id) == 0) return _default;
    char* env_var = ::std::getenv(env_id);
    if(env_var) return std::string{ env_var };
    return _default;
}

inline int
get_env(const char* env_id, int _default)
{
    if(strlen(env_id) == 0) return _default;
    char* env_var = ::std::getenv(env_id);
    if(env_var) return std::stoi(env_var);
    return _default;
}

inline bool
get_env(const char* env_id, bool _default)
{
    if(strlen(env_id) == 0) return _default;
    char* env_var = ::std::getenv(env_id);
    if(env_var)
    {
        if(std::string_view{ env_var }.find_first_not_of("0123456789") ==
           std::string_view::npos)
            return static_cast<bool>(std::stoi(env_var));
        else
        {
            for(size_t i = 0; i < strlen(env_var); ++i)
                env_var[i] = tolower(env_var[i]);
            for(const auto& itr : { "off", "false", "no", "n", "f", "0" })
                if(strcmp(env_var, itr) == 0) return false;
        }
        return true;
    }
    return _default;
}
}  // namespace
}  // namespace common
}  // namespace omnitrace
