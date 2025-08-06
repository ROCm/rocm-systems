// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <fmt/core.h>
#include <glog/logging.h>

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace rocprofiler_register
{
namespace common
{
namespace
{
inline std::string
get_env_impl(std::string_view env_id, std::string_view _default)
{
    if(env_id.empty()) return std::string{ _default };
    char* env_var = ::std::getenv(env_id.data());
    if(env_var) return std::string{ env_var };
    return std::string{ _default };
}

inline std::string
get_env_impl(std::string_view env_id, const char* _default)
{
    return get_env_impl(env_id, std::string_view{ _default });
}

inline int
get_env_impl(std::string_view env_id, int _default)
{
    if(env_id.empty()) return _default;
    char* env_var = ::std::getenv(env_id.data());
    if(env_var)
    {
        try
        {
            return std::stoi(env_var);
        } catch(std::exception& _e)
        {
            LOG(ERROR) << fmt::format(
                "[rocprofiler_register][get_env] Exception thrown converting getenv({}) "
                "= {} to integer :: {}. Using default value of {}",
                env_id.data(),
                env_var,
                _e.what(),
                _default);
        }
        return _default;
    }
    return _default;
}

inline bool
get_env_impl(std::string_view env_id, bool _default)
{
    if(env_id.empty()) return _default;
    char* env_var = ::std::getenv(env_id.data());
    if(env_var)
    {
        if(std::string_view{ env_var }.empty())
        {
            throw std::runtime_error(std::string{ "No boolean value provided for " } +
                                     std::string{ env_id });
        }

        if(std::string_view{ env_var }.find_first_not_of("0123456789") ==
           std::string_view::npos)
        {
            return static_cast<bool>(std::stoi(env_var));
        }

        for(size_t i = 0; i < strlen(env_var); ++i)
            env_var[i] = tolower(env_var[i]);
        for(const auto& itr : { "off", "false", "no", "n", "f", "0" })
            if(strcmp(env_var, itr) == 0) return false;

        return true;
    }
    return _default;
}

inline int
set_env_impl(std::string_view env_id, bool value, int overwrite)
{
    return ::setenv(env_id.data(), (value) ? "1" : "0", overwrite);
}

template <typename Tp>
int
set_env_impl(std::string_view env_id, Tp value, int overwrite)
{
    auto str_value = std::stringstream{};
    str_value << value;
    return ::setenv(env_id.data(), str_value.str().c_str(), overwrite);
}
}  // namespace

template <typename Tp>
inline auto
get_env(std::string_view env_id, Tp&& _default)
{
    if constexpr(std::is_enum<Tp>::value)
    {
        using Up = std::underlying_type_t<Tp>;
        // cast to underlying type -> get_env -> cast to enum type
        return static_cast<Tp>(get_env_impl(env_id, static_cast<Up>(_default)));
    }
    else
    {
        return get_env_impl(env_id, std::forward<Tp>(_default));
    }
}

template <typename Tp>
inline auto
set_env(std::string_view env_id, Tp&& value, int overwrite = 0)
{
    return set_env_impl(env_id, std::forward<Tp>(value), overwrite);
}

struct env_config
{
    std::string env_name  = {};
    std::string env_value = {};
    int         overwrite = 0;

    auto operator()() const
    {
        if(env_name.empty()) return -1;
        LOG(INFO) << fmt::format(
            "setenv({}, {}, {})", env_name.c_str(), env_value.c_str(), overwrite);
        return setenv(env_name.c_str(), env_value.c_str(), overwrite);
    }
};
}  // namespace common
}  // namespace rocprofiler_register
