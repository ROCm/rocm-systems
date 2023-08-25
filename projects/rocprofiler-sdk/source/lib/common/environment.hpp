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

#include "lib/common/log.hpp"

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#if !defined(ROCPROFILER_ENVIRON_LOG_NAME)
#    if defined(ROCPROFILER_COMMON_LIBRARY_NAME)
#        define ROCPROFILER_ENVIRON_LOG_NAME "[" ROCPROFILER_COMMON_LIBRARY_NAME "]"
#    else
#        define ROCPROFILER_ENVIRON_LOG_NAME "[environ]"
#    endif
#endif

#if !defined(ROCPROFILER_ENVIRON_LOG_START)
#    if defined(ROCPROFILER_COMMON_LIBRARY_LOG_START)
#        define ROCPROFILER_ENVIRON_LOG_START ROCPROFILER_COMMON_LIBRARY_LOG_START
#    elif defined(ROCPROFILER_LOG_COLORS_AVAILABLE)
#        define ROCPROFILER_ENVIRON_LOG_START                                                      \
            fprintf(stderr, "%s", ::rocprofiler::common::log::color::dmesg());
#    else
#        define ROCPROFILER_ENVIRON_LOG_START
#    endif
#endif

#if !defined(ROCPROFILER_ENVIRON_LOG_END)
#    if defined(ROCPROFILER_COMMON_LIBRARY_LOG_END)
#        define ROCPROFILER_ENVIRON_LOG_END ROCPROFILER_COMMON_LIBRARY_LOG_END
#    elif defined(ROCPROFILER_LOG_COLORS_AVAILABLE)
#        define ROCPROFILER_ENVIRON_LOG_END                                                        \
            fprintf(stderr, "%s", ::rocprofiler::common::log::color::dmesg());
#    else
#        define ROCPROFILER_ENVIRON_LOG_END
#    endif
#endif

#define ROCPROFILER_ENVIRON_LOG(CONDITION, ...)                                                    \
    if(CONDITION)                                                                                  \
    {                                                                                              \
        fflush(stderr);                                                                            \
        ROCPROFILER_ENVIRON_LOG_START                                                              \
        fprintf(stderr, "[rocprofiler]" ROCPROFILER_ENVIRON_LOG_NAME "[%i] ", getpid());           \
        fprintf(stderr, __VA_ARGS__);                                                              \
        ROCPROFILER_ENVIRON_LOG_END                                                                \
        fflush(stderr);                                                                            \
    }

namespace rocprofiler
{
namespace common
{
namespace
{
inline std::string
get_env_impl(std::string_view env_id, std::string_view _default)
{
    if(env_id.empty()) return std::string{_default};
    char* env_var = ::std::getenv(env_id.data());
    if(env_var) return std::string{env_var};
    return std::string{_default};
}

inline std::string
get_env_impl(std::string_view env_id, const char* _default)
{
    return get_env_impl(env_id, std::string_view{_default});
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
            fprintf(stderr,
                    "[rocprofiler][get_env] Exception thrown converting getenv(\"%s\") = "
                    "%s to integer :: %s. Using default value of %i\n",
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
        if(std::string_view{env_var}.empty())
        {
            throw std::runtime_error(std::string{"No boolean value provided for "} +
                                     std::string{env_id});
        }

        if(std::string_view{env_var}.find_first_not_of("0123456789") == std::string_view::npos)
        {
            return static_cast<bool>(std::stoi(env_var));
        }

        for(size_t i = 0; i < strlen(env_var); ++i)
            env_var[i] = tolower(env_var[i]);
        for(const auto& itr : {"off", "false", "no", "n", "f", "0"})
            if(strcmp(env_var, itr) == 0) return false;

        return true;
    }
    return _default;
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

struct env_config
{
    std::string env_name  = {};
    std::string env_value = {};
    int         override  = 0;

    auto operator()(bool _verbose = false) const
    {
        if(env_name.empty()) return -1;
        ROCPROFILER_ENVIRON_LOG(_verbose,
                                "setenv(\"%s\", \"%s\", %i)\n",
                                env_name.c_str(),
                                env_value.c_str(),
                                override);
        return setenv(env_name.c_str(), env_value.c_str(), override);
    }
};
}  // namespace common
}  // namespace rocprofiler
