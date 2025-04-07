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
#include "common/delimit.hpp"
#include "common/environment.hpp"
#include "common/join.hpp"
#include "common/path.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <ios>
#include <link.h>
#include <linux/limits.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(ROCPROFSYS_SETUP_LOG_NAME)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_NAME)
#        define ROCPROFSYS_SETUP_LOG_NAME "[" ROCPROFSYS_COMMON_LIBRARY_NAME "]"
#    else
#        define ROCPROFSYS_SETUP_LOG_NAME
#    endif
#endif

#if !defined(ROCPROFSYS_SETUP_LOG_START)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_LOG_START)
#        define ROCPROFSYS_SETUP_LOG_START ROCPROFSYS_COMMON_LIBRARY_LOG_START
#    elif defined(TIMEMORY_LOG_COLORS_AVAILABLE)
#        define ROCPROFSYS_SETUP_LOG_START                                               \
            fprintf(stderr, "%s", ::tim::log::color::info());
#    else
#        define ROCPROFSYS_SETUP_LOG_START
#    endif
#endif

#if !defined(ROCPROFSYS_SETUP_LOG_END)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_LOG_END)
#        define ROCPROFSYS_SETUP_LOG_END ROCPROFSYS_COMMON_LIBRARY_LOG_END
#    elif defined(TIMEMORY_LOG_COLORS_AVAILABLE)
#        define ROCPROFSYS_SETUP_LOG_END fprintf(stderr, "%s", ::tim::log::color::end());
#    else
#        define ROCPROFSYS_SETUP_LOG_END
#    endif
#endif

#define ROCPROFSYS_SETUP_LOG(CONDITION, ...)                                             \
    if(CONDITION)                                                                        \
    {                                                                                    \
        fflush(stderr);                                                                  \
        ROCPROFSYS_SETUP_LOG_START                                                       \
        fprintf(stderr, "[rocprof-sys]" ROCPROFSYS_SETUP_LOG_NAME "[%i] ", getpid());    \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ROCPROFSYS_SETUP_LOG_END                                                         \
        fflush(stderr);                                                                  \
    }

namespace rocprofsys
{
inline namespace common
{
inline std::vector<env_config>
get_environ(int _verbose, std::string _search_paths = {},
            std::string _omnilib    = "librocprof-sys.so",
            std::string _omnilib_dl = "librocprof-sys-dl.so")
{
    auto _data            = std::vector<env_config>{};
    auto _omnilib_path    = path::get_origin(_omnilib);
    auto _omnilib_dl_path = path::get_origin(_omnilib_dl);

    if(!_omnilib_path.empty())
    {
        _omnilib      = join('/', _omnilib_path, ::basename(_omnilib.c_str()));
        _search_paths = join(':', _omnilib_path, _search_paths);
    }

    if(!_omnilib_dl_path.empty())
    {
        _omnilib_dl   = join('/', _omnilib_dl_path, ::basename(_omnilib_dl.c_str()));
        _search_paths = join(':', _omnilib_dl_path, _search_paths);
    }

    _omnilib    = common::path::find_path(_omnilib, _verbose, _search_paths);
    _omnilib_dl = common::path::find_path(_omnilib_dl, _verbose, _search_paths);

#if defined(ROCPROFSYS_USE_OMPT) && ROCPROFSYS_USE_OMPT > 0
    if(get_env("ROCPROFSYS_USE_OMPT", true))
    {
        std::string _omni_omp_libs = _omnilib_dl;
        const char* _omp_libs      = getenv("OMP_TOOL_LIBRARIES");
        int         _override      = 0;
        if(_omp_libs != nullptr &&
           std::string_view{ _omp_libs }.find(_omnilib_dl) == std::string::npos)
        {
            _override      = 1;
            _omni_omp_libs = common::join(':', _omp_libs, _omnilib_dl);
        }
        ROCPROFSYS_SETUP_LOG(_verbose >= 2, "setting OMP_TOOL_LIBRARIES to '%s'\n",
                             _omni_omp_libs.c_str());
        _data.emplace_back(
            env_config{ "OMP_TOOL_LIBRARIES", _omni_omp_libs.c_str(), _override });
    }
#endif

    return _data;
}

inline void
setup_environ(int _verbose, const std::string& _search_paths = {},
              std::string _omnilib    = "librocprof-sys.so",
              std::string _omnilib_dl = "librocprof-sys-dl.so")
{
    auto _data =
        get_environ(_verbose, _search_paths, std::move(_omnilib), std::move(_omnilib_dl));
    for(const auto& itr : _data)
        itr(_verbose >= 3);
}
}  // namespace common
}  // namespace rocprofsys
