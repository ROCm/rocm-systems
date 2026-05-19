// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
