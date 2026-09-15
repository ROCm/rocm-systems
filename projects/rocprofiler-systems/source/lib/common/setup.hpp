// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/environment.hpp"
#include "common/path.hpp"
#include <fmt/format.h>

#include <cstdlib>
#include <dlfcn.h>
#include <link.h>
#include <linux/limits.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace rocprofsys
{
inline namespace common
{
inline std::vector<env_config<>>
get_environ(int _verbose, std::string _search_paths = {},
            std::string _omnilib    = "librocprof-sys.so",
            std::string _omnilib_dl = "librocprof-sys-dl.so")
{
    auto _data            = std::vector<env_config<>>{};
    auto _omnilib_path    = path::get_origin(_omnilib);
    auto _omnilib_dl_path = path::get_origin(_omnilib_dl);

    if(!_omnilib_path.empty())
    {
        _omnilib      = fmt::format("{}/{}", _omnilib_path, path::filename(_omnilib));
        _search_paths = fmt::format("{}:{}", _omnilib_path, _search_paths);
    }

    if(!_omnilib_dl_path.empty())
    {
        _omnilib_dl = fmt::format("{}/{}", _omnilib_dl_path, path::filename(_omnilib_dl));
        _search_paths = fmt::format("{}:{}", _omnilib_dl_path, _search_paths);
    }

    if(_search_paths.find_first_not_of(':') == std::string::npos)
    {
        _search_paths = get_default_lib_search_paths();
    }

    _omnilib    = common::path::find_library(_omnilib, _verbose, _search_paths);
    _omnilib_dl = common::path::find_library(_omnilib_dl, _verbose, _search_paths);

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
        itr();
}
}  // namespace common
}  // namespace rocprofsys
