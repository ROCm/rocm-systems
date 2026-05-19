// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "logger/debug.hpp"

#include <dlfcn.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace rocprofsys
{
std::string
find_library_path(const std::string& _name, const std::vector<std::string>& _env_vars,
                  const std::vector<std::string>& _hints,
                  const std::vector<std::string>& _path_suffixes = { "lib", "lib64" });

struct dynamic_library
{
    dynamic_library()                                      = delete;
    dynamic_library(const dynamic_library&)                = delete;
    dynamic_library(dynamic_library&&) noexcept            = default;
    dynamic_library& operator=(const dynamic_library&)     = delete;
    dynamic_library& operator=(dynamic_library&&) noexcept = default;

    dynamic_library(std::string _env, std::string _fname,
                    int _flags = (RTLD_LAZY | RTLD_GLOBAL), bool _open = true,
                    bool _query_env = true, bool _store = true);

    ~dynamic_library();

    bool open();
    int  close() const;
    bool is_open() const;

    template <typename RetT, typename... Args>
    RetT invoke(std::string_view, RetT (*&_func)(Args...), Args...);

    std::string envname  = {};
    std::string filename = {};
    int         flags    = 0;
    void*       handle   = nullptr;
};

template <typename RetT, typename... Args>
inline RetT
dynamic_library::invoke(std::string_view _name, RetT (*&_func)(Args...), Args... _args)
{
    if(!handle) open();
    if(handle)
    {
        *(void**) (&_func) = dlsym(handle, _name.data());
        if(_func)
        {
            return (*_func)(_args...);
        }
        else
        {
            LOG_WARNING("[rocprof-sys][pid={}]> {} :: {}", getpid(), _name, dlerror());
        }
    }
    return RetT{};
}
}  // namespace rocprofsys
