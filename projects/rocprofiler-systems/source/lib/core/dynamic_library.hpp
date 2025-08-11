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

#include "defines.hpp"

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
            fprintf(stderr, "[rocprof-sys][pid=%i]> %s :: %s\n", getpid(), _name.data(),
                    dlerror());
        }
    }
    return RetT{};
}
}  // namespace rocprofsys
