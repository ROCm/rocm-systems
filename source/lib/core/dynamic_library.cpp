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

#include "dynamic_library.hpp"
#include "common.hpp"
#include "debug.hpp"
#include "defines.hpp"

#include <timemory/environment.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <string>
#include <utility>

namespace rocprofsys
{
namespace procfs = ::tim::procfs;

std::string
find_library_path(const std::string& _name, const std::vector<std::string>& _env_vars,
                  const std::vector<std::string>& _hints,
                  const std::vector<std::string>& _path_suffixes)
{
    if(_name.find('/') == 0) return _name;

    for(const auto& itr : procfs::get_maps(process::get_id(), true))
    {
        auto&& _path = itr.pathname;
        if(_path.find(_name) != std::string::npos && filepath::exists(_path))
            return _path;
    }

    auto _paths = std::vector<std::string>{};
    for(const std::string& itr : _env_vars)
    {
        auto _env_val = get_env(itr, std::string{});
        for(auto vitr : tim::delimit(_env_val, ":"))
            if(!vitr.empty()) _paths.emplace_back(vitr);
    }

    for(const std::string& itr : _hints)
    {
        if(!itr.empty()) _paths.emplace_back(itr);
    }

    for(auto& itr : _paths)
    {
        auto _v = JOIN('/', itr, _name);
        if(filepath::exists(_v)) return _v;
        for(const auto& litr : _path_suffixes)
        {
            _v = JOIN('/', itr, litr, _name);
            if(filepath::exists(_v)) return _v;
        }
    }

    return _name;
}

dynamic_library::dynamic_library(std::string _env, std::string _fname, int _flags,
                                 bool _open, bool _query_env, bool _store)
: envname{ std::move(_env) }
, filename{ std::move(_fname) }
, flags{ _flags }
{
    // check the memory maps
    filename = find_library_path(filename, {}, {});

    if(_query_env)
    {
        auto _env_val = get_env(envname, std::string{}, _store);
        // if the environment variable is set to an absolute path that exists,
        // override with value
        if(!_env_val.empty())
        {
            if(_env_val.find('/') == 0 && filepath::exists(_env_val))
            {
                filename = _env_val;
            }
            else if(_env_val.find('/') == 0)
            {
                ROCPROFSYS_VERBOSE_F(
                    1,
                    "Ignoring environment variable %s=\"%s\" because the "
                    "filepath does not exist. Using \"%s\" instead...\n",
                    envname.c_str(), _env_val.c_str(), filename.c_str())
            }
            else if(_env_val.find('/') != 0 && filename.find('/') == 0)
            {
                ROCPROFSYS_VERBOSE_F(
                    1,
                    "Ignoring environment variable %s=\"%s\" because the "
                    "filepath is relative. Using absolute path \"%s\" instead...\n",
                    envname.c_str(), _env_val.c_str(), filename.c_str())
            }
        }
    }

    if(_open) open();
}

dynamic_library::~dynamic_library() { close(); }

bool
dynamic_library::open()
{
    if(!filename.empty())
    {
        handle = dlopen(filename.c_str(), flags);
        if(!handle)
        {
            ROCPROFSYS_VERBOSE(2, "[dynamic_library] Error opening %s=\"%s\" :: %s.\n",
                               envname.c_str(), filename.c_str(), dlerror());
        }
        dlerror();  // Clear any existing error
    }
    return (handle != nullptr);
}

int
dynamic_library::close() const
{
    if(handle) return dlclose(handle);
    return -1;
}

bool
dynamic_library::is_open() const
{
    return (handle != nullptr);
}
}  // namespace rocprofsys
