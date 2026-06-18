// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dynamic_library.hpp"
#include "common.hpp"

#include "common/environment.hpp"
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include "logger/debug.hpp"

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
        auto _env_val = get_env(itr.c_str(), std::string{});
        for(auto vitr : tim::delimit(_env_val, ":"))
            if(!vitr.empty()) _paths.emplace_back(vitr);
    }

    for(const std::string& itr : _hints)
    {
        if(!itr.empty()) _paths.emplace_back(itr);
    }

    for(auto& itr : _paths)
    {
        auto _v = fmt::format("{}/{}", itr, _name);
        if(filepath::exists(_v)) return _v;
        for(const auto& litr : _path_suffixes)
        {
            _v = fmt::format("{}/{}/{}", itr, litr, _name);
            if(filepath::exists(_v)) return _v;
        }
    }

    return _name;
}

dynamic_library::dynamic_library(std::string _env, std::string _fname, int _flags,
                                 bool _open, bool _query_env)
: envname{ std::move(_env) }
, filename{ std::move(_fname) }
, flags{ _flags }
{
    // check the memory maps
    filename = find_library_path(filename, {}, {});

    if(_query_env)
    {
        auto _env_val = common::get_env(envname.c_str(), std::string{});
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
                LOG_WARNING("Ignoring environment variable {}=\"{}\" because the "
                            "filepath does not exist. Using \"{}\" instead...",
                            envname, _env_val, filename);
            }
            else if(_env_val.find('/') != 0 && filename.find('/') == 0)
            {
                LOG_WARNING("Ignoring environment variable {}=\"{}\" because the "
                            "filepath is relative. Using absolute path \"{}\" instead...",
                            envname, _env_val, filename);
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
            LOG_WARNING("[dynamic_library] Error opening {}=\"{}\" :: {}.", envname,
                        filename, dlerror());
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
