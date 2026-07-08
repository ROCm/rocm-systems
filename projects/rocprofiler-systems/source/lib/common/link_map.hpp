// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Dynamic-linker introspection (dlopen/dlinfo) yielding library paths.
// NOT a filesystem operation (survey G19); split out of the path.hpp god-header.
// Consumes path::exists from common/path.hpp.

#include "common/defines.h"
#include "common/path.hpp"

#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
inline namespace common
{
namespace path
{
inline std::vector<std::string>
get_link_map(const char*, std::vector<int>&& = { (RTLD_LAZY | RTLD_NOLOAD) },
             bool _include_self = false) ROCPROFSYS_INTERNAL_API;

inline auto
get_link_map(const char* _name, bool&& _include_self,
             std::vector<int>&& _open_modes = {
                 (RTLD_LAZY | RTLD_NOLOAD) }) ROCPROFSYS_INTERNAL_API;

inline std::string
get_origin(const std::string&,
           std::vector<int>&& = { (RTLD_LAZY | RTLD_NOLOAD) }) ROCPROFSYS_INTERNAL_API;

//--------------------------------------------------------------------------------------//

inline std::vector<std::string>
get_link_map(const char* _name, std::vector<int>&& _open_modes, bool _include_self)
{
    void* _handle = nullptr;
    bool  _noload = false;
    for(auto _mode : _open_modes)
    {
        _handle = dlopen(_name, _mode);
        _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
        if(_handle) break;
    }

    auto _chain = std::vector<std::string>{};
    if(_handle)
    {
        struct link_map* _link_map = nullptr;
        dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
        // if include_self is false, start at next library
        struct link_map* _next = (_include_self) ? _link_map : _link_map->l_next;
        while(_next)
        {
            if(_next->l_name != nullptr && !std::string_view{ _next->l_name }.empty())
            {
                _chain.emplace_back(_next->l_name);
            }
            _next = _next->l_next;
        }

        if(_noload == false) dlclose(_handle);
    }
    return _chain;
}

inline auto
get_link_map(const char* _name, bool&& _include_self, std::vector<int>&& _open_modes)
{
    return get_link_map(_name, std::move(_open_modes), _include_self);
}

inline std::string
get_origin(const std::string& _filename, std::vector<int>&& _open_modes)
{
    void* _handle = nullptr;
    bool  _noload = false;
    for(auto _mode : _open_modes)
    {
        _handle = dlopen(_filename.c_str(), _mode);
        _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
        if(_handle) break;
    }

    auto _chain = std::vector<std::string>{};
    if(_handle)
    {
        char _buffer[PATH_MAX];
        memset(_buffer, '\0', PATH_MAX * sizeof(char));
        if(dlinfo(_handle, RTLD_DI_ORIGIN, &_buffer) == 0)
        {
            auto _origin = std::string{ _buffer };
            if(exists(_origin)) return _origin;
        }

        if(_noload == false) dlclose(_handle);
    }

    return std::string{};
}

}  // namespace path
}  // namespace common
}  // namespace rocprofsys
