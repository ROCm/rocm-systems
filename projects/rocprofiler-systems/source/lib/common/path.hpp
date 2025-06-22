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

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <link.h>
#include <linux/limits.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(ROCPROFSYS_PATH_LOG_NAME)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_NAME)
#        define ROCPROFSYS_PATH_LOG_NAME "[" ROCPROFSYS_COMMON_LIBRARY_NAME "]"
#    else
#        define ROCPROFSYS_PATH_LOG_NAME
#    endif
#endif

#if !defined(ROCPROFSYS_PATH_LOG_START)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_LOG_START)
#        define ROCPROFSYS_PATH_LOG_START ROCPROFSYS_COMMON_LIBRARY_LOG_START
#    elif defined(TIMEMORY_LOG_COLORS_AVAILABLE)
#        define ROCPROFSYS_PATH_LOG_START                                                \
            fprintf(stderr, "%s", ::tim::log::color::info());
#    else
#        define ROCPROFSYS_PATH_LOG_START
#    endif
#endif

#if !defined(ROCPROFSYS_PATH_LOG_END)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_LOG_END)
#        define ROCPROFSYS_PATH_LOG_END ROCPROFSYS_COMMON_LIBRARY_LOG_END
#    elif defined(TIMEMORY_LOG_COLORS_AVAILABLE)
#        define ROCPROFSYS_PATH_LOG_END fprintf(stderr, "%s", ::tim::log::color::end());
#    else
#        define ROCPROFSYS_PATH_LOG_END
#    endif
#endif

#define ROCPROFSYS_PATH_LOG(CONDITION, ...)                                              \
    if(CONDITION)                                                                        \
    {                                                                                    \
        fflush(stderr);                                                                  \
        ROCPROFSYS_PATH_LOG_START                                                        \
        fprintf(stderr, "[rocprof-sys]" ROCPROFSYS_PATH_LOG_NAME "[%i] ", getpid());     \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ROCPROFSYS_PATH_LOG_END                                                          \
        fflush(stderr);                                                                  \
    }

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

inline bool
exists(const std::string& _fname) ROCPROFSYS_INTERNAL_API;

template <typename RetT = std::string>
inline RetT
get_default_lib_search_paths() ROCPROFSYS_INTERNAL_API;

inline std::string
find_path(const std::string& _path, int _verbose,
          const std::string& _search_paths = {}) ROCPROFSYS_INTERNAL_API;

inline std::string
dirname(const std::string& _fname) ROCPROFSYS_INTERNAL_API;

inline std::string
realpath(const std::string& _relpath,
         std::string*       _resolved = nullptr) ROCPROFSYS_INTERNAL_API;

inline bool
is_text_file(const std::string& filename) ROCPROFSYS_INTERNAL_API;

inline bool
is_link(const std::string& _path) ROCPROFSYS_INTERNAL_API;

inline std::string
readlink(const std::string& _path) ROCPROFSYS_INTERNAL_API;

struct ROCPROFSYS_INTERNAL_API path_type
{
    enum path_type_e
    {
        directory = 0,
        regular,
        link,
        unknown
    };

    inline path_type(const std::string&);
    ~path_type()                           = default;
    path_type(const path_type&)            = default;
    path_type(path_type&&)                 = default;
    path_type& operator=(const path_type&) = default;
    path_type& operator=(path_type&&)      = default;

    bool     exists() const { return m_type < unknown; }
    explicit operator bool() const { return exists(); }

private:
    path_type_e m_type = unknown;
};

//--------------------------------------------------------------------------------------//
//
//      Implementation
//
//--------------------------------------------------------------------------------------//

path_type::path_type(const std::string& _fname)
{
    struct stat _buffer;
    if(lstat(_fname.c_str(), &_buffer) == 0)
    {
        if(S_ISDIR(_buffer.st_mode) != 0)
            m_type = directory;
        else if(S_ISREG(_buffer.st_mode) != 0)
            m_type = regular;
        else if(S_ISLNK(_buffer.st_mode) != 0)
            m_type = link;
    }
}

bool
exists(const std::string& _fname)
{
    struct stat _buffer;
    if(lstat(_fname.c_str(), &_buffer) == 0)
        return (S_ISDIR(_buffer.st_mode) != 0 || S_ISREG(_buffer.st_mode) != 0 ||
                S_ISLNK(_buffer.st_mode) != 0);
    return false;
}

template <typename RetT>
RetT
get_default_lib_search_paths()
{
    auto _paths =
        join(":", get_env("ROCPROFSYS_PATH", ""), get_env("LD_LIBRARY_PATH", ""),
             get_env("LIBRARY_PATH", ""), get_env("PWD", ""), ".");
    if constexpr(std::is_same<RetT, std::string>::value)
        return _paths;
    else
        return delimit(_paths, ":");
}

std::string
find_path(const std::string& _path, int _verbose, const std::string& _search_paths)
{
    if(exists(_path) && !_path.empty() && _path.at(0) == '/') return _path;

    auto _paths = delimit(_search_paths, ":");
    if(_paths.empty())
    {
        _paths = get_default_lib_search_paths<std::vector<std::string>>();
    }

    constexpr int _verbose_lvl = 2;
    for(const auto& itr : _paths)
    {
        auto _f = join('/', itr, _path);
        ROCPROFSYS_PATH_LOG(_verbose >= _verbose_lvl + 1,
                            "searching for '%s' in '%s' ...\n", _path.c_str(),
                            itr.c_str());
        if(exists(_f))
        {
            ROCPROFSYS_PATH_LOG(_verbose >= _verbose_lvl, "found '%s' in '%s' ...\n",
                                _path.c_str(), itr.c_str());
            return _f;
        }
    }

    for(const auto& itr : _paths)
    {
        if(std::string_view{ ::basename(itr.c_str()) }.find("lib") ==
               std::string_view::npos &&
           !dirname(itr).empty())
        {
            for(const auto* sitr : { "lib", "lib64", "../lib", "../lib64" })
            {
                auto _f = join('/', dirname(itr), sitr, _path);
                ROCPROFSYS_PATH_LOG(_verbose >= _verbose_lvl + 1,
                                    "searching for '%s' in '%s' ...\n", _path.c_str(),
                                    common::join('/', itr, sitr).c_str());
                if(exists(_f))
                {
                    ROCPROFSYS_PATH_LOG(_verbose >= _verbose_lvl,
                                        "found '%s' in '%s' ...\n", _path.c_str(),
                                        itr.c_str());
                    return _f;
                }
            }
        }
    }

    return _path;
}

std::string
dirname(const std::string& _fname)
{
    if(_fname.find('/') != std::string::npos)
        return _fname.substr(0, _fname.find_last_of('/'));
    return std::string{};
}

bool
is_link(const std::string& _path)
{
    struct stat _buffer;
    if(lstat(_path.c_str(), &_buffer) == 0) return (S_ISLNK(_buffer.st_mode) != 0);
    return false;
}

std::string
readlink(const std::string& _path)
{
    constexpr size_t MaxLen = PATH_MAX;
    // if not a symbolic link, just return the path
    if(!is_link(_path)) return _path;

    char    _buffer[MaxLen];
    ssize_t _buffer_len = MaxLen;
    _buffer_len         = ::readlink(_path.c_str(), _buffer, _buffer_len);
    if(_buffer_len < 0 || _buffer_len == (MaxLen))
    {
        auto* _path_rp = ::realpath(_path.c_str(), nullptr);
        if(_path_rp)
        {
            auto _ret = std::string{ _path_rp };
            free(_path_rp);
            return _ret;
        }
    }
    else
    {
        _buffer[_buffer_len] = '\0';
        return _buffer;
    }
    return _path;
}

std::string
realpath(const std::string& _relpath, std::string* _resolved)
{
    constexpr size_t MaxLen = PATH_MAX;
    auto             _len   = std::min<size_t>(_relpath.length(), MaxLen);

    char        _buffer[MaxLen] = { '\0' };
    const char* _result         = _buffer;

    if(::realpath(_relpath.c_str(), _buffer) == nullptr)
    {
        _result = _relpath.data();
    }

    if(_resolved)
    {
        _resolved->clear();
        _len = strnlen(_result, MaxLen);
        _resolved->resize(_len);
        for(size_t i = 0; i < _len; ++i)
            (*_resolved)[i] = _result[i];
    }

    return (_resolved) ? *_resolved : std::string{ _result };
}

bool
is_text_file(const std::string& filename)
{
    std::ifstream _file{ filename, std::ios::in | std::ios::binary };
    if(!_file.is_open())
    {
        ROCPROFSYS_PATH_LOG(0, "Error! '%s' could not be opened...\n", filename.c_str());
        return false;
    }

    constexpr size_t buffer_size = 1024;
    char             buffer[buffer_size];
    while(_file.read(buffer, sizeof(buffer)))
    {
        for(char itr : buffer)
        {
            if(itr == '\0') return false;
        }
    }

    if(_file.gcount() > 0)
    {
        for(std::streamsize i = 0; i < _file.gcount(); ++i)
        {
            if(buffer[i] == '\0') return false;
        }
    }

    return true;
}

std::vector<std::string>
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

auto
get_link_map(const char* _name, bool&& _include_self, std::vector<int>&& _open_modes)
{
    return get_link_map(_name, std::move(_open_modes), _include_self);
}

std::string
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
