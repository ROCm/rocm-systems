// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "common/delimit.hpp"

#include <spdlog/fmt/fmt.h>

#include <cerrno>
#include <cstdio>
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
#include <utility>
#include <vector>

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

namespace rocprofsys::inline common::path
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

inline std::string
dirname(const std::string& _fname) ROCPROFSYS_INTERNAL_API;

inline std::string
basename(std::string_view _fname);

inline std::string
realpath(const std::string& _relpath,
         std::string*       _resolved = nullptr) ROCPROFSYS_INTERNAL_API;

inline bool
is_text_file(const std::string& filename) ROCPROFSYS_INTERNAL_API;

inline bool
is_link(const std::string& _path) ROCPROFSYS_INTERNAL_API;

inline bool
is_directory(const std::string& _path);

inline std::string
readlink(const std::string& _path) ROCPROFSYS_INTERNAL_API;

inline std::string
get_cwd();

inline int
makedir(const std::string& _dir, int umask = 0777);

template <typename... Args>
inline bool
open(std::ofstream& _ofs, std::string_view _fpath, Args&&... _args);

template <typename... Args>
inline bool
open(std::ifstream& _ifs, std::string_view _fpath, Args&&... _args);

inline std::FILE*
fopen(std::string_view _fpath, const char* _mode);

inline std::string
get_rocprofsys_root() ROCPROFSYS_INTERNAL_API;

inline std::string
get_internal_libpath(const std::string& _lib) ROCPROFSYS_INTERNAL_API;

inline std::string
get_internal_script_path() ROCPROFSYS_INTERNAL_API;

inline std::string
get_internal_libdir() ROCPROFSYS_INTERNAL_API;

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

std::string
dirname(const std::string& _fname)
{
    if(_fname.find('/') != std::string::npos)
        return _fname.substr(0, _fname.find_last_of('/'));
    return std::string{};
}

std::string
basename(std::string_view _fname)
{
    // owning last path component; matches GNU ::basename semantics (the substring after
    // the final '/'), without the aliasing/mutation hazard of glibc's char* return
    auto _pos = _fname.find_last_of('/');
    if(_pos == std::string_view::npos) return std::string{ _fname };
    return std::string{ _fname.substr(_pos + 1) };
}

bool
is_link(const std::string& _path)
{
    struct stat _buffer;
    if(lstat(_path.c_str(), &_buffer) == 0) return (S_ISLNK(_buffer.st_mode) != 0);
    return false;
}

bool
is_directory(const std::string& _path)
{
    struct stat _buffer;
    // stat (not lstat) so that a symlink to a directory is reported as a directory
    if(stat(_path.c_str(), &_buffer) == 0) return (S_ISDIR(_buffer.st_mode) != 0);
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
get_cwd()
{
    char _default_buffer[PATH_MAX];
    _default_buffer[0] = '\0';

    size_t _sz     = PATH_MAX;
    char*  _buffer = _default_buffer;

    auto _alloc_buffer = [&]() {
        if(_buffer != _default_buffer) delete[] _buffer;
        _sz *= 2;
        _buffer    = new char[_sz];
        _buffer[0] = '\0';
    };

    while(::getcwd(_buffer, _sz) == nullptr)
    {
        auto _err = errno;
        if(_err == ERANGE)
        {
            _alloc_buffer();
            continue;
        }
        else
        {
            ROCPROFSYS_PATH_LOG(1, "getcwd failed :: %s\n", strerror(_err));
            if(_buffer != _default_buffer) delete[] _buffer;
            return std::string{};
        }
    }

    auto _v = std::string{ _buffer };
    if(_buffer != _default_buffer) delete[] _buffer;
    return _v;
}

int
makedir(const std::string& _dir, int umask)
{
    if(_dir.empty()) return 0;

    if(is_directory(_dir)) return 0;

    auto _make_dir = [umask](std::string_view _v) {
        int _err = 0;
        int _ret = ::mkdir(_v.data(), umask);
        if(_ret != 0) _err = errno;
        return _err;
    };

    if(_make_dir(_dir) != 0)
    {
        std::string _base = (_dir.find('/') == 0) ? "" : get_cwd();
        for(const auto& itr : rocprofsys::common::delimit(_dir, "/"))
        {
            if(itr == ".")
            {
                continue;
            }
            else if(itr == "..")
            {
                _base = dirname(_base);
            }
            else
            {
                _base += "/" + itr;
                if(!is_directory(_base))
                {
                    auto _err = _make_dir(_base);
                    // If two competing MPI ranks call makedir() on same dir
                    // simultaneously, first rank creates dir, second gets EEXIST
                    if(_err != 0 && _err != EEXIST)
                    {
                        ROCPROFSYS_PATH_LOG(1, "mkdir(\"%s\", %i) failed: %s\n",
                                            _base.c_str(), umask, strerror(_err));
                        return -1;
                    }
                }
            }
        }
    }
    return 0;
}

template <typename... Args>
bool
open(std::ofstream& _ofs, std::string_view _fpath, Args&&... _args)
{
    std::string _file{ _fpath };
    auto        _dir  = dirname(_file);
    auto        _base = basename(_file);

    // no directory component -> open relative to the current directory
    if(_dir.empty()) _file = "./" + _base;
    // create the parent tree; on failure fall back to ./<base> in the current directory
    else if(makedir(_dir) != 0)
        _file = "./" + _base;

    _ofs.open(_file, std::forward<Args>(_args)...);
    return (_ofs && _ofs.is_open() && _ofs.good());
}

template <typename... Args>
bool
open(std::ifstream& _ifs, std::string_view _fpath, Args&&... _args)
{
    std::string _file{ _fpath };
    auto        _dir  = dirname(_file);
    auto        _base = basename(_file);

    // input stream: never create directories; only normalize a bare filename
    if(_dir.empty()) _file = "./" + _base;

    _ifs.open(_file, std::forward<Args>(_args)...);
    return (_ifs && _ifs.is_open() && _ifs.good());
}

std::FILE*
fopen(std::string_view _fpath, const char* _mode)
{
    std::string _file{ _fpath };
    auto        _dir  = dirname(_file);
    auto        _base = basename(_file);

    // no directory component -> open relative to the current directory
    if(_dir.empty()) _file = "./" + _base;
    // create the parent tree; on failure fall back to ./<base> in the current directory
    else if(makedir(_dir) != 0)
        _file = "./" + _base;

    return std::fopen(_file.c_str(), _mode);
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

std::string
get_rocprofsys_root()
{
    auto _exe_rp  = realpath("/proc/self/exe");
    auto _exe_dir = dirname(_exe_rp);
    if(_exe_dir.empty()) _exe_dir = "./";
    return fmt::format("{}/{}", _exe_dir, "..");
}

std::string
get_internal_libpath(const std::string& _lib)
{
    auto _root = get_rocprofsys_root();
    for(const auto* libdir : { "lib", "lib64" })
    {
        auto _candidate = fmt::format("{}/{}/{}", _root, libdir, _lib);
        if(exists(_candidate)) return _candidate;
    }
    return fmt::format("{}/lib/{}", _root, _lib);
}

std::string
get_internal_script_path()
{
    auto _root = get_rocprofsys_root();
    return _root + "/libexec/rocprofiler-systems";
}

std::string
get_internal_libdir()
{
    return get_rocprofsys_root() + "/lib";
}

}  // namespace rocprofsys::inline common::path
