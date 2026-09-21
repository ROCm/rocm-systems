// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "common/delimit.hpp"
#include <fmt/format.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <ios>
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

inline std::string
find_library(const std::string& _path, int _verbose,
             const std::string& _search_paths) ROCPROFSYS_INTERNAL_API;

[[nodiscard]] inline std::string
parent_path(std::string_view fpath, std::uint16_t levels = 1) ROCPROFSYS_INTERNAL_API;

[[nodiscard]] inline std::string
filename(std::string_view path) ROCPROFSYS_INTERNAL_API;

[[nodiscard]] inline std::string
realpath(const std::string& path) ROCPROFSYS_INTERNAL_API;

inline bool
is_text_file(const std::string& filename) ROCPROFSYS_INTERNAL_API;

[[nodiscard]] inline std::string
read_symlink(const std::string& path) ROCPROFSYS_INTERNAL_API;

[[nodiscard]] inline bool
is_directory(std::string_view path) ROCPROFSYS_INTERNAL_API;

[[nodiscard]] inline bool
is_regular_file(std::string_view path) ROCPROFSYS_INTERNAL_API;

[[nodiscard]] inline bool
create_parent_dirs_and_open_ofstream(
    std::ofstream& out_fstream, const std::string& filepath,
    std::ios::openmode mode = std::ios::out) ROCPROFSYS_INTERNAL_API;

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

std::string
find_library(const std::string& _path, int _verbose, const std::string& _search_paths)
{
    if(!_path.empty() && _path.at(0) == '/' && is_regular_file(_path))
    {
        return _path;
    }

    auto _paths = delimit(_search_paths, ":");

    constexpr int _verbose_lvl = 2;
    for(const auto& itr : _paths)
    {
        auto _f = fmt::format("{}/{}", itr, _path);
        ROCPROFSYS_PATH_LOG(_verbose >= _verbose_lvl + 1,
                            "searching for '%s' in '%s' ...\n", _path.c_str(),
                            itr.c_str());
        if(is_regular_file(_f))
        {
            ROCPROFSYS_PATH_LOG(_verbose >= _verbose_lvl, "found '%s' in '%s' ...\n",
                                _path.c_str(), itr.c_str());
            return _f;
        }
    }

    for(const auto& itr : _paths)
    {
        if(path::filename(itr).find("lib") == std::string::npos &&
           !parent_path(itr).empty())
        {
            for(const auto* sitr : { "lib", "lib64", "../lib", "../lib64" })
            {
                auto _f = fmt::format("{}/{}/{}", parent_path(itr), sitr, _path);
                ROCPROFSYS_PATH_LOG(_verbose >= _verbose_lvl + 1,
                                    "searching for '%s' in '%s' ...\n", _path.c_str(),
                                    fmt::format("{}/{}", itr, sitr).c_str());
                if(is_regular_file(_f))
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

/**
 * Get parent directory of @p fpath, walking up @p levels times.
 * Pure lexical operation: no filesystem access and no '.'/'..' resolution.
 * Absolute paths clamp at "/"; relative paths bottom out at "".
 * @code parent_path("/a/b/c", 2) @endcode returns "/a".
 * @param fpath  the path to take the parent of
 * @param levels number of components to strip (0 returns @p fpath unchanged)
 * @return the parent path, or "" / "/" at the relative / absolute limit
 */
[[nodiscard]] std::string
parent_path(std::string_view fpath, std::uint16_t levels)
{
    std::filesystem::path result{ fpath };
    for(std::uint16_t i = 0; i < levels; ++i)
    {
        auto parent = result.parent_path();
        if(parent == result) break;  // reached root ("/") or relative bottom ("")
        result = std::move(parent);
    }
    return result.string();
}

/**
 * Get the component of @p path after the last '/'.
 * Pure lexical operation: no filesystem access, no normalization.
 * @code filename("/a/b.so") @endcode returns "b.so".
 * @code filename("/a/b/") @endcode returns "".
 * @param path the path to take the filename of
 * @return the filename component, or "" if @p path ends in '/'
 */
[[nodiscard]] std::string
filename(std::string_view path)
{
    const auto pos = path.rfind('/');
    return std::string{ (pos == std::string_view::npos) ? path : path.substr(pos + 1) };
}

/** @brief Read the symbolic link target.
 *  @param path The filesystem path to inspect.
 *  @return The link target as a string, or @p path unchanged if @p path is not
 *          a symbolic link or if any filesystem error occurs.
 */
[[nodiscard]] std::string
read_symlink(const std::string& path)
{
    std::error_code error;
    auto            target = std::filesystem::read_symlink(path, error);
    return (error) ? path : target.string();
}

/**
 * @brief Check whether a path exists and is a directory.
 *
 * Follows symbolic links: a symlink that points at a directory returns true.
 * Never throws; any filesystem error (missing path, broken symlink,
 * permission failure) yields false.
 *
 * @param path Filesystem path to test.
 * @return true if @p path resolves to a directory, false otherwise.
 */
[[nodiscard]] bool
is_directory(std::string_view path)
{
    std::error_code unused_error_code;
    return std::filesystem::is_directory(path, unused_error_code);
}

/**
 * @brief Check whether a path exists and is a regular file.
 *
 * Follows symbolic links: a link that resolves to a regular file returns true,
 * a broken link returns false. Directories, device nodes, FIFOs and sockets are
 * not regular files and yield false. Never throws; any filesystem error
 * (missing path, permission failure) yields false.
 *
 * @param path Filesystem path to test.
 * @return true if @p path resolves to a regular file, false otherwise.
 */
[[nodiscard]] bool
is_regular_file(std::string_view path)
{
    std::error_code unused_error_code;
    return std::filesystem::is_regular_file(path, unused_error_code);
}

/**
 * Resolve @p path to its canonical absolute form.
 * Filesystem operation: follows symlinks and collapses '.'/'..'; the path
 * must exist. On any error (missing path, permission denied) @p path is
 * returned unchanged.
 * @code realpath("/a/./b/../c") @endcode returns "/a/c" (when it exists).
 * @param path the path to canonicalize
 * @return the canonical absolute path, or @p path unchanged on error
 */
[[nodiscard]] std::string
realpath(const std::string& path)
{
    std::error_code error;
    auto            canon = std::filesystem::canonical(path, error);
    return (error) ? path : canon.string();
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
        for(const char itr : buffer)
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

/**
 * @brief Create the parent directory of @p filepath, then open @p out_fstream on it.
 * The parent directory tree is created if absent; an already-existing
 * directory is not an error. A @p filepath with no directory component (e.g.
 * "out.txt") creates nothing and is opened relative to the current directory.
 * @param out_fstream Closed output stream to open. Left closed if the parent directory
 *                    could not be created.
 * @param filepath    Path of the file to open.
 * @param mode        Open mode forwarded to std::ofstream::open. Defaults to
 *                    std::ios::out (output only, truncating any existing file).
 * @return true if the parent directory is in place and @p out_fstream is open and good.
 */
bool
create_parent_dirs_and_open_ofstream(std::ofstream&     out_fstream,
                                     const std::string& filepath, std::ios::openmode mode)
{
    const auto parent = parent_path(filepath);
    if(!parent.empty())
    {
        std::error_code error;
        std::filesystem::create_directories(parent, error);
        if(error)
        {
            return false;
        }
    }

    out_fstream.open(filepath, mode);
    return out_fstream.is_open() && out_fstream.good();
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
        const struct link_map* next = _include_self ? _link_map : _link_map->l_next;
        while(next)
        {
            if(next->l_name != nullptr && !std::string_view{ next->l_name }.empty())
            {
                _chain.emplace_back(next->l_name);
            }
            next = next->l_next;
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
            if(is_directory(_origin)) return _origin;
        }

        if(_noload == false) dlclose(_handle);
    }

    return std::string{};
}

std::string
get_rocprofsys_root()
{
    // strip 2 levels from exe filepath to reach root
    return parent_path(realpath("/proc/self/exe"), 2);
}

std::string
get_internal_libpath(const std::string& _lib)
{
    auto _root = get_rocprofsys_root();
    for(const auto* libdir : { "lib", "lib64" })
    {
        auto _candidate = fmt::format("{}/{}/{}", _root, libdir, _lib);
        if(is_regular_file(_candidate)) return _candidate;
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

}  // namespace path
}  // namespace common
}  // namespace rocprofsys
