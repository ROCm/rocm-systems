// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "common/delimit.hpp"
#include "common/env_vars.hpp"
#include "common/environment.hpp"
#include <spdlog/fmt/fmt.h>

#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <link.h>
#include <linux/limits.h>
#include <ostream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
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

template <typename RetT>
RetT
get_default_lib_search_paths()
{
    auto _paths = fmt::format("{}:{}:{}:{}:.", get_env(env_vars::PATH, ""),
                              get_env("LD_LIBRARY_PATH", ""), get_env("LIBRARY_PATH", ""),
                              get_env("PWD", ""));
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
        auto _f = fmt::format("{}/{}", itr, _path);
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
                auto _f = fmt::format("{}/{}/{}", dirname(itr), sitr, _path);
                ROCPROFSYS_PATH_LOG(_verbose >= _verbose_lvl + 1,
                                    "searching for '%s' in '%s' ...\n", _path.c_str(),
                                    fmt::format("{}/{}", itr, sitr).c_str());
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

//--------------------------------------------------------------------------------------//
//
//      New unified filesystem API (design §5) — additive; std::filesystem-based.
//      Coexists with the legacy functions above until call sites migrate.
//
//--------------------------------------------------------------------------------------//

// ---- 5.1 Path decomposition (lexical, pure, no I/O) ----

/// Parent directory, optionally walking up `levels` times. Pure lexical strip.
/// Absolute paths clamp at "/"; relative paths bottom out at "". levels==0 is identity.
[[nodiscard]] inline std::string
parent_path(std::string_view p, unsigned levels = 1)
{
    if(levels == 0) return std::string{ p };
    std::filesystem::path _p{ std::string{ p } };
    for(unsigned i = 0; i < levels; ++i)
        _p = _p.parent_path();
    return _p.string();
}

/// Final path component. filename("/a/b.so") -> "b.so". Owning => no dangling.
[[nodiscard]] inline std::string
filename(std::string_view p)
{
    return std::filesystem::path{ std::string{ p } }.filename().string();
}

/// Component without final extension. stem("/a/b.tar.gz") -> "b.tar".
[[nodiscard]] inline std::string
stem(std::string_view p)
{
    return std::filesystem::path{ std::string{ p } }.stem().string();
}

/// Final extension, WITH dot. extension("/a/b.so") -> ".so"; ("a") -> "".
[[nodiscard]] inline std::string
extension(std::string_view p)
{
    return std::filesystem::path{ std::string{ p } }.extension().string();
}

/// Collapse '.'/'..'/'//': normalize("a/./b/../c") -> "a/c". No disk access.
[[nodiscard]] inline std::string
normalize(std::string_view p)
{
    return std::filesystem::path{ std::string{ p } }.lexically_normal().string();
}

[[nodiscard]] inline bool
is_absolute(std::string_view p) noexcept
{
    return !p.empty() && p.front() == '/';
}

[[nodiscard]] inline bool
is_relative(std::string_view p) noexcept
{
    return !is_absolute(p);
}

// ---- 5.2 Path classification (string tests, C++20) ----

/// True if the final extension equals `ext` (with or without leading dot).
[[nodiscard]] inline bool
has_extension(std::string_view p, std::string_view ext) noexcept
{
    if(!ext.empty() && ext.front() == '.') ext.remove_prefix(1);
    if(ext.empty()) return false;
    if(p.size() < ext.size() + 1) return false;
    if(p[p.size() - ext.size() - 1] != '.') return false;
    return p.substr(p.size() - ext.size()) == ext;
}

/// True if the path ends with ANY of the given extensions.
[[nodiscard]] inline bool
has_any_extension(std::string_view                       p,
                  std::initializer_list<std::string_view> exts) noexcept
{
    for(auto ext : exts)
        if(has_extension(p, ext)) return true;
    return false;
}

/// If the path ends with one of `exts`, return it stripped; else unchanged.
[[nodiscard]] inline std::string
strip_known_extension(std::string_view                       p,
                      std::initializer_list<std::string_view> exts)
{
    for(auto ext : exts)
    {
        if(has_extension(p, ext))
        {
            if(!ext.empty() && ext.front() == '.') ext.remove_prefix(1);
            return std::string{ p.substr(0, p.size() - ext.size() - 1) };
        }
    }
    return std::string{ p };
}

// ---- 5.3 Path resolution (touches disk) ----

/// Symlink target (one level). Returns `p` if not a symlink. == ::readlink.
[[nodiscard]] inline std::string
read_symlink(std::string_view p)
{
    return readlink(std::string{ p });
}

// ---- 5.4 Existence & type (throwing fs + catch(...) -> fallback) ----

/// Type check: is a directory (following symlinks). == fs::is_directory.
[[nodiscard]] inline bool
is_directory(std::string_view p)
{
    try
    {
        return std::filesystem::is_directory(std::filesystem::path{ std::string{ p } });
    } catch(...)
    {
        return false;
    }
}

/// Type check: is a regular file (following symlinks). == fs::is_regular_file.
[[nodiscard]] inline bool
is_regular_file(std::string_view p)
{
    try
    {
        return std::filesystem::is_regular_file(
            std::filesystem::path{ std::string{ p } });
    } catch(...)
    {
        return false;
    }
}

/// Is a symlink (does NOT follow). == legacy is_link.
[[nodiscard]] inline bool
is_symlink(std::string_view p)
{
    return is_link(std::string{ p });
}

/// True if the file begins with the ELF magic (0x7F 'E' 'L' 'F'). FALSE on
/// non-ELF, empty, too-short, or unopenable files. Pure predicate.
[[nodiscard]] inline bool
is_elf(std::string_view p)
{
    std::ifstream _file{ std::string{ p }, std::ios::in | std::ios::binary };
    if(!_file.is_open()) return false;
    char _magic[4] = { 0, 0, 0, 0 };
    if(!_file.read(_magic, sizeof(_magic))) return false;
    return _magic[0] == 0x7F && _magic[1] == 'E' && _magic[2] == 'L' &&
           _magic[3] == 'F';
}

/// Size in bytes of a regular file. Returns 0 on missing/error AND for an empty file.
[[nodiscard]] inline std::uintmax_t
file_size_or_zero(std::string_view p)
{
    try
    {
        return std::filesystem::file_size(std::filesystem::path{ std::string{ p } });
    } catch(...)
    {
        return 0;
    }
}

// ---- 5.5 Directory operations ----

/// mkdir -p. Idempotent. Returns true if the dir exists afterward.
[[nodiscard]] inline bool
make_dirs(std::string_view p)
{
    try
    {
        std::filesystem::path _p{ std::string{ p } };
        std::filesystem::create_directories(_p);
        return std::filesystem::is_directory(_p);
    } catch(...)
    {
        return false;
    }
}

/// Ensure the PARENT directory of a file path exists. Empty parent (cwd) => true.
[[nodiscard]] inline bool
make_parent_dirs(std::string_view file_path)
{
    auto _parent = parent_path(file_path);
    if(_parent.empty()) return true;
    return make_dirs(_parent);
}

/// Remove a file/empty dir; true if gone afterward (tolerates missing).
[[nodiscard]] inline bool
remove(std::string_view p)
{
    try
    {
        std::filesystem::remove(std::filesystem::path{ std::string{ p } });
        return true;  // no throw => path is gone (removed or was already absent)
    } catch(...)
    {
        return false;
    }
}

/// Recursive remove. Returns count removed (0 on error/missing).
[[nodiscard]] inline std::uintmax_t
remove_all(std::string_view p)
{
    try
    {
        return std::filesystem::remove_all(std::filesystem::path{ std::string{ p } });
    } catch(...)
    {
        return 0;
    }
}

/// Directory entry names ('.'/'..' excluded). Empty on missing/not-a-dir.
[[nodiscard]] inline std::vector<std::string>
list_directory(std::string_view dir)
{
    std::vector<std::string> _result;
    try
    {
        for(const auto& _e : std::filesystem::directory_iterator{
                std::filesystem::path{ std::string{ dir } } })
            _result.emplace_back(_e.path().filename().string());
    } catch(...)
    {}
    return _result;
}

/// Directory entry names kept by `keep(name)`. Empty on missing/not-a-dir.
template <typename Pred>
[[nodiscard]] inline std::vector<std::string>
list_directory(std::string_view dir, Pred keep)
{
    std::vector<std::string> _result;
    try
    {
        for(const auto& _e : std::filesystem::directory_iterator{
                std::filesystem::path{ std::string{ dir } } })
        {
            auto _name = _e.path().filename().string();
            if(keep(_name)) _result.emplace_back(std::move(_name));
        }
    } catch(...)
    {}
    return _result;
}

// ---- 5.6 Streams — the auto-mkdir open shim (one template) ----

/// True if `stream` can write (ofstream/fstream) — controls the auto-mkdir branch.
template <typename S>
concept OutputStream = std::derived_from<S, std::ostream>;

/// Open `stream` on `filepath`, forwarding extra open args (e.g. ios::binary).
/// Output streams: create parent dir tree first, fall back to "./<filename>" on
/// mkdir failure. Input streams: no directory creation.
template <typename StreamT, typename... Args>
[[nodiscard]] inline bool
open(StreamT& stream, std::string_view filepath, Args&&... args)
{
    if constexpr(OutputStream<StreamT>)
    {
        std::string _target{ filepath };
        if(!make_parent_dirs(_target)) _target = "./" + filename(filepath);
        stream.open(_target, std::forward<Args>(args)...);
    }
    else
    {
        stream.open(std::string{ filepath }, std::forward<Args>(args)...);
    }
    return stream.is_open() && stream.good();
}

/// C-stream variant (auto-mkdir parent + "./<filename>" fallback).
[[nodiscard]] inline std::FILE*
fopen(std::string_view filepath, const char* mode)
{
    std::string _target{ filepath };
    if(!make_parent_dirs(_target)) _target = "./" + filename(filepath);
    return ::fopen(_target.c_str(), mode);
}

// ---- 5.7 Process / environment paths ----

/// $TMPDIR else /tmp.
[[nodiscard]] inline std::string
temp_dir()
{
    if(const char* _env = std::getenv("TMPDIR"); _env != nullptr && _env[0] != '\0')
        return std::string{ _env };
    return std::string{ "/tmp" };
}

/// realpath("/proc/self/exe").
[[nodiscard]] inline std::string
executable_path()
{
    return realpath("/proc/self/exe");
}

/// First existing "dir/name" (optionally trying lib/lib64 subdirs); input if none.
[[nodiscard]] inline std::string
find_in_dirs(std::string_view name, const std::vector<std::string>& dirs,
             bool try_lib_subdirs = false)
{
    for(const auto& _d : dirs)
    {
        auto _cand = fmt::format("{}/{}", _d, name);
        if(exists(_cand)) return _cand;
    }
    if(try_lib_subdirs)
    {
        for(const auto& _d : dirs)
            for(const auto* _sub : { "lib", "lib64" })
            {
                auto _cand = fmt::format("{}/{}/{}", _d, _sub, name);
                if(exists(_cand)) return _cand;
            }
    }
    return std::string{ name };
}

}  // namespace path
}  // namespace common
}  // namespace rocprofsys
