// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include <cstdint>

#include "common/join.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <timemory/utility/filepath.hpp>
#include <type_traits>
#include <unistd.h>
#include <unordered_set>

#if !defined(ROCPROFSYS_ENVIRON_LOG_NAME)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_NAME)
#        define ROCPROFSYS_ENVIRON_LOG_NAME "[" ROCPROFSYS_COMMON_LIBRARY_NAME "]"
#    else
#        define ROCPROFSYS_ENVIRON_LOG_NAME
#    endif
#endif

#if !defined(ROCPROFSYS_ENVIRON_LOG_START)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_LOG_START)
#        define ROCPROFSYS_ENVIRON_LOG_START ROCPROFSYS_COMMON_LIBRARY_LOG_START
#    elif defined(TIMEMORY_LOG_COLORS_AVAILABLE)
#        define ROCPROFSYS_ENVIRON_LOG_START                                             \
            fprintf(stderr, "%s", ::tim::log::color::info());
#    else
#        define ROCPROFSYS_ENVIRON_LOG_START
#    endif
#endif

#if !defined(ROCPROFSYS_ENVIRON_LOG_END)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_LOG_END)
#        define ROCPROFSYS_ENVIRON_LOG_END ROCPROFSYS_COMMON_LIBRARY_LOG_END
#    elif defined(TIMEMORY_LOG_COLORS_AVAILABLE)
#        define ROCPROFSYS_ENVIRON_LOG_END                                               \
            fprintf(stderr, "%s", ::tim::log::color::end());
#    else
#        define ROCPROFSYS_ENVIRON_LOG_END
#    endif
#endif

#define ROCPROFSYS_ENVIRON_LOG(CONDITION, ...)                                           \
    if(CONDITION)                                                                        \
    {                                                                                    \
        fflush(stderr);                                                                  \
        ROCPROFSYS_ENVIRON_LOG_START                                                     \
        fprintf(stderr, "[rocprof-sys]" ROCPROFSYS_ENVIRON_LOG_NAME "[%i] ", getpid());  \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ROCPROFSYS_ENVIRON_LOG_END                                                       \
        fflush(stderr);                                                                  \
    }

namespace rocprofsys
{
inline namespace common
{
namespace
{

inline std::string
get_env_impl(std::string_view env_id, std::string_view _default)
{
    if(env_id.empty()) return std::string{ _default };
    char* env_var = ::std::getenv(env_id.data());
    if(env_var) return std::string{ env_var };
    return std::string{ _default };
}

inline std::string
get_env_impl(std::string_view env_id, const char* _default)
{
    return get_env_impl(env_id, std::string_view{ _default });
}

inline int
get_env_impl(std::string_view env_id, int _default)
{
    if(env_id.empty()) return _default;
    char* env_var = ::std::getenv(env_id.data());
    if(env_var)
    {
        try
        {
            return std::stoi(env_var);
        } catch(std::exception& _e)
        {
            fprintf(stderr,
                    "[rocprof-sys][get_env] Exception thrown converting getenv(\"%s\") = "
                    "%s to integer :: %s. Using default value of %i\n",
                    env_id.data(), env_var, _e.what(), _default);
        }
        return _default;
    }
    return _default;
}

inline bool
get_env_impl(std::string_view env_id, bool _default)
{
    if(env_id.empty()) return _default;
    char* env_var = ::std::getenv(env_id.data());
    if(env_var)
    {
        if(std::string_view{ env_var }.empty())
            throw std::runtime_error(std::string{ "No boolean value provided for " } +
                                     std::string{ env_id });

        if(std::string_view{ env_var }.find_first_not_of("0123456789") ==
           std::string_view::npos)
        {
            return static_cast<bool>(std::stoi(env_var));
        }
        else
        {
            for(size_t i = 0; i < strlen(env_var); ++i)
                env_var[i] = tolower(env_var[i]);
            for(const auto& itr : { "off", "false", "no", "n", "f", "0" })
                if(strcmp(env_var, itr) == 0) return false;
        }
        return true;
    }
    return _default;
}
}  // namespace

template <typename Tp>
inline auto
get_env(std::string_view env_id, Tp&& _default)
{
    if constexpr(std::is_enum<Tp>::value)
    {
        using Up = std::underlying_type_t<Tp>;
        // cast to underlying type -> get_env -> cast to enum type
        return static_cast<Tp>(get_env_impl(env_id, static_cast<Up>(_default)));
    }
    else
    {
        return get_env_impl(env_id, std::forward<Tp>(_default));
    }
}

struct ROCPROFSYS_INTERNAL_API env_config
{
    std::string env_name  = {};
    std::string env_value = {};
    int         override  = 0;

    auto operator()(bool _verbose = false) const
    {
        if(env_name.empty()) return -1;
        ROCPROFSYS_ENVIRON_LOG(_verbose, "setenv(\"%s\", \"%s\", %i)\n", env_name.c_str(),
                               env_value.c_str(), override);
        return setenv(env_name.c_str(), env_value.c_str(), override);
    }
};

inline void
remove_env(std::vector<std::string>& _environ, std::string_view _env_var,
           const std::unordered_set<std::string>& _original_envs)
{
    auto key = join("", _env_var, "=");

    _environ.erase(std::remove_if(_environ.begin(), _environ.end(),
                                  [&key](const std::string& entry) {
                                      return std::string_view{ entry }.find(key) == 0;
                                  }),
                   _environ.end());

    // Restore from original_envs if previously existed
    for(const auto& orig : _original_envs)
    {
        if(std::string_view{ orig }.find(key) == 0) _environ.emplace_back(orig);
    }
}

inline std::string
discover_llvm_libdir_for_ompt(bool verbose = false)
{
    auto strip = [](std::string s) {
        if(!s.empty() && s.back() == '/') s.pop_back();
        return s;
    };

    // Common ROCm envs
    const auto rocm_dir  = strip(get_env<std::string>("ROCM_PATH", "/opt/rocm"));
    const auto rocmv_dir = strip(get_env<std::string>("ROCmVersion_DIR", ""));

    std::vector<std::string> candidates;
    candidates.reserve(6);

    auto push_unique = [&](const std::string& p) {
        if(p.empty()) return;
        if(std::find(candidates.begin(), candidates.end(), p) == candidates.end())
            candidates.emplace_back(p);
    };

    if(!rocmv_dir.empty())
    {
        push_unique(rocmv_dir + "/llvm/lib");
        push_unique(rocmv_dir + "/lib");
    }
    push_unique(rocm_dir + "/llvm/lib");
    push_unique(rocm_dir + "/lib/llvm/lib");
    push_unique("/opt/rocm/llvm/lib");
    push_unique("/opt/rocm/lib/llvm/lib");

    auto has_libomptarget = [](const std::string& dir) {
        const std::string so = dir + "/libomptarget.so";
        return ::tim::filepath::exists(so);
    };

    // Pick the first candidate that contains libomptarget.so
    auto it = std::find_if(candidates.begin(), candidates.end(), has_libomptarget);
    if(it != candidates.end())
    {
        ROCPROFSYS_ENVIRON_LOG(verbose, "Using LLVM libdir: %s\n", it->c_str());
        return *it;
    }

    ROCPROFSYS_ENVIRON_LOG(verbose,
                           "libomptarget.so not found in candidate LLVM libdirs\n");
    return {};
}

inline bool
is_python_interpreter(std::string_view executable)
{
    if(executable.empty()) return false;

    const auto slash_pos = executable.rfind('/');
    const auto basename  = (slash_pos != std::string_view::npos)
                               ? executable.substr(slash_pos + 1)
                               : executable;

    if(basename == "python" || basename == "python3") return true;

    constexpr std::string_view python3_prefix = "python3.";

    const bool has_valid_prefix =
        basename.size() > python3_prefix.size() &&
        basename.substr(0, python3_prefix.size()) == python3_prefix;
    if(!has_valid_prefix) return false;

    const auto version_digits = basename.substr(python3_prefix.size());

    return std::all_of(version_digits.begin(), version_digits.end(),
                       [](unsigned char c) { return std::isdigit(c); });
}

inline std::string
discover_torch_libpath(const std::string& python_binary, bool verbose = false)
{
    if(python_binary.empty()) return {};

    const auto is_safe_executable_path = [](const std::string& path) {
        // Allow only a conservative set of characters in the executable path to
        // avoid injection when used in a shell command.
        for(unsigned char c : path)
        {
            if(std::isalnum(c) != 0) continue;
            switch(c)
            {
                case '/':
                case '.':
                case '_':
                case '-':
                case '+': break;
                default: return false;
            }
        }
        return true;
    };

    if(!is_safe_executable_path(python_binary))
    {
        ROCPROFSYS_ENVIRON_LOG(
            verbose, "Unsafe characters detected in Python interpreter path: %s\n",
            python_binary.c_str());
        return {};
    }

    const auto cmd = "\"" + python_binary +
                     "\" -c \"import torch; print(torch.__path__[0])\" 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if(!pipe)
    {
        ROCPROFSYS_ENVIRON_LOG(verbose, "Failed to execute command: %s\n", cmd.c_str());
        return {};
    }

    char        buffer[1024];
    std::string result;
    while(fgets(buffer, sizeof(buffer), pipe))
    {
        result.append(buffer);
        // stop if we've read the full line (torch path is printed on a single line)
        if(!result.empty() && result.back() == '\n') break;
    }

    int status = pclose(pipe);

    if(status != 0 || result.empty())
    {
        ROCPROFSYS_ENVIRON_LOG(verbose, "torch not found for Python interpreter: %s\n",
                               python_binary.c_str());
        return {};
    }

    while(!result.empty() &&
          (result.back() == '\n' || result.back() == '\r' || result.back() == ' '))
    {
        result.pop_back();
    }

    if(result.empty()) return {};

    std::string torch_libdir = result + "/lib";

    if(!::tim::filepath::direxists(torch_libdir))
    {
        ROCPROFSYS_ENVIRON_LOG(verbose, "torch lib directory does not exist: %s\n",
                               torch_libdir.c_str());
        return {};
    }

    ROCPROFSYS_ENVIRON_LOG(verbose, "Discovered torch library path: %s\n",
                           torch_libdir.c_str());
    return torch_libdir;
}

enum class update_mode : std::uint8_t
{
    REPLACE = 0,
    PREPEND,
    APPEND,
    WEAK,
};

template <typename Tp>
inline std::string
to_env_string(Tp&& val)
{
    using T = std::decay_t<Tp>;
    static_assert(std::is_same_v<T, std::string> || std::is_same_v<T, const char*> ||
                      std::is_same_v<T, bool> || std::is_arithmetic_v<T>,
                  "to_env_string: unsupported type. Use string, bool, or numeric types.");

    if constexpr(std::is_same_v<T, std::string> || std::is_same_v<T, const char*>)
        return std::string{ val };
    else if constexpr(std::is_same_v<T, bool>)
        return val ? "true" : "false";
    else
        return std::to_string(val);
}

template <typename Tp, typename UpdatedEnvsT>
inline void
update_env(std::vector<std::string>& _environ, std::string_view _env_var, Tp&& _env_val,
           update_mode _mode, std::string_view _join_delim, UpdatedEnvsT& _updated_envs,
           const std::unordered_set<std::string>& _original_envs)
{
    using updated_value_t = typename UpdatedEnvsT::value_type;
    _updated_envs.emplace(updated_value_t{ _env_var });

    const auto _env_val_str = to_env_string(std::forward<Tp>(_env_val));
    const auto _key         = join("", _env_var, "=");

    const auto matches_key = [&_key](const std::string& entry) {
        return std::string_view{ entry }.find(_key) == 0;
    };

    auto first = std::find_if(_environ.begin(), _environ.end(), matches_key);
    if(first == _environ.end())
    {
        _environ.emplace_back(join('=', _env_var, _env_val_str));
        return;
    }

    switch(_mode)
    {
        case update_mode::WEAK:
            if(_original_envs.find(*first) == _original_envs.end()) return;
            *first = join('=', _env_var, _env_val_str);
            return;

        case update_mode::PREPEND:
        case update_mode::APPEND:
        {
            if(first->find(_env_val_str) != std::string::npos) return;
            auto _val = first->substr(_key.size());
            *first    = (_mode == update_mode::PREPEND)
                            ? join('=', _env_var, join(_join_delim, _env_val_str, _val))
                            : join('=', _env_var, join(_join_delim, _val, _env_val_str));
            return;
        }

        case update_mode::REPLACE:
            *first = join('=', _env_var, _env_val_str);
            _environ.erase(std::remove_if(std::next(first), _environ.end(), matches_key),
                           _environ.end());
            return;
    }
}

template <typename UpdatedEnvsT>
inline void
add_torch_library_path(std::vector<std::string>& envp, std::string_view executable,
                       bool verbose, UpdatedEnvsT& updated_envs)
{
    if(executable.empty()) return;
    if(!is_python_interpreter(executable)) return;

    auto torch_libpath = discover_torch_libpath(std::string{ executable }, verbose);
    if(torch_libpath.empty()) return;

    std::unordered_set<std::string> seen{ torch_libpath };
    std::string                     result = torch_libpath;

    constexpr std::string_view ld_prefix = "LD_LIBRARY_PATH=";

    auto is_ld_path = [&](const std::string& entry) {
        return std::string_view{ entry }.substr(0, ld_prefix.length()) == ld_prefix;
    };

    for(const auto& entry : envp)
    {
        if(!is_ld_path(entry)) continue;

        std::istringstream stream{ entry.substr(ld_prefix.length()) };
        for(std::string path; std::getline(stream, path, ':');)
        {
            if(!path.empty() && seen.insert(path).second) result += ":" + path;
        }
    }

    envp.erase(std::remove_if(envp.begin(), envp.end(), is_ld_path), envp.end());
    envp.emplace_back(join("", ld_prefix, result));

    updated_envs.emplace(ld_prefix.substr(0, ld_prefix.length() - 1));
}

/// @brief Consolidates duplicate environment variable entries by merging their values.
///
/// When building an environment for execve(), multiple entries for the same variable
/// may accumulate. This function merges them into single entries with unique values.
///
/// For most variables, values are split and joined using ':' (e.g., PATH,
/// LD_LIBRARY_PATH). Certain variables that use ':' in their value syntax use ',' as the
/// delimiter instead.
///
/// @param envp Vector of environment strings in "KEY=VALUE" format. Modified in place.
///
/// Example transformations:
///   - PATH=/usr/bin + PATH=/usr/local/bin -> PATH=/usr/bin:/usr/local/bin
///   - ROCPROFSYS_PAPI_EVENTS=perf::A + ROCPROFSYS_PAPI_EVENTS=perf::B
///         -> ROCPROFSYS_PAPI_EVENTS=perf::A,perf::B
inline void
consolidate_env_entries(std::vector<std::string>& envp)
{
    /// Returns the appropriate delimiter character for splitting/joining values.
    /// Most variables use ':' (like PATH), but some use ':' in their value syntax
    /// and need ',' instead:
    /// - ROCPROFSYS_PAPI_EVENTS: uses perf::EVENT_NAME or net:::interface:metric syntax
    /// - ROCPROFSYS_SAMPLING_OVERFLOW_EVENT: uses perf::EVENT_NAME syntax
    /// - ROCPROFSYS_ROCM_EVENTS: uses EVENT_NAME:device=N syntax
    auto get_delimiter = [](std::string_view key) -> char {
        if(key == "ROCPROFSYS_PAPI_EVENTS" ||
           key == "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT" || key == "ROCPROFSYS_ROCM_EVENTS")
            return ',';
        return ':';
    };

    /// Stores the parsed and deduplicated parts for a single environment variable.
    struct key_data
    {
        std::vector<std::string> parts;  ///< Unique value parts in order of appearance
        std::unordered_set<std::string> seen;  ///< Tracks seen parts for deduplication
        char                            delim = ':';  ///< Delimiter for this variable

        /// Adds a part if non-empty and not already seen.
        void add_unique(std::string part)
        {
            if(!part.empty() && seen.insert(part).second)
                parts.emplace_back(std::move(part));
        }
    };

    /// Parses an environment entry string into key and value components.
    /// @param entry String in "KEY=VALUE" format
    /// @return Optional pair of (key, value) views, or nullopt if no '=' found
    auto parse_entry = [](std::string_view entry)
        -> std::optional<std::pair<std::string_view, std::string_view>> {
        auto eq_pos = entry.find('=');
        if(eq_pos == std::string_view::npos) return std::nullopt;
        return std::make_pair(entry.substr(0, eq_pos), entry.substr(eq_pos + 1));
    };

    /// Reconstructs an environment entry string from key and value parts.
    /// @param key   The environment variable name
    /// @param parts The deduplicated value components (may be empty)
    /// @param delim The delimiter to use when joining parts
    /// @return String in "KEY=part1<delim>part2<delim>..." format, or "KEY="
    ///         when @p parts is empty.
    auto join_parts = [](std::string_view key, const std::vector<std::string>& parts,
                         char delim) {
        std::string result;
        result.reserve(key.size() + 1);
        result.append(key);
        result += '=';

        if(parts.empty()) return result;

        std::size_t total_parts_length = 0;
        for(const auto& part : parts)
            total_parts_length += part.size();

        result.reserve(result.size() + total_parts_length + (parts.size() - 1));

        bool first = true;
        for(const auto& part : parts)
        {
            if(!first) result += delim;
            result.append(part);
            first = false;
        }

        return result;
    };

    std::unordered_map<std::string_view, key_data> key_map;
    std::vector<std::string_view>                  key_order;

    // Phase 1: Parse all entries and aggregate values by key
    for(const auto& entry : envp)
    {
        auto parsed = parse_entry(entry);
        if(!parsed)
        {
            continue;
        }

        auto [key, value] = *parsed;

        // Create new entry if key not seen before, recording its delimiter
        auto [it, inserted] = key_map.try_emplace(key);
        if(inserted)
        {
            key_order.emplace_back(key);
            it->second.delim = get_delimiter(key);
        }

        // Split value by delimiter and add unique parts
        auto&              data = it->second;
        std::istringstream stream{ std::string{ value } };
        for(std::string part; std::getline(stream, part, data.delim);)
        {
            data.add_unique(part);
        }
    }

    // Phase 2: Build consolidated result
    std::vector<std::string> result;
    result.reserve(key_order.size());

    for(auto key : key_order)
    {
        const auto& data = key_map[key];
        result.emplace_back(join_parts(key, data.parts, data.delim));
    }

    envp = std::move(result);
}

}  // namespace common
}  // namespace rocprofsys
