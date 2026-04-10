// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <timemory/utility/filepath.hpp>

#include "logger/debug.hpp"
#include <spdlog/fmt/fmt.h>

namespace rocprofsys
{
inline namespace common
{
namespace
{

inline bool
parse_bool(std::string_view val)
{
    if(val.empty()) throw std::runtime_error(fmt::format("No boolean value provided"));

    const std::array<const char*, 6> falsy  = { "off", "false", "no", "n", "f", "0" };
    const std::array<const char*, 6> truthy = { "on", "true", "yes", "y", "t", "1" };

    std::string lower(val);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    const auto is_falsy  = std::any_of(falsy.begin(), falsy.end(),
                                       [&](const char* value) { return value == lower; });
    const auto is_truthy = std::any_of(truthy.begin(), truthy.end(),
                                       [&](const char* value) { return value == lower; });

    if(!is_truthy && !is_falsy)
        throw std::runtime_error(fmt::format("Invalid boolean value: {}", val));

    return is_truthy;
}

template <typename T, typename Fn>
inline T
try_parse_value(std::string_view env_id, const char* env_var, T _default,
                Fn&& parse_fn) noexcept
{
    if(!env_var) return _default;
    try
    {
        return parse_fn(env_var);
    } catch(const std::exception& _e)
    {
        LOG_ERROR("Exception thrown converting getenv(\"{}\") = {} :: {}. "
                  "Using default value of {}",
                  env_id, env_var, _e.what(), _default);
        return _default;
    }
}

}  // namespace

template <typename Tp>
inline auto
get_env(std::string_view env_id, Tp&& _default) noexcept
{
    using T = std::decay_t<Tp>;

    static_assert(std::is_enum_v<T> || std::is_same_v<T, bool> ||
                      std::is_same_v<T, int> || std::is_convertible_v<T, std::string>,
                  "get_env: unsupported type");

    if(env_id.empty())
    {
        if constexpr(std::is_convertible_v<T, std::string>)
            return std::string{ _default };
        else
            return static_cast<T>(_default);
    }

    if constexpr(std::is_enum_v<T>)
    {
        using Up = std::underlying_type_t<T>;
        return static_cast<T>(get_env(env_id, static_cast<Up>(_default)));
    }

    const char* env_var = ::std::getenv(env_id.data());

    if constexpr(std::is_same_v<T, bool>)
    {
        return try_parse_value(env_id, env_var, _default,
                               [](const char* v) { return parse_bool(v); });
    }

    if constexpr(std::is_same_v<T, int>)
    {
        return try_parse_value(env_id, env_var, _default,
                               [](const char* v) { return std::stoi(v); });
    }

    if constexpr(std::is_convertible_v<T, std::string>)
    {
        return env_var ? std::string{ env_var } : std::string{ _default };
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
        if(_verbose)
            LOG_INFO("setenv(\"{}\", \"{}\", {})", env_name, env_value, override);
        return setenv(env_name.c_str(), env_value.c_str(), override);
    }
};

inline void
remove_env(std::vector<char*>& _environ, std::string_view _env_var,
           const std::unordered_set<std::string>& _original_envs)
{
    auto key = fmt::format("{}=", _env_var);

    auto match = [&key](auto itr) -> bool {
        return itr && std::string_view{ itr }.find(key) == 0;
    };

    // Free memory for matching entries
    for(auto& itr : _environ)
    {
        if(match(itr))
        {
            std::free(itr);
            itr = nullptr;
        }
    }

    // Remove null entries
    _environ.erase(std::remove_if(_environ.begin(), _environ.end(),
                                  [](const char* ptr) { return ptr == nullptr; }),
                   _environ.end());

    // Restore from original_envs if previously existed
    for(const auto& orig : _original_envs)
    {
        if(std::string_view{ orig.data(), orig.size() }.find(key) == 0)
        {
            _environ.emplace_back(strdup(orig.c_str()));
        }
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
        if(verbose) LOG_INFO("Using LLVM libdir: {}", *it);
        return *it;
    }

    if(verbose) LOG_INFO("libomptarget.so not found in candidate LLVM libdirs");
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
        for(const unsigned char c : path)
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
        if(verbose)
            LOG_INFO("Unsafe characters detected in Python interpreter path: {}",
                     python_binary);
        return {};
    }

    const auto cmd = "\"" + python_binary +
                     "\" -c \"import torch; print(torch.__path__[0])\" 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if(!pipe)
    {
        if(verbose) LOG_INFO("Failed to execute command: {}", cmd);
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

    const int status = pclose(pipe);

    if(status != 0 || result.empty())
    {
        if(verbose) LOG_INFO("torch not found for Python interpreter: {}", python_binary);
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
        if(verbose) LOG_INFO("torch lib directory does not exist: {}", torch_libdir);
        return {};
    }

    if(verbose) LOG_INFO("Discovered torch library path: {}", torch_libdir);
    return torch_libdir;
}

enum class update_mode : uint8_t
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

/// Updates or adds an environment variable in _environ.
/// Modes: REPLACE (overwrite, remove duplicates), PREPEND/APPEND (merge values),
/// WEAK (overwrite only if value was in _original_envs).
template <typename Tp>
inline void
update_env(std::vector<char*>& _environ, std::string_view _env_var, Tp&& _env_val,
           update_mode _mode, std::string_view _join_delim,
           std::unordered_set<std::string_view>&  _updated_envs,
           const std::unordered_set<std::string>& _original_envs)
{
    _updated_envs.emplace(_env_var);

    const bool _prepend  = (_mode == update_mode::PREPEND);
    const bool _append   = (_mode == update_mode::APPEND);
    const bool _weak_upd = (_mode == update_mode::WEAK);

    auto _env_val_str = to_env_string(std::forward<Tp>(_env_val));
    auto _new_entry   = fmt::format("{}={}", _env_var, _env_val_str);
    auto _key         = fmt::format("{}=", _env_var);

    bool _found_match = false;

    for(auto it = _environ.begin(); it != _environ.end();)
    {
        auto* itr = *it;
        if(!itr || std::string_view{ itr }.find(_key) != 0)
        {
            ++it;
            continue;
        }

        // WEAK: only update if this entry was in the original environment
        if(_weak_upd && _original_envs.find(std::string{ itr }) == _original_envs.end())
            return;

        if(_prepend || _append)
        {
            _found_match = true;
            // Merge new value into existing (prepend or append) if not already present
            if(std::string_view{ itr }.find(_env_val_str) == std::string_view::npos)
            {
                auto _val = std::string{ itr }.substr(_key.length());
                auto _merged =
                    _prepend ? fmt::format("{}{}{}", _env_val_str, _join_delim, _val)
                             : fmt::format("{}{}{}", _val, _join_delim, _env_val_str);
                auto* _new = strdup(fmt::format("{}={}", _env_var, _merged).c_str());
                if(!_new) throw std::bad_alloc{};
                std::free(itr);
                *it = _new;
            }
            ++it;
        }
        else
        {
            // REPLACE or WEAK: overwrite with new value
            if(_weak_upd)
            {
                auto* _new = strdup(_new_entry.c_str());
                if(!_new) throw std::bad_alloc{};
                std::free(itr);
                *it = _new;
                return;
            }

            // Only REPLACE reaches here (_weak_upd returned above)
            if(_found_match)
            {
                std::free(itr);
                it = _environ.erase(it);
                continue;
            }
            auto* _new = strdup(_new_entry.c_str());
            if(!_new) throw std::bad_alloc{};
            std::free(itr);
            *it          = _new;
            _found_match = true;
            ++it;
        }
    }
    // Add new entry if variable was not found
    if(!_found_match) _environ.emplace_back(strdup(_new_entry.c_str()));
}

template <typename UpdatedEnvsT>
inline void
add_torch_library_path(std::vector<char*>& envp, const std::vector<char*>& argv,
                       bool verbose, UpdatedEnvsT& updated_envs)
{
    if(argv.empty() || argv.front() == nullptr) return;
    if(!is_python_interpreter(argv.front())) return;

    auto torch_libpath = discover_torch_libpath(argv.front(), verbose);
    if(torch_libpath.empty()) return;

    std::unordered_set<std::string> seen{ torch_libpath };
    std::string                     result = torch_libpath;

    constexpr std::string_view ld_prefix = "LD_LIBRARY_PATH=";

    auto is_ld_path = [&](char* entry) {
        return entry &&
               std::string_view{ entry }.substr(0, ld_prefix.length()) == ld_prefix;
    };

    for(auto& entry : envp)
    {
        if(!is_ld_path(entry)) continue;

        std::istringstream stream{ std::string{ entry + ld_prefix.length() } };
        for(std::string path; std::getline(stream, path, ':');)
        {
            if(!path.empty() && seen.insert(path).second) result += ":" + path;
        }

        std::free(entry);
        entry = nullptr;
    }

    envp.erase(std::remove(envp.begin(), envp.end(), nullptr), envp.end());
    envp.emplace_back(strdup(fmt::format("{}{}", ld_prefix, result).c_str()));

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
///             Original strings are freed; new strings are allocated with strdup().
///
/// Example transformations:
///   - PATH=/usr/bin + PATH=/usr/local/bin -> PATH=/usr/bin:/usr/local/bin
///   - ROCPROFSYS_PAPI_EVENTS=perf::A + ROCPROFSYS_PAPI_EVENTS=perf::B
///         -> ROCPROFSYS_PAPI_EVENTS=perf::A,perf::B
inline void
consolidate_env_entries(std::vector<char*>& envp)
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
    /// @param parts The deduplicated value components
    /// @param delim The delimiter to use when joining parts
    /// @return String in "KEY=part1<delim>part2<delim>..." format
    auto join_parts = [](std::string_view key, const std::vector<std::string>& parts,
                         char delim) {
        std::string result;

        const auto total_parts_length = std::accumulate(
            parts.begin(), parts.end(), std::size_t{ 0 },
            [](std::size_t acc, const std::string& part) { return acc + part.size(); });

        const auto delim_count       = parts.size() - 1;
        const auto equal_sign_length = 1;

        result.reserve(key.size() + equal_sign_length + total_parts_length + delim_count);
        result.append(key);
        result += '=';

        // Join all parts with the delimiter
        result =
            std::accumulate(parts.begin(), parts.end(), std::move(result),
                            [delim, &parts](std::string acc, const std::string& part) {
                                if(part != parts.front()) acc += delim;
                                acc.append(part);
                                return acc;
                            });

        return result;
    };

    std::unordered_map<std::string_view, key_data> key_map;
    std::vector<std::string_view>                  key_order;

    // Phase 1: Parse all entries and aggregate values by key
    for(auto* entry : envp)
    {
        if(!entry)
        {
            continue;
        }

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
    std::vector<char*> result;
    result.reserve(key_order.size());

    for(auto key : key_order)
    {
        const auto& data = key_map[key];
        result.emplace_back(strdup(join_parts(key, data.parts, data.delim).c_str()));
    }

    // Phase 3: Free original entries and replace with consolidated result
    for(auto* entry : envp)
    {
        std::free(entry);
        entry = nullptr;
    }

    envp = std::move(result);
}

}  // namespace common
}  // namespace rocprofsys
