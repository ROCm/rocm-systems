// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "common/join.hpp"
#include "logger/debug.hpp"

#include <timemory/utility/filepath.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <set>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <string_view>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocprofsys
{
inline namespace common
{

// POSIX environment backend — calls the real ::getenv / ::setenv.
struct posix_env
{
    static int setenv(const char* name, const char* value, int overwrite)
    {
        return ::setenv(name, value, overwrite);
    }

    static char* getenv(const char* name) { return ::getenv(name); }
};

// environment<EnvType> — all env-read/write logic parameterised over the backend.
// Use environment<posix_env> (the default) for production code.
// Inject a fake backend in unit tests to avoid touching the real process environment.
template <typename EnvType = posix_env>
struct environment
{
private:
    // env_id is always constructed from a null-terminated string at every call site
    static const char* fetch_raw_env(std::string_view env_id)
    {
        if(env_id.empty()) return nullptr;
        return EnvType::getenv(
            env_id.data());  // NOLINT(bugprone-suspicious-stringview-data-usage)
    }

    template <typename Tp>
    static std::string get_env_string(std::string_view env_id, const Tp& fallback)
    {
        const char* raw = fetch_raw_env(env_id);
        return raw ? std::string{ raw } : std::string{ fallback };
    }

    static bool get_env_bool(std::string_view env_id, bool fallback)
    {
        const char* raw = fetch_raw_env(env_id);
        if(!raw) return fallback;

        const std::string_view env_sv{ raw };
        if(env_sv.empty())
        {
            throw std::runtime_error(
                std::string{ "No boolean value provided for " }.append(env_id));
        }

        if(env_sv.find_first_not_of("0123456789") == std::string_view::npos)
        {
            return static_cast<bool>(std::stoi(raw));
        }

        std::string lower{ env_sv };
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char chr) { return std::tolower(chr); });

        constexpr auto false_values = std::array{
            std::string_view{ "off" }, std::string_view{ "false" },
            std::string_view{ "no" },  std::string_view{ "n" },
            std::string_view{ "f" },   std::string_view{ "0" },
        };
        return !std::any_of(false_values.begin(), false_values.end(),
                            [&lower](std::string_view val) { return lower == val; });
    }

    template <typename Tp>
    static Tp get_env_float(std::string_view env_id, Tp fallback)
    {
        const char* raw = fetch_raw_env(env_id);
        if(!raw) return fallback;
        try
        {
            return static_cast<Tp>(std::stod(raw));
        } catch(const std::exception& exc)
        {
            LOG_ERROR("[get_env] Exception thrown converting getenv(\"{}\") = {} to "
                      "float: {}",
                      env_id, raw, exc.what());
        }
        return fallback;
    }

    template <typename Tp>
    static Tp get_env_integral(std::string_view env_id, Tp fallback)
    {
        const char* raw = fetch_raw_env(env_id);
        if(!raw) return fallback;

        // Trim surrounding whitespace so values such as " 42 " still parse.
        constexpr std::string_view whitespace = " \t\n\r\f\v";
        std::string_view           token{ raw };
        const auto                 first = token.find_first_not_of(whitespace);
        if(first == std::string_view::npos)
        {
            LOG_ERROR("[get_env] Cannot convert empty getenv(\"{}\") to integer", env_id);
            return fallback;
        }
        token = token.substr(first, token.find_last_not_of(whitespace) - first + 1);

        // std::from_chars parses against the exact target type: it rejects a
        // leading '-' for unsigned Tp and reports result_out_of_range, so
        // negative or overflowing input falls back to the default instead of
        // silently wrapping or truncating. Signed Tp still accepts '-'.
        Tp          value{};
        const auto* end      = token.data() + token.size();
        const auto [ptr, ec] = std::from_chars(token.data(), end, value);
        if(ec == std::errc{} && ptr == end) return value;

        LOG_ERROR("[get_env] Failed to convert getenv(\"{}\") = \"{}\" to integer",
                  env_id, raw);
        return fallback;
    }

    template <typename Tp>
    static auto get_env_impl(std::string_view env_id, const Tp& fallback)
    {
        if constexpr(std::is_same_v<std::decay_t<Tp>, std::string> ||
                     std::is_same_v<std::decay_t<Tp>, std::string_view> ||
                     std::is_same_v<std::decay_t<Tp>, const char*> ||
                     std::is_same_v<std::decay_t<Tp>, char*>)
        {
            return get_env_string(env_id, fallback);
        }
        else if constexpr(std::is_same_v<Tp, bool>)
        {
            return get_env_bool(env_id, fallback);
        }
        else if constexpr(std::is_floating_point_v<Tp>)
        {
            return get_env_float(env_id, fallback);
        }
        else
        {
            return get_env_integral(env_id, fallback);
        }
    }

public:
    template <typename Tp>
    static auto get_env(std::string_view env_id, Tp&& value_default)
    {
        if constexpr(std::is_enum_v<Tp>)
        {
            using up_t = std::underlying_type_t<Tp>;
            return static_cast<Tp>(
                get_env_impl(env_id, static_cast<up_t>(value_default)));
        }
        else
        {
            return get_env_impl(env_id, std::forward<Tp>(value_default));
        }
    }

    template <typename Tp = std::string>
    static auto get_env(std::string_view env_id)
    {
        return get_env(env_id, Tp{});
    }

    template <typename Tp>
    static void set_env(const std::string& env_var, const Tp& value, int override)
    {
        std::stringstream ss_val;
        ss_val << value;
        EnvType::setenv(env_var.c_str(), ss_val.str().c_str(), override);
    }

    template <typename Tp>
    static auto get_env_choice(std::string_view env_id, Tp value_default,
                               std::set<Tp> choices)
    {
        auto value = get_env(env_id, value_default);
        if(choices.find(value) == choices.end())
        {
            LOG_WARNING("[get_env] Environment variable \"{}\" has invalid value \"{}\". "
                        "Reverting to default.",
                        env_id, get_env<std::string>(env_id));
            return value_default;
        }
        return value;
    }
};

// Templated env_config — operator() calls EnvType::setenv so the backend is injectable.
template <typename EnvType = posix_env>
struct ROCPROFSYS_INTERNAL_API env_config
{
    std::string m_env_name  = {};
    std::string m_env_value = {};
    int         m_override  = 0;

    auto operator()() const
    {
        if(m_env_name.empty()) return -1;
        LOG_DEBUG("setenv(\"{}\", \"{}\", {})", m_env_name, m_env_value, m_override);
        return EnvType::setenv(m_env_name.c_str(), m_env_value.c_str(), m_override);
    }
};

// ── Forwarding free functions ────────────────────────────────────────────────
// These preserve the existing rocprofsys::common::get_env / set_env / get_env_choice
// API so no call sites need to change. They delegate to environment<posix_env>.

template <typename Tp>
inline auto
get_env(std::string_view env_id, Tp&& value_default)
{
    return environment<>::get_env(env_id, std::forward<Tp>(value_default));
}

template <typename Tp = std::string>
inline auto
get_env(std::string_view env_id)
{
    return environment<>::get_env<Tp>(env_id);
}

template <typename Tp>
inline void
set_env(const std::string& env_var, const Tp& value, int override)
{
    environment<>::set_env(env_var, value, override);
}

template <typename Tp>
inline auto
get_env_choice(std::string_view env_id, Tp value_default, std::set<Tp> value_choices)
{
    return environment<>::get_env_choice(env_id, value_default, std::move(value_choices));
}

// ── Env-vector helpers (operate on std::vector<std::string>, not the real env) ──

inline void
remove_env(std::vector<std::string>& env_list, std::string_view env_variable,
           const std::unordered_set<std::string>& original_envs)
{
    auto key = join("", env_variable, "=");

    env_list.erase(std::remove_if(env_list.begin(), env_list.end(),
                                  [&key](const std::string& entry) {
                                      return std::string_view{ entry }.find(key) == 0;
                                  }),
                   env_list.end());

    // Restore from original_envs if previously existed
    for(const auto& orig : original_envs)
    {
        if(std::string_view{ orig }.find(key) == 0) env_list.emplace_back(orig);
    }
}

inline std::string
discover_llvm_libdir_for_ompt()
{
    auto strip = [](std::string value_to_strip) {
        if(!value_to_strip.empty() && value_to_strip.back() == '/')
        {
            value_to_strip.pop_back();
        }
        return value_to_strip;
    };

    // Common ROCm envs
    const auto rocm_dir  = strip(get_env<std::string>("ROCM_PATH", "/opt/rocm"));
    const auto rocmv_dir = strip(get_env<std::string>("ROCmVersion_DIR", ""));

    const constexpr auto number_of_candidates = 6;

    std::vector<std::string> candidates;
    candidates.reserve(number_of_candidates);

    auto push_unique = [&](const std::string& candidate) {
        if(candidate.empty()) return;
        if(std::find(candidates.begin(), candidates.end(), candidate) == candidates.end())
        {
            candidates.emplace_back(candidate);
        }
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
    auto result = std::find_if(candidates.begin(), candidates.end(), has_libomptarget);
    if(result != candidates.end())
    {
        LOG_DEBUG("Using LLVM libdir: {}", *result);
        return *result;
    }

    LOG_DEBUG("libomptarget.so not found in candidate LLVM libdirs");
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
discover_torch_libpath(const std::string& python_binary)
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
        LOG_WARNING("Unsafe characters detected in Python interpreter path: {}",
                    python_binary);
        return {};
    }

    const auto cmd = "\"" + python_binary +
                     "\" -c \"import torch; print(torch.__path__[0])\" 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if(!pipe)
    {
        LOG_WARNING("Failed to execute command: {}", cmd);
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
        LOG_DEBUG("torch not found for Python interpreter: {}", python_binary);
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
        LOG_WARNING("torch lib directory does not exist: {}", torch_libdir);
        return {};
    }

    LOG_DEBUG("Discovered torch library path: {}", torch_libdir);
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
                       UpdatedEnvsT& updated_envs)
{
    if(executable.empty()) return;
    if(!is_python_interpreter(executable)) return;

    auto torch_libpath = discover_torch_libpath(std::string{ executable });
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
        {
            total_parts_length += part.size();
        }

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
