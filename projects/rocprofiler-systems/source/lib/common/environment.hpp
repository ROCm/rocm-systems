// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "common/env_vars.hpp"
#include "common/join.hpp"
#include "logger/debug.hpp"

#include <timemory/utility/filepath.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <set>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>  // POSIX ::setenv / ::getenv
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

/// @brief Production environment backend that forwards to the real POSIX
///        ::getenv / ::setenv. Used as the default @p EnvType of @ref environment;
///        unit tests substitute a fake backend to avoid touching the real process
///        environment.
struct posix_env
{
    /// @brief Forwards to ::setenv.
    /// @param name      Null-terminated variable name.
    /// @param value     Null-terminated value to store.
    /// @param overwrite Non-zero to replace an existing value; zero to keep it.
    /// @return 0 on success, -1 on error.
    static int setenv(const char* name, const char* value, int overwrite)
    {
        return ::setenv(name, value, overwrite);
    }

    /// @brief Forwards to ::getenv.
    /// @param name Null-terminated variable name.
    /// @return Pointer to the value, or nullptr when the variable is unset.
    static char* getenv(const char* name) { return ::getenv(name); }
};

/// @brief Environment variable read/write facade, parameterised over the backend.
///
/// All conversion and parsing logic lives here. Use @c environment<posix_env> (the
/// default) in production; inject a fake backend in unit tests. The free functions
/// @ref get_env / @ref set_env / @ref get_env_choice delegate to @c environment<>.
///
/// @tparam EnvType Backend providing static @c getenv / @c setenv.
template <typename EnvType = posix_env>
struct environment
{
private:
    static const char* fetch_raw_env(const char* env_id)
    {
        if(env_id == nullptr || env_id[0] == '\0') return nullptr;
        return EnvType::getenv(env_id);
    }

    template <typename Tp>
    static std::string get_env_string(const char* env_id, const Tp& fallback)
    {
        const char* raw = fetch_raw_env(env_id);
        return raw ? std::string{ raw } : std::string{ fallback };
    }

    static bool get_env_bool(const char* env_id, bool fallback)
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
            // Parse with from_chars so a very large all-digit value cannot throw
            // (std::stoi would throw std::out_of_range). Any non-zero digit string,
            // including one that overflows, is truthy.
            std::uint64_t numeric{};
            const auto*   last   = env_sv.data() + env_sv.size();
            const auto [ptr, ec] = std::from_chars(env_sv.data(), last, numeric);
            if(ec == std::errc::result_out_of_range) return true;
            if(ec == std::errc{} && ptr == last) return numeric != 0;
            return true;
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
    static Tp get_env_float(const char* env_id, Tp fallback)
    {
        const char* raw = fetch_raw_env(env_id);
        if(!raw) return fallback;

        // Trim surrounding whitespace so values such as " 1.5 " still parse.
        constexpr std::string_view whitespace = " \t\n\r\f\v";
        std::string_view           token{ raw };
        const auto                 first = token.find_first_not_of(whitespace);
        if(first == std::string_view::npos)
        {
            LOG_ERROR("[get_env] Cannot convert empty getenv(\"{}\") to float", env_id);
            return fallback;
        }
        token = token.substr(first, token.find_last_not_of(whitespace) - first + 1);

#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
        // Locale-independent, non-throwing parse mirroring the integral path: a
        // trailing-garbage or unparsable value falls back to the default.
        Tp          value{};
        const auto* end      = token.data() + token.size();
        const auto [ptr, ec] = std::from_chars(token.data(), end, value);
        if(ec == std::errc{} && ptr == end) return value;
#else
        // Fallback for standard libraries without floating-point from_chars
        // (libstdc++ < 11). std::stod is locale-sensitive (assumes C/POSIX locale).
        try
        {
            std::size_t       pos = 0;
            const std::string str{ token };
            const double      parsed = std::stod(str, &pos);
            if(pos == str.size()) return static_cast<Tp>(parsed);
        } catch(const std::exception&)
        {}
#endif
        LOG_ERROR("[get_env] Failed to convert getenv(\"{}\") = \"{}\" to float", env_id,
                  raw);
        return fallback;
    }

    template <typename Tp>
    static Tp get_env_integral(const char* env_id, Tp fallback)
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
    static auto get_env_impl(const char* env_id, const Tp& fallback)
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
    /// @brief Read environment variable @p env_id, converted to the type of
    ///        @p value_default.
    ///
    /// The conversion is selected from @p value_default's type: std::string /
    /// const char* return the raw value, bool is parsed permissively
    /// (1/true/yes/on are true; 0/false/no/n/f/off are false), floating-point via
    /// std::stod, and integral types via std::from_chars so that negative or
    /// out-of-range input for the target type yields @p value_default instead of
    /// wrapping. Enums are read through their underlying integral type.
    ///
    /// Primary, allocation-free overload: @p env_id must be a null-terminated C
    /// string (literal, const char*, or constexpr const char*).
    /// @param env_id        Null-terminated variable name.
    /// @param value_default Returned when the variable is unset or unparsable.
    /// @return The parsed value, or @p value_default.
    /// @throws std::runtime_error when a bool is requested but the value is empty.
    template <typename Tp>
    static auto get_env(const char* env_id, Tp&& value_default)
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

    /// @brief Read environment variable @p env_id as @p Tp, using a
    ///        value-initialised @c Tp{} as the fallback.
    /// @tparam Tp Target type (defaults to std::string).
    /// @param env_id Null-terminated variable name.
    /// @return The parsed value, or @c Tp{} when the variable is unset.
    template <typename Tp = std::string>
    static auto get_env(const char* env_id)
    {
        return get_env(env_id, Tp{});
    }

    /// @brief Set environment variable @p env_var to the stringified @p value.
    ///
    /// @p value is rendered via operator<< before being stored. Primary,
    /// allocation-free overload: @p env_var must be null-terminated.
    /// @param env_var  Null-terminated variable name.
    /// @param value    Value to store (stringified).
    /// @param override Non-zero to replace an existing value; zero to keep it.
    template <typename Tp>
    static void set_env(const char* env_var, const Tp& value, int override)
    {
        std::stringstream ss_val;
        ss_val << value;
        EnvType::setenv(env_var, ss_val.str().c_str(), override);
    }

    /// @brief Read environment variable @p env_id constrained to a set of allowed
    ///        values.
    ///
    /// Reads @p env_id as @p Tp; if the result is not in @p choices, logs a warning
    /// and returns @p value_default. Primary, allocation-free overload (@p env_id
    /// must be null-terminated).
    /// @param env_id        Null-terminated variable name.
    /// @param value_default Returned when unset or not among @p choices.
    /// @param choices       Set of accepted values.
    /// @return The value when valid, otherwise @p value_default.
    template <typename Tp>
    static auto get_env_choice(const char* env_id, Tp value_default, std::set<Tp> choices)
    {
        auto value = get_env(env_id, value_default);
        if(choices.find(value) == choices.end())
        {
            const char* raw = fetch_raw_env(env_id);
            LOG_WARNING("[get_env] Environment variable \"{}\" has invalid value \"{}\". "
                        "Reverting to default.",
                        env_id, raw ? raw : "");
            return value_default;
        }
        return value;
    }
};

/// @brief Deferred @c setenv command: stores a name/value/override triple and
///        applies it when invoked. Templated on the backend so tests can inject a
///        fake.
template <typename EnvType = posix_env>
struct ROCPROFSYS_INTERNAL_API env_config
{
    std::string m_env_name  = {};
    std::string m_env_value = {};
    int         m_override  = 0;

    /// @brief Apply the stored setenv command.
    /// @return The backend setenv result, or -1 when @c m_env_name is empty.
    auto operator()() const
    {
        if(m_env_name.empty()) return -1;
        LOG_DEBUG("setenv(\"{}\", \"{}\", {})", m_env_name, m_env_value, m_override);
        return EnvType::setenv(m_env_name.c_str(), m_env_value.c_str(), m_override);
    }
};

// ── Forwarding free functions ────────────────────────────────────────────────
// These are the primary public API; they delegate to environment<posix_env>.

/// @brief Read environment variable @p env_id as the type of @p value_default.
///        Allocation-free overload for null-terminated names. See
///        @ref environment::get_env for conversion rules.
/// @param env_id        Null-terminated variable name.
/// @param value_default Returned when the variable is unset or unparsable.
/// @return The parsed value, or @p value_default.
template <typename Tp>
inline auto
get_env(const char* env_id, Tp&& value_default)
{
    return environment<>::get_env(env_id, std::forward<Tp>(value_default));
}

/// @brief Read environment variable @p env_id as @p Tp, falling back to @c Tp{}.
/// @tparam Tp Target type (defaults to std::string).
/// @param env_id Null-terminated variable name.
/// @return The parsed value, or @c Tp{} when unset.
template <typename Tp = std::string>
inline auto
get_env(const char* env_id)
{
    return environment<>::get_env<Tp>(env_id);
}

/// @brief Set environment variable @p env_var to the stringified @p value.
///        Allocation-free overload for null-terminated names.
/// @param env_var  Null-terminated variable name.
/// @param value    Value to store (stringified).
/// @param override Non-zero to replace an existing value; zero to keep it.
template <typename Tp>
inline void
set_env(const char* env_var, const Tp& value, int override)
{
    environment<>::set_env(env_var, value, override);
}

/// @brief Read environment variable @p env_id constrained to @p value_choices,
///        returning @p value_default when unset or not allowed. Allocation-free
///        overload for null-terminated names.
/// @param env_id        Null-terminated variable name.
/// @param value_default Returned when unset or not among @p value_choices.
/// @param value_choices Set of accepted values.
/// @return The value when valid, otherwise @p value_default.
template <typename Tp>
inline auto
get_env_choice(const char* env_id, Tp value_default, std::set<Tp> value_choices)
{
    return environment<>::get_env_choice(env_id, std::move(value_default),
                                         std::move(value_choices));
}

// ── Env-vector helpers (operate on std::vector<std::string>, not the real env) ──

/// @brief Remove all "KEY=VALUE" entries for @p env_variable from @p env_list,
///        then restore any entries for that key found in @p original_envs.
/// @param env_list      Environment vector to modify in place.
/// @param env_variable  Variable name (without '=') to remove.
/// @param original_envs Baseline entries used to restore a pre-existing value.
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

/// @brief Locate the ROCm LLVM library directory that contains libomptarget.so.
///
/// Probes candidates derived from @c ROCM_PATH and @c ROCmVersion_DIR plus the
/// standard /opt/rocm locations, returning the first that contains the library.
/// @return The matching libdir, or an empty string when none is found.
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

/// @brief Test whether @p executable names a Python interpreter.
/// @param executable Path or basename to inspect.
/// @return true for "python", "python3", or "python3.<digits>"; false otherwise.
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

/// @brief Discover the PyTorch library directory for a given Python interpreter.
///
/// Runs @p python_binary to query torch's install path (the path is validated for
/// safe characters first to avoid shell injection) and appends "/lib".
/// @param python_binary Path to the Python interpreter.
/// @return The torch lib directory, or an empty string when torch is unavailable,
///         the path is unsafe, or the directory does not exist.
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

/// @brief How @ref update_env combines a new value with an existing entry.
enum class update_mode : std::uint8_t
{
    REPLACE = 0,  ///< Overwrite the value and drop duplicate entries.
    PREPEND,      ///< Insert the new value before the existing one.
    APPEND,       ///< Insert the new value after the existing one.
    WEAK,         ///< Update only when the current entry matches the original env.
};

/// @brief Render @p val as an environment-variable string.
///        bool becomes "true"/"false"; arithmetic types use std::to_string;
///        strings pass through.
/// @return The string representation of @p val.
template <typename Tp>
    requires(std::is_same_v<std::decay_t<Tp>, std::string> ||
             std::is_same_v<std::decay_t<Tp>, const char*> ||
             std::is_same_v<std::decay_t<Tp>, bool> ||
             std::is_arithmetic_v<std::decay_t<Tp>>)
inline std::string
to_env_string(Tp&& val)
{
    using T = std::decay_t<Tp>;

    if constexpr(std::is_same_v<T, std::string> || std::is_same_v<T, const char*>)
        return std::string{ val };
    else if constexpr(std::is_same_v<T, bool>)
        return val ? "true" : "false";
    else
        return std::to_string(val);
}

/// @brief Insert or update an "KEY=VALUE" entry in an environment vector.
///
/// Adds a new entry when @p _env_var is absent; otherwise combines values per
/// @p _mode (see @ref update_mode). Records @p _env_var in @p _updated_envs.
/// @param _environ      Environment vector to modify in place.
/// @param _env_var      Variable name (without '=').
/// @param _env_val      New value (stringified via @ref to_env_string).
/// @param _mode         Combination strategy.
/// @param _join_delim   Delimiter used when prepending/appending.
/// @param _updated_envs Set receiving the names touched by this call.
/// @param _original_envs Baseline entries consulted by @ref update_mode::WEAK.
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

/// @brief Prepend the PyTorch library directory to @c LD_LIBRARY_PATH in @p envp.
///
/// No-op unless @p executable is a Python interpreter with torch installed. Merges
/// the discovered torch lib dir with existing @c LD_LIBRARY_PATH entries
/// (deduplicated) and records the touched name in @p updated_envs.
/// @param envp         Environment vector to modify in place.
/// @param executable   Interpreter path used to locate torch.
/// @param updated_envs Set receiving the names touched by this call.
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
        if(key == env_vars::PAPI_EVENTS || key == env_vars::SAMPLING_OVERFLOW_EVENT ||
           key == env_vars::ROCM_EVENTS)
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
