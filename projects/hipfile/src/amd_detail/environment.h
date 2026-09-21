/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "context.h"
#include "sys.h"

#include <charconv>
#include <concepts>
#include <cstdlib>
#include <limits>
#include <optional>
#include <strings.h>
#include <string_view>

namespace hipFile {

class Environment {
public:
    /// @brief Get a value from the environment
    /// @param key The name of the variable
    /// @return An optional value of type T if the key was set, nullopt if the key
    /// was not set or had an invalid value for the type T
    template <typename T> static std::optional<T> get(const char *key);

    template <typename T>
        requires std::integral<T> && (!std::same_as<T, bool>)
    static std::optional<T> get(const char *key)
    {
        const char *value{Context<Sys>::get()->getenv(key)};
        if (value == nullptr)
            return std::nullopt;
        std::string_view sv{value};
        T                x{};
        auto [ptr, ec]{std::from_chars(sv.data(), sv.data() + sv.size(), x, 10)};
        if (ec == std::errc::invalid_argument || *ptr != '\0')
            return std::nullopt;
        else if (ec == std::errc::result_out_of_range)
            return std::numeric_limits<T>::max();
        return x;
    }

    /// @brief Allow I/O operations to fallback to POSIX I/O APIs
    ///
    /// If enabled (default), I/O operations can use the POSIX I/O path if they
    /// cannot be completed by the AIS path.
    /// If disabled, I/O operations that cannot be completed by the AIS path
    /// will be rejected.
    static constexpr const char *const ALLOW_COMPAT_MODE{"HIPFILE_ALLOW_COMPAT_MODE"};

    /// @brief Get the value of HIPFILE_ALLOW_COMPAT_MODE from the environment
    /// @return An optional boolean value if HIPFILE_ALLOW_COMPAT_MODE was set,
    /// nullopt if HIPFILE_ALLOW_COMPAT_MODE was unset or had a value other than
    /// true or false.
    static std::optional<bool> allow_compat_mode();

    /// @brief Force I/O operations to use POSIX I/O APIs
    ///
    /// If enabled, I/O operations will be forced to use the POSIX I/O path.
    /// If disabled (default), I/O operations can use the AIS path if conditions
    /// are satisfied
    static constexpr const char *const FORCE_COMPAT_MODE{"HIPFILE_FORCE_COMPAT_MODE"};

    static std::optional<bool> force_compat_mode();

    /// @brief Control how much information is collected to be reported to ais-stats
    ///
    /// 0 indicates no stats should be recorded.
    /// >= 1 indicates basic stats will be collected.
    static constexpr const char *const STATS_LEVEL{"HIPFILE_STATS_LEVEL"};

    static std::optional<unsigned int> stats_level();

    /// @brief Allow unsupported file systems in the fastpath backend
    ///
    /// If enabled, the fastpath backend will allow I/O on file systems other than
    /// ext4 (with ordered journaling) and xfs. If disabled (default), only supported
    /// file systems are permitted.
    static constexpr const char *const UNSUPPORTED_FILE_SYSTEMS{"HIPFILE_UNSUPPORTED_FILE_SYSTEMS"};

    /// @brief Get the value of HIPFILE_UNSUPPORTED_FILE_SYSTEMS from the environment
    /// @return An optional boolean value if HIPFILE_UNSUPPORTED_FILE_SYSTEMS was set,
    /// nullopt if HIPFILE_UNSUPPORTED_FILE_SYSTEMS was unset or had a value other than
    /// true or false.
    static std::optional<bool> unsupported_file_systems();

    /// @brief Control the size of the host bounce buffer used for async fallback I/O operations
    ///
    /// The default value is 1 MiB.
    static constexpr const char *const ASYNC_BUFFER_SIZE{"HIPFILE_ASYNC_BUFFER_SIZE"};

    static std::optional<size_t> async_buffer_size();
};
}
