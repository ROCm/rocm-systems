/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "configuration.h"
#include "context.h"
#include "environment.h"
#include "hip.h"
#include "hipfile-literals.h"
#include "sys.h"

#include <cstdio>
#include <fcntl.h>
#include <optional>
#include <string_view>
#include <system_error>

using namespace hipFile;

/* On WSL2, ROCr drives the GPU through librocdxg (/dev/dxg) instead of KFD.
 * librocdxg does not implement AIS, so fastpath I/O fails with an error that
 * is not eligible for fallback.
 */
static bool
rocrUsesDxg() noexcept
{
    const char *detection{Context<Sys>::get()->getenv("HSA_ENABLE_DXG_DETECTION")};
    if (detection && std::string_view{detection} == "0") {
        return false;
    }

    try {
        Context<Sys>::get()->statx(AT_FDCWD, "/dev/dxg", 0, 0);
    }
    catch (const std::system_error &e) {
        // Only ENOENT/ENOTDIR prove /dev/dxg does not exist, i.e. ROCr is using KFD (allow fastpath).
        if (e.code() == std::errc::no_such_file_or_directory || e.code() == std::errc::not_a_directory) {
            return false;
        }

        // Any other error is unexpected and leaves us unable to tell, so fail safe, i.e. assume dxg
        // (disable fastpath).
        // TODO: replace with logging, e.g., hipfilelog(WARN, ...)
        fprintf(stderr,
                "hipFile: unable to check for /dev/dxg (%s); assuming ROCr is using dxg and disabling "
                "the fastpath.\n",
                e.code().message().c_str());
    }
    return true;
}

bool
Configuration::fastpath() const noexcept
{
    static bool fastpath_env{!Environment::force_compat_mode().value_or(false)};
    static bool readExists{!!getHipAmdFileReadPtr()};
    static bool writeExists{!!getHipAmdFileWritePtr()};
    static bool usesDxg{rocrUsesDxg()};
    return readExists && writeExists && !usesDxg && m_fastpath_override.value_or(fastpath_env);
}

void
Configuration::fastpath(bool enabled) noexcept
{
    m_fastpath_override = enabled;
}

bool
Configuration::fallback() const noexcept
{
    static bool fallback_env{[] {
        bool force_compat = Environment::force_compat_mode().value_or(false);
        bool allow_compat = Environment::allow_compat_mode().value_or(true);
        if (force_compat && !allow_compat) {
            // TODO: replace with logging
            fprintf(stderr, "hipFile: HIPFILE_FORCE_COMPAT_MODE=true and HIPFILE_ALLOW_COMPAT_MODE=false "
                            "would disable all I/O backends; enabling the fallback path to avoid "
                            "failing all I/O.\n");
            return true;
        }
        return allow_compat;
    }()};
    return m_fallback_override.value_or(fallback_env);
}

void
Configuration::fallback(bool enabled) noexcept
{
    m_fallback_override = enabled;
}

unsigned int
Configuration::statsLevel() const noexcept
{
    static unsigned int stats_level_env{Environment::stats_level().value_or(1)};
    return stats_level_env;
}

bool
Configuration::unsupportedFileSystems() const noexcept
{
    static bool unsupported_file_systems_env{Environment::unsupported_file_systems().value_or(false)};
    return unsupported_file_systems_env;
}

size_t
Configuration::asyncBufferSize() const noexcept
{
    static size_t async_buffer_size_env{[] {
        constexpr size_t default_size{16_MiB};
        auto             env = Environment::async_buffer_size();
        return (env && *env > 0) ? *env : default_size;
    }()};
    return async_buffer_size_env;
}

bool
Configuration::host() const noexcept
{
    static bool host_env{Environment::host().value_or(true)};
    return m_host_override.value_or(host_env);
}

void
Configuration::host(bool enabled) noexcept
{
    m_host_override = enabled;
}

size_t
Configuration::deviceStreamPoolSize() const noexcept
{
    static size_t device_stream_pool_size_env{[] {
        constexpr size_t default_size{4};
        auto             env = Environment::device_stream_pool_size();
        return (env && *env > 0) ? *env : default_size;
    }()};
    return device_stream_pool_size_env;
}
