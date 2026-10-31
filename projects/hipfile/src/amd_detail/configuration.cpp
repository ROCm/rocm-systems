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

    int fd;
    try {
        fd = Context<Sys>::get()->open("/dev/dxg", O_RDWR | O_CLOEXEC);
    }
    catch (const std::system_error &) {
        return false;
    }
    try {
        Context<Sys>::get()->close(fd);
    }
    catch (const std::system_error &) {
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
