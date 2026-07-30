/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "configuration.h"
#include "environment.h"
#include "hip.h"

#include <cstdio>
#include <optional>

using namespace hipFile;

bool
Configuration::fastpath() const noexcept
{
    static bool fastpath_env{!Environment::force_compat_mode().value_or(false)};
    static bool readExists{!!getHipAmdFileReadPtr()};
    static bool writeExists{!!getHipAmdFileWritePtr()};
    return readExists && writeExists && m_fastpath_override.value_or(fastpath_env);
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
