/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "configuration.h"
#include "environment.h"
#include "hip.h"
#include "static.h"

#include <optional>

using namespace hipFile;

bool
Configuration::fastpath() const noexcept
{
    HIPFILE_STATIC bool fastpath_env{!Environment::force_compat_mode().value_or(false)};
    HIPFILE_STATIC bool readExists{!!getHipAmdFileReadPtr()};
    HIPFILE_STATIC bool writeExists{!!getHipAmdFileWritePtr()};
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
    HIPFILE_STATIC bool fallback_env{Environment::allow_compat_mode().value_or(true)};
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
    HIPFILE_STATIC unsigned int stats_level_env{Environment::stats_level().value_or(0)};
    return stats_level_env;
}

bool
Configuration::unsupportedFileSystems() const noexcept
{
    HIPFILE_STATIC bool unsupported_file_systems_env{Environment::unsupported_file_systems().value_or(false)};
    return unsupported_file_systems_env;
}
