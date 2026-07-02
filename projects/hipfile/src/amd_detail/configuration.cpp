/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "configuration.h"
#include "environment.h"
#include "hip.h"

#include <cstdio>
#include <mutex>
#include <optional>

using namespace hipFile;

bool
Configuration::fastpath() const noexcept
{
    std::call_once(m_fastpath_once, [this] {
        m_fastpath_env          = !Environment::force_compat_mode().value_or(false);
        m_fastpath_read_exists  = !!getHipAmdFileReadPtr();
        m_fastpath_write_exists = !!getHipAmdFileWritePtr();
    });
    return m_fastpath_read_exists && m_fastpath_write_exists && m_fastpath_override.value_or(m_fastpath_env);
}

void
Configuration::fastpath(bool enabled) noexcept
{
    m_fastpath_override = enabled;
}

bool
Configuration::fallback() const noexcept
{
    std::call_once(m_fallback_once, [this] {
        bool force_compat = Environment::force_compat_mode().value_or(false);
        bool allow_compat = Environment::allow_compat_mode().value_or(true);
        if (force_compat && !allow_compat) {
            // TODO: replace with logging
            fprintf(stderr, "hipFile: HIPFILE_FORCE_COMPAT_MODE=true and HIPFILE_ALLOW_COMPAT_MODE=false "
                            "would disable all I/O backends; enabling the fallback path to avoid "
                            "failing all I/O.\n");
            m_fallback_env = true;
        }
        else {
            m_fallback_env = allow_compat;
        }
    });
    return m_fallback_override.value_or(m_fallback_env);
}

void
Configuration::fallback(bool enabled) noexcept
{
    m_fallback_override = enabled;
}

unsigned int
Configuration::statsLevel() const noexcept
{
    std::call_once(m_stats_once, [this] { m_stats_level = Environment::stats_level().value_or(1); });
    return m_stats_level;
}

bool
Configuration::unsupportedFileSystems() const noexcept
{
    std::call_once(m_fs_once, [this] {
        m_unsupported_file_systems = Environment::unsupported_file_systems().value_or(false);
    });
    return m_unsupported_file_systems;
}
