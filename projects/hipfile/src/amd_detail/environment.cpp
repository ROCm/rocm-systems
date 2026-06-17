/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "environment.h"
#include "sys.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <strings.h>

using namespace hipFile;
using namespace std;

template <>
std::optional<bool>
Environment::get<bool>(const char *key)
{
    const char *value{Context<Sys>::get()->getenv(key)};
    if (value) {
        if (!strcasecmp(value, "true")) {
            return true;
        }
        if (!strcasecmp(value, "false")) {
            return false;
        }
    }
    return std::nullopt;
}

template <>
std::optional<unsigned int>
Environment::get<unsigned int>(const char *key)
{
    const char *value{Context<Sys>::get()->getenv(key)};
    if (value == nullptr)
        return std::nullopt;
    errno = 0;
    char         *end;
    unsigned long x{std::strtoul(value, &end, 10)};
    if (errno != 0)
        return std::nullopt;
    if (end == value)
        return std::nullopt;
    if (*end != '\0')
        return std::nullopt;
    if (x > static_cast<unsigned long>(std::numeric_limits<unsigned int>::max()))
        return std::numeric_limits<unsigned int>::max();
    return static_cast<unsigned int>(x);
}

optional<bool>
Environment::allow_compat_mode()
{
    return Environment::get<bool>(Environment::ALLOW_COMPAT_MODE);
}

optional<bool>
Environment::force_compat_mode()
{
    return Environment::get<bool>(Environment::FORCE_COMPAT_MODE);
}

optional<bool>
Environment::unsupported_file_systems()
{
    return Environment::get<bool>(Environment::UNSUPPORTED_FILE_SYSTEMS);
}

optional<unsigned int>
Environment::stats_level()
{
    return Environment::get<unsigned int>(Environment::STATS_LEVEL);
}
