/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "environment.h"
#include "sys.h"

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

optional<size_t>
Environment::async_buffer_size()
{
    return Environment::get<size_t>(Environment::ASYNC_BUFFER_SIZE);
}
