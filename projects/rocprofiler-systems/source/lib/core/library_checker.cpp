// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library_checker.hpp"
#include "common/version.hpp"
#include "logger/debug.hpp"

#include <fmt/format.h>

#include <cstdlib>
#include <string>
#include <string_view>

namespace rocprofsys::library_checker
{

namespace
{

std::string
format_version_mismatch(std::string_view library_name, const version& built,
                        const version& loaded)
{
    return fmt::format("{} version mismatch: built against v{}.{}.{} but v{}.{}.{} is "
                       "loaded",
                       library_name, built.major, built.minor, built.patch, loaded.major,
                       loaded.minor, loaded.patch);
}

}  // namespace

void
check_version(std::string_view library_name, const version& built, const version& loaded)
{
    if(built == loaded)
    {
        return;
    }

    LOG_CRITICAL("{}. Check LD_LIBRARY_PATH and the installed {}. Aborting.",
                 format_version_mismatch(library_name, built, loaded), library_name);
    std::exit(EXIT_FAILURE);
}

}  // namespace rocprofsys::library_checker
