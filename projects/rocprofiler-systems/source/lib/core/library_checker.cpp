// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library_checker.hpp"
#include "common/defines.h"
#include "common/version.hpp"

#include <profiler-hub/cpp/storage.hpp>

#include <fmt/format.h>

#include <stdexcept>
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

    throw std::runtime_error{ fmt::format(
        "{}. Check LD_LIBRARY_PATH and the installed {}.",
        format_version_mismatch(library_name, built, loaded), library_name) };
}

void
check_profiler_hub_version_mismatch()
{
    // the storage is never opened here; its constructor only stores the arguments
    const auto loaded = profiler_hub::storage_t{ "", "" }.get_storage_version();

    check_version(
        "profiler-hub",
        version{ .major = ROCPROFSYS_PROFILER_HUB_VERSION_MAJOR,
                 .minor = ROCPROFSYS_PROFILER_HUB_VERSION_MINOR,
                 .patch = ROCPROFSYS_PROFILER_HUB_VERSION_PATCH },
        version{ .major = loaded.major, .minor = loaded.minor, .patch = loaded.patch });
}

}  // namespace rocprofsys::library_checker
