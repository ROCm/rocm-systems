// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library_checker.hpp"
#include "common/defines.h"
#include "common/version.hpp"
#include "logger/debug.hpp"

#include <profiler-hub/storage.hpp>

#include <cstdlib>

namespace rocprofsys::library_checker
{

void
check_profiler_hub_version_mismatch()
{
    // the storage is never opened here; its constructor only stores the arguments
    const auto loaded = profiler_hub::storage_t{ "", "" }.get_storage_version();

    const auto mismatch = describe_version_mismatch(
        "profiler-hub",
        version{ .major = ROCPROFSYS_PROFILER_HUB_VERSION_MAJOR,
                 .minor = ROCPROFSYS_PROFILER_HUB_VERSION_MINOR,
                 .patch = ROCPROFSYS_PROFILER_HUB_VERSION_PATCH },
        version{ .major = loaded.major, .minor = loaded.minor, .patch = loaded.patch });

    if(mismatch)
    {
        LOG_CRITICAL("{}. Check LD_LIBRARY_PATH and the installed profiler-hub. "
                     "Aborting.",
                     *mismatch);
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace rocprofsys::library_checker
