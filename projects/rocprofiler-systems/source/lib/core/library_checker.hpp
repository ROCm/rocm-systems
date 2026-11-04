// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocprofsys::library_checker
{

/**
 * @brief Compare the profiler-hub version this build was configured against with the
 * version of the profiler-hub library loaded at runtime.
 *
 * Logs a critical message and exits the process with `EXIT_FAILURE` on a mismatch.
 */
void
check_profiler_hub_version_mismatch();

}  // namespace rocprofsys::library_checker
