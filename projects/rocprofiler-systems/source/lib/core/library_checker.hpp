// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/version.hpp"

#include <string_view>

namespace rocprofsys::library_checker
{

/**
 * @brief Compare the version a library was built against with the version loaded at
 * runtime.
 *
 * @throws std::runtime_error on a mismatch.
 *
 * @param library_name Name used in the message.
 * @param built Version the headers were configured against.
 * @param loaded Version reported by the library at runtime.
 */
void
check_version(std::string_view library_name, const version& built, const version& loaded);

/**
 * @brief Compare the profiler-hub version this build was configured against with the
 * version of the profiler-hub library loaded at runtime.
 *
 * @throws std::runtime_error on a mismatch.
 */
void
check_profiler_hub_version_mismatch();

/**
 * @brief Compare the rocprofiler-sdk version this build was configured against with the
 * version of the rocprofiler-sdk library loaded at runtime.
 *
 * @throws std::runtime_error on a mismatch.
 *
 * @tparam Backend rocprofiler-sdk backend exposing `compile_time_version` and
 * `get_version(major*, minor*, patch*)`.
 */
template <typename Backend>
void
check_rocprofiler_sdk_version_mismatch()
{
    auto loaded = version{};
    Backend::get_version(&loaded.major, &loaded.minor, &loaded.patch);

    check_version("rocprofiler-sdk",
                  version::from_formatted(Backend::compile_time_version), loaded);
}

}  // namespace rocprofsys::library_checker
