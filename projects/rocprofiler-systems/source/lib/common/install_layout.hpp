// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// rocprofiler-systems install-layout discovery: locate the install root and the
// internal lib/script directories relative to the running executable.
// Split out of the path.hpp god-header; consumes the core path ops.

#include "common/defines.h"
#include "common/path.hpp"

#include <spdlog/fmt/fmt.h>

#include <string>

namespace rocprofsys
{
inline namespace common
{
namespace path
{
inline std::string
get_rocprofsys_root() ROCPROFSYS_INTERNAL_API;

inline std::string
get_internal_libpath(const std::string& _lib) ROCPROFSYS_INTERNAL_API;

inline std::string
get_internal_script_path() ROCPROFSYS_INTERNAL_API;

inline std::string
get_internal_libdir() ROCPROFSYS_INTERNAL_API;

//--------------------------------------------------------------------------------------//

inline std::string
get_rocprofsys_root()
{
    auto _exe_rp  = realpath("/proc/self/exe");
    auto _exe_dir = dirname(_exe_rp);
    if(_exe_dir.empty()) _exe_dir = "./";
    return fmt::format("{}/{}", _exe_dir, "..");
}

inline std::string
get_internal_libpath(const std::string& _lib)
{
    auto _root = get_rocprofsys_root();
    for(const auto* libdir : { "lib", "lib64" })
    {
        auto _candidate = fmt::format("{}/{}/{}", _root, libdir, _lib);
        if(exists(_candidate)) return _candidate;
    }
    return fmt::format("{}/lib/{}", _root, _lib);
}

inline std::string
get_internal_script_path()
{
    auto _root = get_rocprofsys_root();
    return _root + "/libexec/rocprofiler-systems";
}

inline std::string
get_internal_libdir()
{
    return get_rocprofsys_root() + "/lib";
}

}  // namespace path
}  // namespace common
}  // namespace rocprofsys
