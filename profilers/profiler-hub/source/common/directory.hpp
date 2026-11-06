// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace profiler_hub::common
{

inline std::string
dirname(const std::string& path)
{
    // parent_path() is empty for a bare filename; POSIX dirname returns ".".
    const auto parent = std::filesystem::path(path).parent_path().string();
    return parent.empty() ? std::string{ "." } : parent;
}

inline bool
direxists(const std::string& path)
{
    try
    {
        std::error_code ec;
        return std::filesystem::is_directory(path, ec);
    } catch(const std::filesystem::filesystem_error&)
    {
        return false;
    }
}

inline bool
makedir(const std::string& path)
{
    if(path.empty())
    {
        return false;
    }

    if(direxists(path))
    {
        return true;
    }

    namespace fs = std::filesystem;

    // Collect ancestors we create so we can tighten their mode after creation.
    // Stop when parent_path() equals p: on Windows a missing drive root such as
    // Z:/ is its own parent, so ascending forever would hang and grow created.
    std::vector<fs::path> created;
    for(fs::path p = path; !p.empty();)
    {
        std::error_code exists_ec;
        if(fs::exists(p, exists_ec))
        {
            break;
        }
        created.push_back(p);
        const fs::path parent = p.parent_path();
        if(parent == p)
        {
            break;
        }
        p = parent;
    }

    std::error_code ec;
    fs::create_directories(path, ec);
    if(ec && !direxists(path))
    {
        return false;
    }

#if !defined(_WIN32)
    // Keep the mode create_directories chose (it already applied umask).
    // Only strip group/other write; do not force 0755, which would make
    // private dirs (umask 0077 to 0700) world-readable again.
    constexpr auto drop_write = fs::perms::group_write | fs::perms::others_write;
    for(const auto& p : created)
    {
        fs::permissions(p, drop_write, fs::perm_options::remove, ec);
    }
#endif

    return true;
}

}  // namespace profiler_hub::common
