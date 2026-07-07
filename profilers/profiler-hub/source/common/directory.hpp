// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace profiler_hub::common
{

inline std::string
dirname(const std::string& path)
{
    return std::filesystem::path(path).parent_path().string();
}

inline bool
direxists(const std::string& path)
{
    return std::filesystem::is_directory(path);
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

    // Collect the ancestors that don't exist yet so we can apply 0755 to each
    // one we create, matching the previous mkdir(..., 0755) behavior on POSIX.
    std::vector<fs::path> created;
    for(fs::path p = path; !p.empty() && !fs::exists(p); p = p.parent_path())
    {
        created.push_back(p);
    }

    std::error_code ec;
    fs::create_directories(path, ec);
    if(ec && !direxists(path))
    {
        return false;
    }

    constexpr auto mode = fs::perms::owner_all | fs::perms::group_read |
                          fs::perms::group_exec | fs::perms::others_read |
                          fs::perms::others_exec;
    for(const auto& p : created)
    {
        fs::permissions(p, mode, ec);
    }

    return true;
}

}  // namespace profiler_hub::common
