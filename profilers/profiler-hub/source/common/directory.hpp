// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cerrno>
#include <cstddef>
#include <libgen.h>
#include <string>
#include <sys/stat.h>

namespace profiler_hub::common
{

inline std::string
dirname(const std::string& path)
{
    std::string path_copy = path;
    return ::dirname(path_copy.data());
}

inline bool
direxists(const std::string& path)
{
    struct stat info
    {};
    if(stat(path.c_str(), &info) != 0) return false;
    return (info.st_mode & S_IFDIR) != 0;
}

inline bool
makedir(const std::string& path)
{
    if(path.empty()) return false;
    if(direxists(path)) return true;

    size_t pos    = 0;
    bool   status = true;
    do
    {
        pos         = path.find_first_of("/\\", pos + 1);
        auto subdir = path.substr(0, pos);
        if(subdir.empty()) continue;
        if(!direxists(subdir))
        {
            if(mkdir(subdir.c_str(), 0755) != 0 && errno != EEXIST)
            {
                status = false;
                break;
            }
        }
    } while(pos != std::string::npos);

    if(!direxists(path))
    {
        if(mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) status = false;
    }
    return status;
}

}  // namespace profiler_hub::common
