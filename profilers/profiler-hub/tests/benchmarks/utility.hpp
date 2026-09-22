// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <string>
#include <sys/stat.h>
namespace utility
{

inline std::string
format_file_size(size_t _bytes)
{
    constexpr double k_kb = 1024.0;
    constexpr double k_mb = k_kb * 1024.0;
    constexpr double k_gb = k_mb * 1024.0;

    char buffer[64];
    if(_bytes >= k_gb)
    {
        std::snprintf(buffer, sizeof(buffer), "%.2f GB", _bytes / k_gb);
    }
    else if(_bytes >= k_mb)
    {
        std::snprintf(buffer, sizeof(buffer), "%.2f MB", _bytes / k_mb);
    }
    else if(_bytes >= k_kb)
    {
        std::snprintf(buffer, sizeof(buffer), "%.2f KB", _bytes / k_kb);
    }
    else
    {
        std::snprintf(buffer, sizeof(buffer), "%zu B", _bytes);
    }
    return buffer;
}

inline size_t
get_file_size(const std::string& _path)
{
    struct stat st;
    if(stat(_path.c_str(), &st) == 0)
    {
        return static_cast<size_t>(st.st_size);
    }
    return 0;
}

}  // namespace utility
