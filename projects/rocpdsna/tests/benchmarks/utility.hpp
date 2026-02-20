// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
