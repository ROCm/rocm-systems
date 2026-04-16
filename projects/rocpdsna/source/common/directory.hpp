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

#include <cerrno>
#include <cstddef>
#include <libgen.h>
#include <string>
#include <sys/stat.h>

namespace rocpdsna::common
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

}  // namespace rocpdsna::common
