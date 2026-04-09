// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "att_lib_wrapper.hpp"

#include <map>
#include <vector>
#include "util.hpp"

namespace rocprofiler
{
namespace att_wrapper
{
class FilenameMgr
{
public:
    struct Coord
    {
        int  se{0};
        int  sm{0};
        int  sl{0};
        int  id{0};
        bool operator==(const Coord& other) const
        {
            return se == other.se && sm == other.sm && sl == other.sl && id == other.id;
        }
        bool operator<(const Coord& other) const
        {
            if(se != other.se) return se < other.se;
            if(sm != other.sm) return sm < other.sm;
            if(sl != other.sl) return sl < other.sl;
            return id < other.id;
        }
    };
    struct WaveName
    {
        std::string name{};
        size_t      begin{0};
        size_t      end{0};
    };

    FilenameMgr(const Fspath& _dir)
    : dir(_dir)
    , filename(_dir / "filenames.json")
    {}
    ~FilenameMgr();

    void addwave(const Fspath& file, Coord coord, size_t start, size_t end);

    Fspath                    dir{};
    Fspath                    filename{};
    std::map<Coord, WaveName> streams{};
    std::vector<std::string>  perfcounters{};
    int                       gfxip = 9;
};

}  // namespace att_wrapper
}  // namespace rocprofiler
