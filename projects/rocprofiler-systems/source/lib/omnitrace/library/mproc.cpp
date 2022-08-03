// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/mproc.hpp"
#include "library/common.hpp"
#include "library/debug.hpp"

#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>

namespace omnitrace
{
namespace mproc
{
std::set<int>
get_concurrent_processes(int _ppid)
{
    std::set<int> _children = {};
    if(_ppid > 0)
    {
        auto          _inp = JOIN('/', "/proc", _ppid, "task", _ppid, "children");
        std::ifstream _ifs{ _inp };
        if(!_ifs)
        {
            OMNITRACE_VERBOSE_F(2, "Warning! File '%s' cannot be read\n", _inp.c_str());
            return _children;
        }

        while(_ifs)
        {
            int _v = -1;
            _ifs >> _v;
            if(!_ifs.good() || _ifs.eof()) break;
            if(_v < 0) continue;
            _children.emplace(_v);
        }
    }
    return _children;
}

int
get_process_index(int _pid, int _ppid)
{
    auto _children = get_concurrent_processes(_ppid);
    for(auto itr = _children.begin(); itr != _children.end(); ++itr)
    {
        if(*itr == _pid) return std::distance(_children.begin(), itr);
    }
    return -1;
}
}  // namespace mproc
}  // namespace omnitrace
