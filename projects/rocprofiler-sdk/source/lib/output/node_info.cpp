// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/output/node_info.hpp"
#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <sys/utsname.h>

#include <fstream>

namespace rocprofiler
{
namespace tool
{
using utsname_t = struct utsname;

node_info&
read_node_info(node_info& _info)
{
    {
        if(auto ifs = std::ifstream{"/etc/machine-id"})
        {
            auto _mach_id = std::string{};
            if((ifs >> _mach_id) && !_mach_id.empty())
                _info.machine_id = sdk::parse::strip(std::move(_mach_id), "\n\t\r ");
        }
    }

    auto _sys_info = utsname_t{};
    if(uname(&_sys_info) == 0)
    {
        auto _assign = [](auto& _dst, const char* _src) {
            if(_src) _dst = std::string{_src};
        };

        _assign(_info.system_name, _sys_info.sysname);
        _assign(_info.hostname, _sys_info.nodename);
        _assign(_info.release, _sys_info.release);
        _assign(_info.version, _sys_info.version);
        _assign(_info.hardware_name, _sys_info.machine);
        _assign(_info.domain_name, _sys_info.domainname);
    }
    else
    {
        ROCP_WARNING << "error retrieving uname info";
    }

    return _info;
}

node_info
read_node_info()
{
    auto _val = node_info{};
    return read_node_info(_val);
}
}  // namespace tool
}  // namespace rocprofiler
