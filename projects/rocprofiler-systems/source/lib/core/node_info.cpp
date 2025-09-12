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

#include "node_info.hpp"
#include "debug.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sys/utsname.h>

namespace rocprofsys
{

node_info::node_info()
{
    auto ifs = std::ifstream{ "/etc/machine-id" };
    if(!ifs.is_open())
    {
        ROCPROFSYS_WARNING(0, "Error: Unable to open /etc/machine-id!");
        return;
    }
    if(!(ifs >> machine_id) || machine_id.empty())
    {
        ROCPROFSYS_WARNING(0, "Error: Unable to read machine ID from /etc/machine-id!");
    }

    hash = std::hash<std::string>{}(machine_id) % std::numeric_limits<int64_t>::max();
    id   = hash % std::numeric_limits<size_t>::max();

    struct utsname _sys_info;
    if(uname(&_sys_info))
    {
        ROCPROFSYS_WARNING(0, "Error: Unable to get system information!");
        return;
    }

    system_name = _sys_info.sysname;
    node_name   = _sys_info.nodename;
    release     = _sys_info.release;
    version     = _sys_info.version;
    machine     = _sys_info.machine;
    domain_name = _sys_info.domainname;
}

node_info&
node_info::get_instance()
{
    static node_info instance;
    return instance;
}

}  // namespace rocprofsys
