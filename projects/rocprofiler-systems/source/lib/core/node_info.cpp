// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "node_info.hpp"
#include <cstdint>

#include <fstream>
#include <iostream>
#include <limits>
#include <sys/utsname.h>

#include "logger/debug.hpp"

namespace rocprofsys
{

node_info::node_info()
{
    struct utsname sys_info;
    if(uname(&sys_info) != 0)
    {
        LOG_WARNING("Error: Unable to get system information!");
        return;
    }

    system_name = sys_info.sysname;
    node_name   = sys_info.nodename;
    release     = sys_info.release;
    version     = sys_info.version;
    machine     = sys_info.machine;
    domain_name = sys_info.domainname;

    auto ifs = std::ifstream{ "/etc/machine-id" };
    if(!ifs.is_open())
    {
        LOG_WARNING("Error: Unable to open /etc/machine-id!");
        return;
    }
    if(!(ifs >> machine_id) || machine_id.empty())
    {
        LOG_WARNING("Error: Unable to read machine ID from /etc/machine-id!");
    }

    hash =
        std::hash<std::string>{}(machine_id) % std::numeric_limits<std::int64_t>::max();
    id = hash % std::numeric_limits<size_t>::max();
}

node_info&
node_info::get_instance()
{
    static node_info instance;
    return instance;
}

}  // namespace rocprofsys
