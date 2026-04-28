// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys::sampling
{

struct inline_frame
{
    std::string name;      // function name (demangled)
    std::string location;  // file:line
    int         line = 0;
};

struct stack_frame
{
    std::string               name;  // demangled symbol; empty if unresolved
    std::string               name_mangled;
    uintptr_t                 address      = 0;
    uintptr_t                 line_address = 0;
    std::string               location;  // file:line
    std::vector<inline_frame> inlines;
};

}  // namespace rocprofsys::sampling
