// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "csv.h"

namespace rocprofiler_compute_tool::csv
{

std::string quote(std::string_view field)
{
    std::string quoted;
    quoted.reserve(field.size() + 2);
    quoted.push_back('"');
    for (char c : field)
    {
        if (c == '"')
            quoted.push_back('"');
        quoted.push_back(c);
    }
    quoted.push_back('"');
    return quoted;
}

}  // namespace rocprofiler_compute_tool::csv
