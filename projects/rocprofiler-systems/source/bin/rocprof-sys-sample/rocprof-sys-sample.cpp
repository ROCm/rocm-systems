// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/tool_runner.hpp"

int
main(int argc, char** argv)
{
    return rocprofsys::common_utils::run_tool(
        argc, argv, rocprofsys::common_utils::tool_mode::sample);
}
