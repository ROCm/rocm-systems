// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocprofsys::common_utils
{

enum class tool_mode : std::uint8_t
{
    run,
    sample
};

[[nodiscard]] int
run_tool(int argc, char** argv, tool_mode mode);

}  // namespace rocprofsys::common_utils
