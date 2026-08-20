// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <vector>

namespace roctx_range_intercept
{

void                     start_recording();
std::vector<std::string> stop_recording();

}  // namespace roctx_range_intercept
