// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <vector>

namespace roctx_recordfn::detail
{

// Start and stop recording of the emitted wire strings.
void                     start_capture();
std::vector<std::string> stop_capture();

}  // namespace roctx_recordfn::detail
