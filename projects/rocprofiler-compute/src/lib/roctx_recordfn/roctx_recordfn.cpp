// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Definitions for the roctx_recordfn capture hooks.

#include "capture_buffer.h"
#include "capture_control.h"

#include <string>
#include <vector>

namespace roctx_recordfn::detail
{

void start_capture()
{
    g_capture.start();
}

std::vector<std::string> stop_capture()
{
    return g_capture.stop();
}

}  // namespace roctx_recordfn::detail
