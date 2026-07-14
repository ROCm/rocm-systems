// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

namespace roctx_recordfn::detail
{

struct StackEntry
{
    std::string marker;
    std::string context;
};

}  // namespace roctx_recordfn::detail
