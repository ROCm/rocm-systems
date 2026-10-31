// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace rocprofsys::rocsys
{

/// @brief A usage or launch failure. Main prints `error:` and, when set, `hint:`.
struct diagnostic
{
    std::string error = {};
    std::string hint  = {};
};

}  // namespace rocprofsys::rocsys
