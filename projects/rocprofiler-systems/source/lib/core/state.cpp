// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "state.hpp"
#include "config.hpp"

namespace rocprofsys
{
bool
config_policy::get_debug_init()
{
    return config::get_debug_init();
}
}  // namespace rocprofsys
