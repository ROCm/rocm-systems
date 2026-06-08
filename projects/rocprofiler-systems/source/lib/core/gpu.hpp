// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocprofsys
{
namespace gpu
{
int
device_count();

void
add_device_metadata();
}  // namespace gpu
}  // namespace rocprofsys
