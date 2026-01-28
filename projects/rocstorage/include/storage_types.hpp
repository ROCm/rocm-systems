// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

namespace rocstorage
{

struct version_t
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};
}  // namespace rocstorage
