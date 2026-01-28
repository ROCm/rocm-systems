// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

namespace rocstorage
{
enum class storage_type_t
{
    auto_detect = 0,
    in_memory   = 1,
    on_disk     = 2,
    mmap        = 3,
};

struct version_t
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};
}  // namespace rocstorage
