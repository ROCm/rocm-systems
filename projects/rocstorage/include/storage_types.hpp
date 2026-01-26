// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

namespace rocstorage
{
enum class database_type_t
{
    in_memory = 0,
    on_disk   = 1,
    mmap      = 2,
};

struct version_t
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};
}  // namespace rocstorage
