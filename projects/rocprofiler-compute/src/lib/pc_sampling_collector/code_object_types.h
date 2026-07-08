// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rocprofiler_compute_tool
{
struct symbol_t
{
    std::string name{};
    uint64_t    code_object_offset = 0;
    uint64_t    virtual_address    = 0;
    uint64_t    size               = 0;
};

struct kernel_t
{
    uint64_t    kernel_id{};
    std::string name{};
};

struct instruction_t
{
    std::string name{};
    std::string comment{};
    uint64_t    virtual_address = 0;
    uint64_t    code_obj_offset = 0;
    size_t      size{0};
};
}  // namespace rocprofiler_compute_tool
