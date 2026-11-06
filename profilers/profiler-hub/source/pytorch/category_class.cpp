// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "category_class.hpp"

#include <array>

namespace profiler_hub::pytorch
{
namespace
{
using reader_types::event_type_t;

constexpr category_class_t k_host_in_tree{ false, true, false, event_type_t::region };
constexpr category_class_t k_launch_site{ false, true, true, event_type_t::region };
constexpr category_class_t k_kernel{ true, false, false, event_type_t::kernel_dispatch };
constexpr category_class_t k_copy{ true, false, false, event_type_t::memory_copy };
constexpr category_class_t k_device_other{ true, false, false, std::nullopt };

struct row_t
{
    std::string_view name;
    category_class_t cls;
};

constexpr std::array<row_t, 12> k_rows{ {
    { "cpu_op", k_host_in_tree },
    { "python_function", k_host_in_tree },
    { "cuda_runtime", k_launch_site },
    { "cuda_driver", k_launch_site },
    { "kernel", k_kernel },
    { "gpu_memcpy", k_copy },
    { "gpu_memset", k_device_other },
    { "gpu_user_annotation", k_device_other },
    // Older Kineto spellings. TraceLens does not map these; here they take their
    // modern row.
    { "Runtime", k_launch_site },
    { "Kernel", k_kernel },
    { "Memcpy", k_copy },
    { "Memset", k_device_other },
} };
}  // namespace

category_class_t
classify_category(std::string_view category)
{
    for(const auto& row : k_rows)
    {
        if(row.name == category)
        {
            return row.cls;
        }
    }
    return {};
}
}  // namespace profiler_hub::pytorch
