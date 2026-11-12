// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "category_class.hpp"

#include <array>

namespace profiler_hub::pytorch
{
namespace
{
using reader_types::event_type_t;

constexpr category_class_t k_host{ false, false, event_type_t::region, {}, std::nullopt };
constexpr category_class_t k_launch_site{ false,
                                          true,
                                          event_type_t::region,
                                          {},
                                          std::nullopt };
constexpr category_class_t k_kernel{ true,
                                     false,
                                     event_type_t::kernel_dispatch,
                                     {},
                                     std::nullopt };
constexpr category_class_t k_copy{ true,
                                   false,
                                   event_type_t::memory_copy,
                                   {},
                                   std::nullopt };
constexpr category_class_t k_device_other{ true, false, std::nullopt, {}, std::nullopt };
// MTIA device work: a DMA request moves data, every other name is taken as a kernel.
constexpr category_class_t k_mtia_device{ true,
                                          false,
                                          event_type_t::kernel_dispatch,
                                          "dma_request",
                                          event_type_t::memory_copy };
// CUDA synchronisation on a stream; waiting on an event is the one HIP event analogue.
constexpr category_class_t k_sync{ true,
                                   false,
                                   std::nullopt,
                                   "Stream Wait Event",
                                   event_type_t::hip_event };

struct row_t
{
    std::string_view name;
    category_class_t cls;
};

constexpr std::array<row_t, 15> k_rows{ {
    { "cpu_op", k_host },
    { "python_function", k_host },
    { "cuda_runtime", k_launch_site },
    { "cuda_driver", k_launch_site },
    { "kernel", k_kernel },
    { "gpu_memcpy", k_copy },
    { "gpu_memset", k_device_other },
    { "gpu_user_annotation", k_device_other },
    { "cuda_sync", k_sync },
    { "mtia_ccp_events", k_mtia_device },
    { "mtia_runtime", k_launch_site },
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

reader_types::event_type_t
event_type_of(const category_class_t& cls, std::string_view name)
{
    if(cls.named_type && name == cls.named_event) return *cls.named_type;
    return cls.event_type.value_or(reader_types::event_type_t::region);
}
}  // namespace profiler_hub::pytorch
