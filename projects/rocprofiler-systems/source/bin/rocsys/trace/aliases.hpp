// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <string_view>

namespace rocprofsys::rocsys::trace
{

/// @brief A friendly `--traces` name. @ref domains is what `ROCPROFSYS_ROCM_DOMAINS`
///        receives. An empty slot means the name expands to one domain. `hip` is the
///        runtime and compiler API; legacy `--rocm=hip` is the runtime API only.
struct friendly_name
{
    std::string_view                name    = {};
    std::array<std::string_view, 2> domains = {};
    std::string_view                what    = {};
};

inline constexpr friendly_name k_friendly_names[] = {
    { .name = "hip", .domains = { "hip_api" }, .what = "HIP runtime and compiler API" },
    { .name = "hsa", .domains = { "hsa_api" }, .what = "HSA API" },
    { .name = "roctx", .domains = { "marker_api" }, .what = "ROCTX markers" },
    { .name = "marker", .domains = { "marker_api" }, .what = "ROCTX markers" },
    { .name = "kernel", .domains = { "kernel_dispatch" }, .what = "Kernel dispatches" },
    { .name    = "memory",
      .domains = { "memory_copy", "memory_allocation" },
      .what    = "Memory copies and allocations" },
    { .name = "scratch", .domains = { "scratch_memory" }, .what = "Scratch memory" },
    { .name = "kfd", .domains = { "kfd_events" }, .what = "KFD events" },
    { .name = "rocshmem", .domains = { "rocshmem_api" }, .what = "ROCSHMEM API" },
    { .name = "rocdecode", .domains = { "rocdecode_api" }, .what = "rocDecode API" },
    { .name = "rocjpeg", .domains = { "rocjpeg_api" }, .what = "rocJPEG API" },
    { .name = "hipfile", .domains = { "hipfile_api" }, .what = "hipFile API" },
};

}  // namespace rocprofsys::rocsys::trace
