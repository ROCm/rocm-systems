// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/defines.hpp"
#include "core/timemory.hpp"

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0
#    include <rocprofiler-sdk/rocprofiler.h>
#    include <rocprofiler-sdk/version.h>
#    if __has_include(<rocprofiler-sdk/experimental/registration.h>)
#        include <rocprofiler-sdk/experimental/registration.h>
#    else
#        include <rocprofiler-sdk/registration.h>
#    endif
#endif

#include <cstdint>
#include <vector>

namespace rocprofsys
{
namespace rocm
{
using hardware_counter_info = ::tim::hardware_counters::info;

std::vector<hardware_counter_info>
rocm_events();

#if !defined(ROCPROFSYS_USE_ROCM) || ROCPROFSYS_USE_ROCM == 0
inline std::vector<hardware_counter_info>
rocm_events()
{
    return std::vector<hardware_counter_info>();
}
#endif
}  // namespace rocm
}  // namespace rocprofsys

extern "C"
{
    struct rocprofiler_tool_configure_result_t;
    struct rocprofiler_client_id_t;

    using rocprofiler_configure_t =
        rocprofiler_tool_configure_result_t* (*) (uint32_t    version,
                                                  const char* runtime_version,
                                                  uint32_t    priority,
                                                  rocprofiler_client_id_t* client_id);

    rocprofiler_tool_configure_result_t* rocprofiler_configure(
        uint32_t version, const char* runtime_version, uint32_t priority,
        rocprofiler_client_id_t* client_id) ROCPROFSYS_PUBLIC_API;
#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0 &&                           \
    ROCPROFILER_VERSION >= 10200
    rocprofiler_tool_configure_attach_result_t* rocprofiler_configure_attach(
        uint32_t version, const char* runtime_version, uint32_t priority,
        rocprofiler_client_id_t* client_id) ROCPROFSYS_PUBLIC_API;
#endif
}
