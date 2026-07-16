// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Defines RCCL_API_TRACE_VERSION_PATCH, which gates the set of RCCL API ids
#include <rocprofiler-sdk/rccl/details/api_trace.h>

#include <cstdint>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
void
rccl_comm_data_initialize();

void
tool_tracing_callback_rccl(std::uint32_t                                 operation,
                           rocprofiler_callback_tracing_rccl_api_data_t* payload,
                           std::uint64_t begin_ts, std::uint64_t end_ts);

}  // namespace rocprofiler_sdk

}  // namespace rocprofsys
