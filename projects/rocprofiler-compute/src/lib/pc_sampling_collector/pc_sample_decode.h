// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef ROCPROFILER_SDK_EXPERIMENTAL
#    define ROCPROFILER_SDK_EXPERIMENTAL
#endif

#include "pc_sample_types.h"

#include <rocprofiler-sdk/fwd.h>

#include <optional>

namespace rocprofiler_compute_tool
{
// Returns std::nullopt when header.category != ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING
// or header.kind is not one of the two supported PC sampling record kinds.
std::optional<pc_sample_record_t> decode_pc_sample_record(const rocprofiler_record_header_t& header);

// Returns std::nullopt when header.category != ROCPROFILER_BUFFER_CATEGORY_TRACING
// or header.kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH.
std::optional<kernel_dispatch_record_t> decode_kernel_dispatch_record(const rocprofiler_record_header_t& header);
}  // namespace rocprofiler_compute_tool
