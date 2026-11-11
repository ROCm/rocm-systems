// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/counter_samples.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/trace_cache/storage_parser.hpp"

namespace rocprofsys::trace_cache
{

using storage_parser_t =
    storage_parser<type_identifier_t, kernel_dispatch_sample, memory_copy_sample,
                   memory_allocate_sample, region_sample, in_time_sample,
                   pmc_event_with_sample, backtrace_region_sample, scratch_memory_sample,
                   kfd_sample, counter_batch, gap_marker>;

}  // namespace rocprofsys::trace_cache
