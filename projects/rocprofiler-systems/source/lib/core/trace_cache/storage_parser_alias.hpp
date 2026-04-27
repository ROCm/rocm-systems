// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/sample_type.hpp"
#include "core/trace_cache/storage_parser.hpp"
#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/gpu/sample.hpp"
#include "library/pmc/collectors/nic/sample.hpp"

namespace rocprofsys::trace_cache
{

using storage_parser_t =
    storage_parser<type_identifier_t, kernel_dispatch_sample, memory_copy_sample,
                   memory_allocate_sample, region_sample, in_time_sample,
                   pmc_event_with_sample, pmc::collectors::gpu::sample,
                   pmc::collectors::nic::sample, pmc::collectors::cpu::sample,
                   backtrace_region_sample, scratch_memory_sample, kfd_sample>;

}  // namespace rocprofsys::trace_cache
