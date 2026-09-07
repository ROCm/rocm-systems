// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/backend.hpp"
#include "backends/rocprofiler_sdk/wrapper.hpp"

#include "library/rocprofiler-sdk/types.hpp"

#include "library/rocprofiler-sdk/buffered/kfd_event_dropped_events.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_event_page_fault.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_event_page_migrate.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_event_queue.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_event_unmap_from_gpu.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_page_fault.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_page_migrate.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_queue.hpp"

#include <array>

namespace rocprofsys::domains::buffered
{
using production_backend = backends::rocprofiler_sdk::backend<rocprofiler_sdk::wrapper>;

struct kfd_externals
{};

/// Every buffered domain this build supports, gated on the rocprofiler-sdk version.
inline constexpr std::array k_buffered_domains_definitions{
    k_kfd_event_dropped_events<production_backend, kfd_externals>,
    k_kfd_event_page_fault<production_backend, kfd_externals>,
    k_kfd_event_page_migrate<production_backend, kfd_externals>,
    k_kfd_event_queue<production_backend, kfd_externals>,
    k_kfd_event_unmap_from_gpu<production_backend, kfd_externals>,
    k_kfd_page_fault<production_backend, kfd_externals>,
    k_kfd_page_migrate<production_backend, kfd_externals>,
    k_kfd_queue<production_backend, kfd_externals>,
};

}  // namespace rocprofsys::domains::buffered
