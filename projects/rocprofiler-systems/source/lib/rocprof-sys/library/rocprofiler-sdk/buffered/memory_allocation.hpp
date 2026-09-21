// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"
#include "logger/debug.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>

namespace rocprofsys::domains::buffered
{

template <policies::domain_service::externals Externals>
inline void
on_memory_allocation_configure()
{}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_memory_allocation(typename SdkBackend::memory_allocation_record_t* record, void* data)
{}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_memory_allocation = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "memory_allocation",
            .id    = SdkBackend::BUFFER_TRACING_MEMORY_ALLOCATION,
            .mode  = collection_mode::buffered,
            .group = std::nullopt,
        },
    .on_records = buffered_callback_dispatcher<
        SdkBackend, typename SdkBackend::memory_allocation_record_t,
        on_memory_allocation<SdkBackend, Externals>>::callback,
    .on_configure = on_memory_allocation_configure<Externals>
};

}  // namespace rocprofsys::domains::buffered
