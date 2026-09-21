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
on_kernel_dispatch_configure()
{}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_kernel_dispatch(typename SdkBackend::kernel_dispatch_record_t* record, void* data)
{}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_kernel_dispatch = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "kernel_dispatch",
            .id    = SdkBackend::BUFFER_TRACING_KERNEL_DISPATCH,
            .mode  = collection_mode::buffered,
            .group = std::nullopt,
        },
    .on_records =
        buffered_callback_dispatcher<SdkBackend,
                                     typename SdkBackend::kernel_dispatch_record_t,
                                     on_kernel_dispatch<SdkBackend, Externals>>::callback,
    .on_configure = on_kernel_dispatch_configure<Externals>
};

}  // namespace rocprofsys::domains::buffered
