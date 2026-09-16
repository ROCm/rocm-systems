// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

namespace rocprofsys::domains::callback::hip
{

template <policies::domain_service::externals Externals>
inline void
on_compiler_api_configure()
{}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_compiler_api(typename SdkBackend::callback_tracing_record_t record,
                typename SdkBackend::user_data_t* user_data, void* callback_data)
{
    (void) record;
    (void) user_data;
    (void) callback_data;
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_compiler_api = callback_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "compiler_api",
            .id    = SdkBackend::CALLBACK_TRACING_HIP_COMPILER_API,
            .mode  = collection_mode::callback,
            .group = domain_group{ .name = "hip_api" },
        },
    .on_record    = on_compiler_api<SdkBackend, Externals>,
    .on_configure = on_compiler_api_configure<Externals>
};

}  // namespace rocprofsys::domains::callback::hip
