// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

namespace rocprofsys::domains::callback::hsa
{

template <policies::domain_service::externals Externals>
inline void
on_amd_ext_api_configure()
{}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_amd_ext_api_enter(typename SdkBackend::callback_tracing_record_t record,
                     typename SdkBackend::user_data_t* user_data, void* callback_data)
{
    (void) record;
    (void) user_data;
    (void) callback_data;
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_amd_ext_api_exit(typename SdkBackend::callback_tracing_record_t record,
                    typename SdkBackend::user_data_t* user_data, void* callback_data)
{
    (void) record;
    (void) user_data;
    (void) callback_data;
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_amd_ext_api = callback_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "amd_ext_api",
            .id    = SdkBackend::CALLBACK_TRACING_HSA_AMD_EXT_API,
            .mode  = collection_mode::callback,
            .group = domain_group{ .name = "hsa_api" },
        },
    .on_record =
        tracing_callback_dispatcher<SdkBackend,
                                    on_amd_ext_api_enter<SdkBackend, Externals>,
                                    on_amd_ext_api_exit<SdkBackend, Externals>>::callback,
    .on_configure = on_amd_ext_api_configure<Externals>
};

}  // namespace rocprofsys::domains::callback::hsa
