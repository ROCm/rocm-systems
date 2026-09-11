// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service.hpp"

#include <optional>

namespace rocprofsys::domains::callback
{

template <policies::rocprofiler_sdk::domain_service_externals Externals>
inline void
on_code_object_configure()
{}

template <policies::rocprofiler_sdk::domain_service_backend   SdkBackend,
          policies::rocprofiler_sdk::domain_service_externals Externals>
inline void
on_code_object(typename SdkBackend::callback_tracing_record_t record,
               typename SdkBackend::user_data_t* user_data, void* callback_data)
{
    (void) record;
    (void) user_data;
    (void) callback_data;
}

template <policies::rocprofiler_sdk::domain_service_backend   SdkBackend,
          policies::rocprofiler_sdk::domain_service_externals Externals>
inline constexpr auto k_code_object = callback_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "code_object",
            .id    = SdkBackend::CALLBACK_TRACING_CODE_OBJECT,
            .mode  = collection_mode::callback,
            .group = std::nullopt,
        },
    .on_record    = on_code_object<SdkBackend, Externals>,
    .on_configure = on_code_object_configure<Externals>
};

}  // namespace rocprofsys::domains::callback
