// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <optional>

namespace rocprofsys::domains::callback
{

template <policies::domain_service::externals Externals>
inline void
on_code_object_configure()
{}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_code_object_enter(typename SdkBackend::callback_tracing_record_t record,
                     typename SdkBackend::user_data_t* user_data, void* callback_data)
{
    (void) record;
    (void) user_data;
    (void) callback_data;
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_code_object_exit(typename SdkBackend::callback_tracing_record_t record,
                    typename SdkBackend::user_data_t* user_data, void* callback_data)
{
    (void) record;
    (void) user_data;
    (void) callback_data;
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_code_object = callback_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "code_object",
            .id    = SdkBackend::CALLBACK_TRACING_CODE_OBJECT,
            .mode  = collection_mode::callback,
            .group = std::nullopt,
        },
    .on_record =
        tracing_callback_dispatcher<SdkBackend,
                                    on_code_object_enter<SdkBackend, Externals>,
                                    on_code_object_exit<SdkBackend, Externals>>::callback,
    .on_configure = on_code_object_configure<Externals>
};

}  // namespace rocprofsys::domains::callback
