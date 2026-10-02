// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <optional>

namespace rocprofsys::domains::callback
{

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_code_object_enter(typename SdkBackend::callback_tracing_record_t     record,
                     [[maybe_unused]] typename SdkBackend::user_data_t* user_data,
                     [[maybe_unused]] void*                             callback_data)
{
    if(record.operation == SdkBackend::CODE_OBJECT_LOAD)
    {
        auto data_v = *static_cast<SdkBackend::code_object_load_data_t*>(record.payload);

        Externals::get_metadata_registry().add_code_object(data_v);
    }
    else if(record.operation == SdkBackend::CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto data_v =
            *static_cast<SdkBackend::code_object_kernel_symbol_register_data_t*>(
                record.payload);
        Externals::get_metadata_registry().add_kernel_symbol(data_v);
    }
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
    .on_record = tracing_callback_dispatcher<
        SdkBackend, on_code_object_enter<SdkBackend, Externals>>::callback,
};

}  // namespace rocprofsys::domains::callback
