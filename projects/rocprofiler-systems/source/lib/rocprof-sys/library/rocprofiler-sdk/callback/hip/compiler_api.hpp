// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp"
#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <string_view>

namespace rocprofsys::domains::callback::hip
{

template <typename Externals>
struct compiler_api_category
{
    using type = Externals::rocm_hip_api_category;

    static constexpr std::string_view k_name = Externals::rocm_hip_api_category_name;
};

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_compiler_api = callback_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "hip_compiler_api",
            .id    = SdkBackend::CALLBACK_TRACING_HIP_COMPILER_API,
            .mode  = collection_mode::callback,
            .group = domain_group{ .name = "hip_api" },
        },
    .on_record = tracing_callback_dispatcher<
        SdkBackend, on_tracing_api_enter<SdkBackend, Externals, compiler_api_category>,
        on_tracing_api_exit<SdkBackend, Externals, compiler_api_category>>::callback,
    .on_configure = on_tracing_api_configure<Externals>
};

}  // namespace rocprofsys::domains::callback::hip
