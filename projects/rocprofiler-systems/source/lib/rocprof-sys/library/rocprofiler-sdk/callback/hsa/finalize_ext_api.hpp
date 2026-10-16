// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp"
#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <string_view>

namespace rocprofsys::domains::callback::hsa
{

template <typename Externals>
struct finalize_ext_api_category
{
    using type = Externals::rocm_hsa_api_category;

    static constexpr std::string_view k_name = Externals::rocm_hsa_api_category_name;
};

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_finalize_ext_api = callback_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "hsa_finalize_ext_api",
            .id    = SdkBackend::CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
            .mode  = collection_mode::callback,
            .group = domain_group{ .name = "hsa_api" },
        },
    .on_record = tracing_callback_dispatcher<
        SdkBackend,
        on_tracing_api_enter<SdkBackend, Externals, finalize_ext_api_category>,
        on_tracing_api_exit<SdkBackend, Externals, finalize_ext_api_category>>::callback,
    .on_configure = on_tracing_api_configure<Externals>
};

}  // namespace rocprofsys::domains::callback::hsa
