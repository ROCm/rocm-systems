// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp"
#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <optional>
#include <string_view>

namespace rocprofsys::domains::callback
{

template <typename Externals>
struct hipfile_api_category
{
    using type = Externals::rocm_hipfile_api_category;

    static constexpr std::string_view k_name = Externals::rocm_hipfile_api_category_name;
};

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_hipfile_api = callback_domain_definition<SdkBackend>{
    .meta      = domain_descriptor{ .name  = "hipfile_api",
                                    .id    = SdkBackend::CALLBACK_TRACING_HIPFILE_API,
                                    .mode  = collection_mode::callback,
                                    .group = std::nullopt },
    .on_record = tracing_callback_dispatcher<
        SdkBackend, on_tracing_api_enter<SdkBackend, Externals, hipfile_api_category>,
        on_tracing_api_exit<SdkBackend, Externals, hipfile_api_category>>::callback,
    .on_configure = on_tracing_api_configure<Externals>
};

}  // namespace rocprofsys::domains::callback
