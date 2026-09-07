// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"

namespace rocprofsys::domains::callback
{

template <typename SdkBackend, typename Externals>
inline void
on_code_object(typename SdkBackend::callback_tracing_record_t record,
               typename SdkBackend::user_data_t* user_data, void* callback_data)
{}

template <typename SdkBackend, typename Externals>
inline constexpr auto k_code_object =
    callback_domain_definition{ .meta =
                                    domain_descriptor{
                                        .name  = "code_object",
                                        .id    = SdkBackend::CALLBACK_TRACING_CODE_OBJECT,
                                        .mode  = collection_mode::callback,
                                        .group = std::nullopt,
                                    },
                                .on_record = on_code_object<SdkBackend, Externals> };

}  // namespace rocprofsys::domains::callback
