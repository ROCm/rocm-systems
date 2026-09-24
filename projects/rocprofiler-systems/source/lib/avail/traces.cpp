// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/traces.hpp"
#include "avail/records.hpp"

#include <exception>
#include <string>
#include <string_view>
#include <utility>

#if defined(ROCPROFSYS_AVAIL_HAS_SDK)
#    include "avail/listing_externals.hpp"
#    include "backends/rocprofiler_sdk/backend.hpp"
#    include "backends/rocprofiler_sdk/wrapper.hpp"
#    include "core/sdk/tracing-config.hpp"
#endif

namespace rocprofsys::avail
{
namespace
{

diagnostic
sdk_diagnostic(capability_kind capability, std::string message)
{
    return diagnostic{
        .capability = capability,
        .source     = source_id::rocprofiler_sdk,
        .message    = std::move(message),
    };
}

}  // namespace

traces_listing_result
query_traces()
{
#if defined(ROCPROFSYS_AVAIL_HAS_SDK)
    using production_tracing_config = rocprofiler_sdk::tracing_config<
        backends::rocprofiler_sdk::backend<rocprofiler_sdk::wrapper>, listing_externals>;

    try
    {
        return inventory::traces<production_tracing_config>();
    } catch(const std::exception& err)
    {
        return traces_listing_result{
            .records  = {},
            .defaults = {},
            .issue    = sdk_diagnostic(capability_kind::traces, err.what()),
        };
    }
#else
    return traces_listing_result{
        .records  = {},
        .defaults = {},
        .issue    = sdk_diagnostic(capability_kind::traces,
                                   std::string{ k_sdk_unavailable_message }),
    };
#endif
}

operations_listing_result
query_operations(std::string_view name)
{
#if defined(ROCPROFSYS_AVAIL_HAS_SDK)
    using production_tracing_config = rocprofiler_sdk::tracing_config<
        backends::rocprofiler_sdk::backend<rocprofiler_sdk::wrapper>, listing_externals>;

    try
    {
        return inventory::operations<production_tracing_config>(name);
    } catch(const std::exception& err)
    {
        return operations_listing_result{
            .records = {},
            .issue   = sdk_diagnostic(capability_kind::trace_operations, err.what()),
        };
    }
#else
    (void) name;
    return operations_listing_result{
        .records = {},
        .issue   = sdk_diagnostic(capability_kind::trace_operations,
                                  std::string{ k_sdk_unavailable_message }),
    };
#endif
}

}  // namespace rocprofsys::avail
