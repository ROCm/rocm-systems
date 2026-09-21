// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/traces.hpp"

#if defined(ROCPROFSYS_AVAIL_SDK_BACKEND) && ROCPROFSYS_AVAIL_SDK_BACKEND == 1

#    include "backends/rocprofiler_sdk/backend.hpp"
#    include "backends/rocprofiler_sdk/wrapper.hpp"

#    include <string>
#    include <vector>

namespace rocprofsys::avail
{
namespace
{
using wrapper_t = rocprofsys::rocprofiler_sdk::wrapper;
using backend_t = backends::rocprofiler_sdk::backend<wrapper_t>;

template <typename Table>
[[nodiscard]] std::vector<trace_kind_entry>
domain_entries(const Table& table)
{
    auto entries = std::vector<trace_kind_entry>{};
    entries.reserve(table.size());
    for(const auto& info : table)
    {
        if(info.name.empty()) continue;
        auto entry  = trace_kind_entry{};
        entry.name  = std::string{ info.name };
        entry.operations.reserve(info.operations.size());
        for(const auto& operation : info.operations)
        {
            if(!is_listed_operation(operation)) continue;
            entry.operations.emplace_back(std::string{ operation });
        }
        entries.push_back(std::move(entry));
    }
    return entries;
}

struct sdk_traces
{
    [[nodiscard]] std::vector<trace_kind_entry> callback_domains() const
    {
        return domain_entries(backend_t::get_callback_tracing_names());
    }

    [[nodiscard]] std::vector<trace_kind_entry> buffered_domains() const
    {
        return domain_entries(backend_t::get_buffer_tracing_names());
    }
};
}  // namespace

trace_query_result
query_traces()
{
    auto source = sdk_traces{};
    return collect_traces(source);
}
}  // namespace rocprofsys::avail

#else  // ROCPROFSYS_AVAIL_SDK_BACKEND

namespace rocprofsys::avail
{
trace_query_result
query_traces()
{
    auto result = trace_query_result{};
    result.diagnostics.push_back(
        { source_id::rocprofiler_sdk,
          "trace catalog unavailable: built without rocprofiler-sdk support "
          "(requires ROCm 6.4 or later)" });
    return result;
}
}  // namespace rocprofsys::avail

#endif  // ROCPROFSYS_AVAIL_SDK_BACKEND
