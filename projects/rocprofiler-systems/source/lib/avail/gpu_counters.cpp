// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/gpu_counters.hpp"
#include <cstdint>

#if defined(ROCPROFSYS_AVAIL_SDK_BACKEND) && ROCPROFSYS_AVAIL_SDK_BACKEND == 1

#    include "backends/rocprofiler_sdk/backend.hpp"
#    include "backends/rocprofiler_sdk/wrapper.hpp"

#    include <fmt/format.h>

#    include <cstddef>
#    include <cstdint>
#    include <stdexcept>
#    include <string>
#    include <utility>
#    include <vector>

namespace rocprofsys::avail
{
namespace
{
using wrapper_t = rocprofsys::rocprofiler_sdk::wrapper;
using backend_t = backends::rocprofiler_sdk::backend<wrapper_t>;

wrapper_t::status_t
collect_counter_ids(wrapper_t::agent_id /*agent*/, wrapper_t::counter_id* counters,
                    std::size_t count, void* user_data)
{
    auto* ids = static_cast<std::vector<std::uint64_t>*>(user_data);
    ids->reserve(ids->size() + count);
    for(std::size_t i = 0; i < count; ++i)
        ids->push_back(counters[i].handle);
    return wrapper_t::STATUS_SUCCESS;
}

[[nodiscard]] std::string
safe_string(const char* value)
{
    return value != nullptr ? std::string{ value } : std::string{};
}

/// Reads the counter catalog through the rocprofiler-sdk backend. Listing is
/// per agent, describing is per counter, matching the SDK's own split.
struct sdk_counters
{
    [[nodiscard]] std::vector<std::uint64_t> counter_ids(std::uint64_t agent_handle) const
    {
        auto ids    = std::vector<std::uint64_t>{};
        auto status = backend_t::iterate_agent_supported_counters(
            backend_t::make_agent_id(agent_handle), collect_counter_ids, &ids);

        if(status != backend_t::status_success)
        {
            throw std::runtime_error{ fmt::format(
                "could not list counters for agent {:#x}: {}", agent_handle,
                safe_string(backend_t::get_status_string(status))) };
        }
        return ids;
    }

    [[nodiscard]] std::vector<counter_record> describe(std::uint64_t counter_id) const
    {
        auto metadata =
            backend_t::query_counter_details(wrapper_t::counter_id{ counter_id });
        auto records = std::vector<counter_record>{};
        records.reserve(metadata.size());

        for(auto& info : metadata)
        {
            auto record        = counter_record{};
            record.id          = info.counter_id;
            record.name        = std::move(info.name);
            record.description = std::move(info.description);
            record.block       = std::move(info.block);
            record.expression  = std::move(info.expression);
            record.is_constant = info.is_constant;
            record.is_derived  = info.is_derived;
            record.dimensions.reserve(info.dimensions.size());
            for(auto& dimension : info.dimensions)
            {
                record.dimensions.push_back(
                    { std::move(dimension.name), dimension.position });
            }
            records.push_back(std::move(record));
        }

        return records;
    }
};
}  // namespace

counter_query_result
query_gpu_counters(const std::vector<device_record>& devices)
{
    auto source = sdk_counters{};
    return collect_gpu_counters(source, devices);
}
}  // namespace rocprofsys::avail

#else  // ROCPROFSYS_AVAIL_SDK_BACKEND

namespace rocprofsys::avail
{
counter_query_result
query_gpu_counters(const std::vector<device_record>& /*devices*/)
{
    auto result = counter_query_result{};
    result.diagnostics.push_back(
        { source_id::rocprofiler_sdk,
          "counter catalog unavailable: built without rocprofiler-sdk support "
          "(requires ROCm 6.4 or later)" });
    return result;
}
}  // namespace rocprofsys::avail

#endif  // ROCPROFSYS_AVAIL_SDK_BACKEND
