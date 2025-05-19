// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace tool
{
constexpr uint32_t lds_block_size = 128 * 4;

using counter_dimension_id_vec_t   = std::vector<rocprofiler_counter_dimension_id_t>;
using counter_dimension_info_vec_t = std::vector<rocprofiler_counter_record_dimension_info_t>;

struct tool_counter_info : rocprofiler_counter_info_v1_t
{
    using parent_type = rocprofiler_counter_info_v1_t;

    tool_counter_info(rocprofiler_agent_id_t         _agent_id,
                      parent_type                    _info,
                      counter_dimension_id_vec_t&&   _dim_ids,
                      counter_dimension_info_vec_t&& _dim_info)
    : parent_type{_info}
    , agent_id{_agent_id}
    , dimension_ids{std::move(_dim_ids)}
    , dimensions{std::move(_dim_info)}
    {}

    ~tool_counter_info()                            = default;
    tool_counter_info(const tool_counter_info&)     = default;
    tool_counter_info(tool_counter_info&&) noexcept = default;
    tool_counter_info& operator=(const tool_counter_info&) = default;
    tool_counter_info& operator=(tool_counter_info&&) noexcept = default;

    rocprofiler_agent_id_t       agent_id      = {};
    counter_dimension_id_vec_t   dimension_ids = {};
    counter_dimension_info_vec_t dimensions    = {};
};

using counter_info_vec_t       = std::vector<tool_counter_info>;
using agent_counter_info_map_t = std::unordered_map<rocprofiler_agent_id_t, counter_info_vec_t>;

struct tool_counter_value_t
{
    rocprofiler_counter_id_t id    = {};
    double                   value = 0;

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("counter_id", id));
        ar(cereal::make_nvp("value", value));
    }
};

struct serialized_counter_record_t
{
    std::optional<std::streampos> fpos = std::nullopt;
};

struct tool_counter_record_t
{
    using container_type = std::vector<tool_counter_value_t>;

    uint64_t                                     thread_id     = 0;
    rocprofiler_dispatch_counting_service_data_t dispatch_data = {};
    serialized_counter_record_t                  record        = {};
    rocprofiler_stream_id_t                      stream_id     = {.handle = 0};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        // should be removed when moving to buffered tracing
        auto tmp = read();

        ar(cereal::make_nvp("thread_id", thread_id));
        ar(cereal::make_nvp("dispatch_data", dispatch_data));
        ar(cereal::make_nvp("records", tmp));
        ar(cereal::make_nvp("stream_id", stream_id));
    }

    container_type read() const;
    void           write(const container_type& data);
};
}  // namespace tool
}  // namespace rocprofiler

namespace cereal
{
#define SAVE_DATA_FIELD(FIELD) ar(make_nvp(#FIELD, data.FIELD))

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::tool_counter_info& data)
{
    SAVE_DATA_FIELD(agent_id);
    cereal::save(ar, static_cast<const rocprofiler_counter_info_v1_t&>(data));
}

#undef SAVE_DATA_FIELD
}  // namespace cereal
